# main.py — full app with secure signup/login, recipe upload, recipe retrieval, Airtable sync

import os
import re
import json
import base64
import uuid
import tempfile
import subprocess
from datetime import date
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse
from models import Recipe, Ingredient, RecipeDB, User, UserCreate, UserDB
from db import get_db
from openai import OpenAI
import requests

client = OpenAI()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_RECIPES_TABLE = "Recipes"
AIRTABLE_USERS_TABLE = "Users"

@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    db_user = UserDB(
        id=user_id,
        name=user.name,
        email=user.email,
        auth_provider="email",
        registration_date=date.today(),
        last_login=date.today(),
        uploaded_count=0,
        saved_count=0
    )
    db.add(db_user)
    db.commit()

    sync_user_to_airtable(db_user)

    return {"success": True, "user_id": user_id}

@app.get("/user-recipes")
def get_user_recipes(user_id: str, db: Session = Depends(get_db)):
    recipes = db.query(RecipeDB).filter(RecipeDB.user_id == user_id).all()
    return [
        Recipe(
            id=r.id,
            title=r.title,
            ingredients=json.loads(r.ingredients),
            steps=json.loads(r.steps),
            cook_time_minutes=r.cook_time_minutes,
            user_id=r.user_id
        ) for r in recipes
    ]

def sync_user_to_airtable(user: UserDB):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "fields": {
            "User ID": user.id,
            "Name": user.name,
            "Email": user.email,
            "Registration Date": str(user.registration_date),
            "Authentication Provider": user.auth_provider,
            "Last Login": str(user.last_login),
            "Number of Uploaded Recipes": user.uploaded_count,
            "Total Saved Recipes": user.saved_count,
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    r = requests.post(url, headers=headers, json=payload)
    print("Airtable user sync status:", r.status_code, r.text)

def sync_recipe_to_airtable(recipe: Recipe):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }

    users_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    params = {"filterByFormula": f'{{User ID}} = "{recipe.user_id}"'}
    user_response = requests.get(users_url, headers=headers, params=params)
    user_data = user_response.json()
    user_airtable_id = user_data["records"][0]["id"] if user_data.get("records") else None

    recipe_payload = {
        "fields": {
            "Title": recipe.title,
            "Cook Time (Minutes)": recipe.cook_time_minutes,
            "Ingredients": json.dumps([i.dict() for i in recipe.ingredients]),
            "Steps": json.dumps(recipe.steps),
            "Recipe ID": recipe.id,
            "User ID": [user_airtable_id] if user_airtable_id else None,
        }
    }

    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
    r = requests.post(url, headers=headers, json=recipe_payload)
    print("Airtable recipe sync status:", r.status_code, r.text)

@app.post("/upload-video", response_model=Recipe)
def upload_video(user_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        with tempfile.TemporaryDirectory() as frame_dir:
            subprocess.run([
                "ffmpeg", "-i", tmp_path,
                "-vf", "fps=1/2",
                os.path.join(frame_dir, "frame_%03d.jpg")
            ], check=True)
            image_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])

            if not image_files:
                raise ValueError("No frames extracted from video.")

            prompt = [
                {"role": "system", "content": (
                    "You are an expert recipe extractor. Based on a sequence of images showing a cooking video, "
                    "you will identify the dish, ingredients, steps, and estimate a cook time. "
                    "Always output valid JSON with this format:\n"
                    "{ \"title\": str, \"ingredients\": [{\"name\": str, \"quantity\": str}], \"steps\": [str], \"cook_time_minutes\": int }\n"
                    "Only return the JSON. Do not explain anything."
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": "These are video frames of a cooking process. Output only the JSON for the recipe:"},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}} for f in image_files[:12]]
                ]}
            ]

            result = client.chat.completions.create(
                model="gpt-4o",
                messages=prompt,
                max_tokens=1000
            )

        raw = result.choices[0].message.content
        if not raw:
            raise ValueError("No response from GPT.")

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if not match:
            raise ValueError(f"GPT did not return JSON. Response:\n{raw}")

        try:
            raw_json = match.group(1).strip()
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON. Error: {e}\nRaw:\n{raw}")

        raw_ingredients = parsed["ingredients"]
        ingredients = []
        for item in raw_ingredients:
            if isinstance(item, str):
                ingredients.append(Ingredient(name=item.strip()))
            elif isinstance(item, dict) and "name" in item:
                ingredients.append(Ingredient(
                    name=item["name"],
                    quantity=item.get("quantity")
                ))

        recipe = Recipe(
            id=str(uuid.uuid4()),
            title=parsed["title"],
            ingredients=ingredients,
            steps=parsed["steps"],
            cook_time_minutes=int(parsed.get("cook_time_minutes", 20)),
            user_id=user_id
        )

        db_recipe = RecipeDB(
            id=recipe.id,
            title=recipe.title,
            ingredients=json.dumps([i.dict() for i in recipe.ingredients]),
            steps=json.dumps(recipe.steps),
            cook_time_minutes=recipe.cook_time_minutes,
            user_id=user_id
        )
        db.add(db_recipe)
        db.commit()

        sync_recipe_to_airtable(recipe)

        return recipe

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




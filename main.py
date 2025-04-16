import os
import re
import json
import base64
import shutil
import smtplib
import tempfile
import subprocess
from datetime import date, datetime
from typing import List, Optional
import sqlite3
import hashlib
import yt_dlp


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker, relationship, declarative_base
from sqlalchemy import Column, String, Integer, Text, ForeignKey, create_engine
from pydantic import BaseModel
from starlette.responses import JSONResponse
from email.mime.text import MIMEText
import requests
from openai import OpenAI
import uuid
from schemas import UserLogin  # make sure you have UserLogin in schemas.py
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_RECIPES_TABLE = "Recipes"
AIRTABLE_USERS_TABLE = "Users"
AIRTABLE_INTERACTIONS_TABLE = "UserInteractions"
AIRTABLE_SAVED_RECIPES_TABLE = "SavedRecipes"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

class UserDB(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    auth_provider = Column(String, default="email")
    registration_date = Column(String)
    last_login = Column(String)
    uploaded_count = Column(Integer, default=0)
    saved_count = Column(Integer, default=0)
    recipes = relationship("RecipeDB", back_populates="user")

class RecipeDB(Base):
    __tablename__ = "recipes"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    ingredients = Column(Text, nullable=False)
    steps = Column(Text, nullable=False)
    cook_time_minutes = Column(Integer)
    user_id = Column(String, ForeignKey("users.id"))
    user = relationship("UserDB", back_populates="recipes")

Base.metadata.create_all(bind=engine)

ingredient_db_path = "ingredients.db"

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None

class Recipe(BaseModel):
    id: str
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: int
    user_id: Optional[str] = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(user: UserLogin = Body(...)):
    """
    Authenticate a user against Airtable.
    """
    # 1. Fetch user record by email
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }
    params = {
        "filterByFormula": f"{{Email}} = '{user.email}'"
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch user")
    records = resp.json().get("records", [])
    if not records:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Verify password
    record = records[0]["fields"]
    hashed_input = hashlib.sha256(user.password.encode()).hexdigest()
    if record.get("Password") != hashed_input:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # 3. Return success with Airtable record ID
    return {
        "success": True,
        "user_id": records[0]["id"],
        "name": record.get("Name")
    }

class UserSignup(BaseModel):
    name: str
    email: str
    password: str

@app.post("/signup")
def signup(user: UserSignup):
    """
    Register a new user in Airtable. Prevents duplicate emails.
    """
    # 1. Check for existing user by email
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params  = {"filterByFormula": f"{{Email}} = '{user.email}'"}
    url     = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    resp    = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch user")
    if resp.json().get("records"):
        raise HTTPException(status_code=400, detail="Email already registered")

    # 2. Create the user
    post_headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "fields": {
            "Name":              user.name,
            "Email":             user.email,
            "Password":          hashlib.sha256(user.password.encode()).hexdigest(),
            "Registration Date": str(date.today())
        }
    }
    r2 = requests.post(url, headers=post_headers, json=payload)

    if r2.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail="Failed to create user")

    res = r2.json()
    # Use Airtable record ID as the user ID
    return {"success": True, "user_id": res.get("id")}

@app.post("/save-recipe")
def save_recipe(payload: dict = Body(...)):
    # Log the incoming payload
    logger.info(f"[save-recipe] payload: {payload}")

    user_id = payload.get("user_id")
    recipe  = payload.get("recipe")
    if not user_id or not recipe:
        raise HTTPException(status_code=400, detail="Missing user_id or recipe")

    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type":  "application/json"
    }
    data = {
        "fields": {
            "User ID": [user_id],  # ← wrap in list
            "Recipe JSON": json.dumps(recipe)
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_SAVED_RECIPES_TABLE}"

    # Log the Airtable request
    logger.info(f"[save-recipe] POST {url} → {data}")
    resp = requests.post(url, headers=headers, json=data)

    if resp.status_code not in (200, 201):
        # Log Airtable’s error response
        logger.error(f"[save-recipe] Airtable error {resp.status_code}: {resp.text}")
        raise HTTPException(
            status_code=500,
            detail=f"Airtable save failed: {resp.status_code}"
        )

    logger.info(f"[save-recipe] success: {resp.json()}")
    return resp.json()

@app.get("/saved-recipes/{user_id}")
def get_saved_recipes(user_id: str):
    """Retrieve saved recipes for a specified user."""
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"filterByFormula": f"{{User ID}} = '{user_id}'"}
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_SAVED_RECIPES_TABLE}"
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Could not retrieve saved recipes")
    records = resp.json().get("records", [])
    # return only fields or full records as needed
    return records

    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }
    params = {
        "filterByFormula": f"{{Email}} = '{user.email}'"
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    check = requests.get(url, headers=headers, params=params)

    if check.status_code == 200 and check.json().get("records"):
        # Duplicate email detected
        raise HTTPException(status_code=400, detail="Email already registered")

    post_headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
    payload = {
        "fields": {
            "User ID": str(uuid.uuid4()),
            "Name": user.name,
            "Email": user.email,
            "Password": hashed_password,
            "Authentication Provider": "email",
            "Registration Date": str(date.today()),
            "Number of Uploaded Recipes": 0
        }
    }
    response = requests.post(url, headers=post_headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to create user")

    return {"success": True, "user_id": payload["fields"]["User ID"]}

class UserInteraction(BaseModel):
    user_id: str
    recipe_id: str
    action: str  # e.g. "saved", "viewed"
    timestamp: Optional[str] = None

@app.post("/interact")
def save_interaction(interaction: UserInteraction):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }

    # Fetch Airtable record IDs for user and recipe
    user_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    recipe_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"

    user_resp = requests.get(user_url, headers=headers, params={"filterByFormula": f'{{User ID}} = "{interaction.user_id}"'})
    recipe_resp = requests.get(recipe_url, headers=headers, params={"filterByFormula": f'{{Recipe ID}} = "{interaction.recipe_id}"'})

    if user_resp.status_code != 200 or recipe_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to lookup user or recipe record")

    user_records = user_resp.json().get("records", [])
    recipe_records = recipe_resp.json().get("records", [])

    if not user_records or not recipe_records:
        raise HTTPException(status_code=404, detail="User or Recipe not found")

    airtable_user_id = user_records[0]["id"]
    airtable_recipe_id = recipe_records[0]["id"]

    post_headers = headers.copy()
    post_headers["Content-Type"] = "application/json"

    now = interaction.timestamp or datetime.utcnow().isoformat()
    payload = {
        "fields": {
            "User ID": [airtable_user_id],
            "Recipe ID": [airtable_recipe_id],
            "Action": interaction.action,
            "Timestamp": now,
            "Unique Key": f"{interaction.user_id} - {interaction.recipe_id} - {interaction.action} - {now[:16]}"
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_INTERACTIONS_TABLE}"
    r = requests.post(url, headers=post_headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to sync interaction: {r.text}")

    return {"success": True, "airtable_id": r.json().get("id")}


@app.get("/interactions/{user_id}")
def get_user_interactions(user_id: str):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }
    params = {
        "filterByFormula": f"{{User ID}} = '{user_id}'"
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_INTERACTIONS_TABLE}"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch interactions")

    records = response.json().get("records", [])
    interactions = [
        {
            "recipe_id": record["fields"].get("Recipe ID", [None])[0],
            "action": record["fields"].get("Action"),
            "timestamp": record["fields"].get("Timestamp")
        }
        for record in records if "Recipe ID" in record["fields"]
    ]

    return {"user_id": user_id, "interactions": interactions}


def classify_image_multiple(images):
    print(f"[DEBUG] Classifying {len(images)} images")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    probabilities = None
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        probabilities = prob if probabilities is None else probabilities + prob
    class_id = probabilities.argmax().item()
    print(f"[DEBUG] Voted class ID: {class_id}")
    return class_id

def get_known_ingredients_and_dishes():
    conn = sqlite3.connect(ingredient_db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM ingredients")
    ingredients = [row[0] for row in c.fetchall()]
    c.execute("SELECT name FROM dishes")
    dishes = [row[0] for row in c.fetchall()]
    conn.close()
    return ingredients, dishes

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(None),
    tiktok_url: str    = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Accept either:
      • a TikTok URL (downloads video + fetches description), or
      • a video file upload.
    Extract frames, classify, and run GPT-4o on combined description + images.
    """
    temp_dir    = None
    description = ""
    frames      = []

    try:
        # 1. If TikTok URL provided, download video and fetch description
        if tiktok_url:
            temp_dir = tempfile.mkdtemp()
            ydl_opts = {
                "quiet": True,
                "skip_download": False,
                "outtmpl": os.path.join(temp_dir, "tiktok.%(ext)s"),
                "format": "mp4"
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(tiktok_url, download=True)
                description = info.get("description", "") or ""
                video_path  = ydl.prepare_filename(info)

        # 2. If file provided, save locally
        if file:
            if not temp_dir:
                temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, file.filename)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 3. Validate input
        if not (tiktok_url or file):
            raise HTTPException(status_code=400, detail="Must provide a TikTok URL or video file")

        # 4. Extract frames via FFmpeg
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vf", "fps=1,scale=128:-1",
            os.path.join(frames_dir, "frame_%04d.jpg")
        ], check=True)

        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith(".jpg")
        ])
        if not frames:
            raise HTTPException(status_code=500, detail="No frames extracted")

        # 5. Optionally classify images
        guess_id = classify_image_multiple(frames) if frames else None

        # 6. Prepare frames for GPT (cap at 90)
        selected = frames[:90]
        mid      = len(selected) // 2

        # 7. Build GPT prompt including description
        def gpt_prompt(frames_subset: List[str]):
            parts = []
            if description:
                parts.append(f"Here is the video description:\\n{description}\\n")
            parts.append(
                "You are an expert recipe extractor. "
                "Based on the images, output valid JSON with keys: "
                "title, ingredients (list of {name,quantity}), steps (list), cook_time_minutes (int)."
            )
            system_msg = {"role": "system", "content": "\\n\\n".join(parts)}

            user_list = [{"type":"text","text":"Extract recipe JSON from these frames:"}]
            for fpath in frames_subset:
                b64 = base64.b64encode(open(fpath, "rb").read()).decode()
                user_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })

            return [system_msg, {"role":"user","content": user_list}]

        # 8. Run GPT in two passes
        first_pass = client.chat.completions.create(
            model="gpt-4o",
            messages=gpt_prompt(selected[:mid]),
            max_tokens=1000
        )
        second_pass = client.chat.completions.create(
            model="gpt-4o",
            messages=gpt_prompt(selected[mid:]),
            max_tokens=1000
        )
        
        # Combine GPT outputs
        combined = first_pass.choices[0].message.content.strip() + "\n" + second_pass.choices[0].message.content.strip()
        
        # Log raw GPT response to help debug format issues
        print(f"[DEBUG] Raw GPT response (first 500 chars):\n{combined[:500]}")
        
        # Try to extract JSON from a ```json code block
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", combined, re.DOTALL)
        
        try:
            parsed = json.loads(match.group(1).strip() if match else combined)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse GPT output: {str(e)}")


        # 9. Construct and return result
        return {
            "title": parsed.get("title"),
            "ingredients": parsed.get("ingredients"),
            "steps": parsed.get("steps"),
            "cook_time_minutes": parsed.get("cook_time_minutes"),
            "debug": {
                "frames_processed": len(frames),
                "model_hint": guess_id or "n/a"
            }
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.get("/user-recipes")
def get_user_recipes(user_id: str):
    """
    Return all saved recipes for a given user from Airtable.
    """
    import logging
    logger = logging.getLogger("main")

    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }

    filter_formula = f"FIND('{user_id}', ARRAYJOIN({{User ID}}))"
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_SAVED_RECIPES_TABLE}"
    params = {"filterByFormula": filter_formula}

    resp = requests.get(url, headers=headers, params=params)
    logger.info(f"[user-recipes] GET {url} → status {resp.status_code}")

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch recipes")

    recipes = []
    for record in resp.json().get("records", []):
        recipe_json_str = record.get("fields", {}).get("Recipe JSON")
        if recipe_json_str:
            try:
                recipe_data = json.loads(recipe_json_str)
                recipes.append({
                    "id": record["id"],
                    "title": recipe_data.get("title"),
                    "cook_time_minutes": recipe_data.get("cookTimeMinutes"),
                    "full": recipe_data
                })
            except Exception as e:
                logger.warning(f"[user-recipes] Could not parse recipe: {e}")
    return recipes


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
            "Number of Uploaded Recipes": user.uploaded_count
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

print("✅ Tables ensured on startup")

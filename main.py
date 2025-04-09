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

import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker, relationship, declarative_base
from sqlalchemy import Column, String, Integer, Text, ForeignKey, create_engine
from pydantic import BaseModel
from starlette.responses import JSONResponse
from email.mime.text import MIMEText
import requests
from openai import OpenAI
import uuid

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_RECIPES_TABLE = "Recipes"
AIRTABLE_USERS_TABLE = "Users"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

print("✅ Tables ensured on startup")
print(f"✅ NumPy available: {np.__version__}")

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

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def classify_image(image_path):
    print(f"[DEBUG] Classifying image: {image_path}")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    class_id = probabilities.argmax().item()
    print(f"[DEBUG] Predicted class ID: {class_id}")
    return class_id

@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    print(f"[DEBUG] Signup request for {user.email}")
    existing = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing:
        print("[DEBUG] Email already registered")
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = UserDB(
        id=str(uuid.uuid4()),
        name=user.name,
        email=user.email,
        password=user.password,
        registration_date=str(datetime.utcnow().date()),
        last_login=str(datetime.utcnow().date())
    )
    db.add(new_user)
    db.commit()
    print("[DEBUG] New user committed to database")
    sync_user_to_airtable(new_user)
    return {"success": True, "user_id": new_user.id}

@app.post("/upload-video")
def upload_video(file: UploadFile = File(...), user_id: Optional[str] = Form(None), db: Session = Depends(get_db)):
    try:
        print("[DEBUG] Uploading video")
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vf", "fps=1/2",
            os.path.join(frames_dir, "frame_%04d.jpg")
        ], check=True)

        frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        print(f"[DEBUG] Extracted {len(frames)} frames")
        if not frames:
            raise HTTPException(status_code=500, detail="No frames extracted")

        guess_id = classify_image(frames[0])

        prompt = [
            {"role": "system", "content": (
                "You are an expert recipe extractor. Based on a sequence of images showing a cooking video, "
                f"the dish may resemble ImageNet class ID {guess_id}. Use that as guidance. "
                "Identify the dish, ingredients, steps, and estimate a cook time. Always output valid JSON with this format:\n"
                "{ \"title\": str, \"ingredients\": [{\"name\": str, \"quantity\": str}], \"steps\": [str], \"cook_time_minutes\": int }\n"
                "Only return the JSON. Do not explain anything."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": "These are video frames of a cooking process. Output only the JSON for the recipe:"},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}} for f in frames[:12]]
            ]}
        ]

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=1000
        )

        raw = result.choices[0].message.content.strip()
        print(f"[DEBUG] Raw GPT response: {raw[:200]}...")
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        parsed = json.loads(match.group(1).strip() if match else raw)

        ingredients = []
        for item in parsed["ingredients"]:
            if isinstance(item, dict):
                ingredients.append(Ingredient(name=item["name"], quantity=item.get("quantity")))
            elif isinstance(item, str):
                ingredients.append(Ingredient(name=item))

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
        print("[DEBUG] Recipe saved to database")

        sync_recipe_to_airtable(recipe)
        shutil.rmtree(temp_dir)
        return recipe

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recipes")
def get_recipes(user_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(RecipeDB)
    if user_id:
        query = query.filter(RecipeDB.user_id == user_id)
    recipes = query.all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "ingredients": json.loads(r.ingredients),
            "steps": json.loads(r.steps),
            "cook_time_minutes": r.cook_time_minutes,
            "user_id": r.user_id
        } for r in recipes
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

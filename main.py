# main.py — full app with login, signup, upload, Airtable sync, and per-user recipes

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from openai import OpenAI
import tempfile
import subprocess
import os
import json
import base64
import re
import requests
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_RECIPES_TABLE = "Recipes"
AIRTABLE_USERS_TABLE = "Users"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

# -----------------------------
# Airtable Sync
# -----------------------------

def sync_user_to_airtable(user_id: str, email: str, name: str = "Guest", provider: str = "email"):
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
        return
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "fields": {
            "User ID": user_id,
            "Name": name,
            "Email": email,
            "Authentication Provider": provider,
            "Registration Date": str(datetime.utcnow().date()),
            "Last Login": str(datetime.utcnow().date()),
            "Number of Uploaded Recipes": 0
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    response = requests.post(url, headers=headers, json=data)
    print("Airtable user sync status:", response.status_code, response.text)

def sync_recipe_to_airtable(recipe: Recipe):
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
        return
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "fields": {
            "Recipe ID": recipe.id,
            "Title": recipe.title,
            "Cook Time (Minutes)": recipe.cook_time_minutes,
            "Ingredients": json.dumps([i.dict() for i in recipe.ingredients]),
            "Steps": json.dumps(recipe.steps),
            "User ID": recipe.user_id or "",
            "Created At": str(datetime.utcnow().date())
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
    requests.post(url, headers=headers, json=data)

# -----------------------------
# Auth endpoints
# -----------------------------

@app.post("/signup")
def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing = db.query(UserDB).filter(UserDB.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user_id = str(uuid.uuid4())
    user = UserDB(id=user_id, name=name, email=email, password=password)
    db.add(user)
    db.commit()
    sync_user_to_airtable(user_id=user.id, email=user.email, name=user.name)
    return {"success": True, "user_id": user.id}

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == email, UserDB.password == password).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"success": True, "user_id": user.id}

@app.get("/user-recipes", response_model=List[Recipe])
def get_user_recipes(user_id: str, db: Session = Depends(get_db)):
    rows = db.query(RecipeDB).filter(RecipeDB.user_id == user_id).all()
    return [
        Recipe(
            id=r.id,
            title=r.title,
            ingredients=json.loads(r.ingredients),
            steps=json.loads(r.steps),
            cook_time_minutes=r.cook_time_minutes,
            user_id=r.user_id
        ) for r in rows
    ]

# -----------------------------
# Upload Endpoint
# -----------------------------

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

            prompt = [
                {"role": "system", "content": "You're a recipe parser. Output JSON only. Format: {title, ingredients, steps, cook_time_minutes}"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here are images from a cooking video. Generate the recipe."},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}} for f in image_files[:12]]
                ]}
            ]
            result = client.chat.completions.create(
                model="gpt-4o",
                messages=prompt,
                max_tokens=1000
            )

        raw = result.choices[0].message.content
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        raw_json = match.group(1).strip() if match else raw.strip()
        parsed = json.loads(raw_json)

        recipe = Recipe(
            id=str(uuid.uuid4()),
            title=parsed["title"],
            ingredients=[Ingredient(**i) for i in parsed["ingredients"]],
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

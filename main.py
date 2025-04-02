# main.py — full code with Airtable sync on recipe and user upload + user signup

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
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

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

class RecipeDB(Base):
    __tablename__ = "recipes"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    ingredients = Column(Text, nullable=False)
    steps = Column(Text, nullable=False)
    cook_time_minutes = Column(Integer)

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

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

def clean_json_output(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()

def estimate_cook_time(title: str, steps: List[str]) -> int:
    keywords = {
        "slow cooker": 240,
        "butter chicken": 60,
        "lasagna": 90,
        "omelette": 10,
        "pasta": 25,
        "salad": 15,
        "soup": 40
    }
    for key, val in keywords.items():
        if key in title.lower():
            return val
    return max(10, len(steps) * 5)

def fix_ingredients(ingredients: List[dict]) -> List[Ingredient]:
    fixed = []
    for item in ingredients:
        name = item.get("name", "").strip()
        quantity = item.get("quantity", "").strip() if item.get("quantity") else ""
        if not quantity and any(x in name.lower() for x in ["garnish", "as needed"]):
            quantity, name = name, "cilantro" if "cilantro" in quantity else name
        fixed.append(Ingredient(name=name, quantity=quantity or "to taste"))
    return fixed

def safe_parse_minutes(value) -> Optional[int]:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def validate_recipe_fields(data: dict):
    required = ["title", "ingredients", "steps", "cook_time_minutes"]
    missing = [k for k in required if k not in data or not data[k]]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields in recipe: {', '.join(missing)}")

def describe_frame_batches(frames: list[list[str]]) -> str:
    descriptions = []
    for i, batch in enumerate(frames):
        messages = [
            {"role": "system", "content": "You describe what is visually happening in a batch of video frames from a cooking video."},
            {"role": "user", "content": [
                {"type": "text", "text": "What’s happening in this sequence of frames? Describe only the cooking actions, ingredients, and transitions."},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}} for f in batch]
            ]},
        ]
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        descriptions.append(result.choices[0].message.content.strip())
    return "\n".join(descriptions)

def use_gpt4_vision_on_frames(frames_dir: str) -> Recipe:
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    batches = [image_files[i:i+4] for i in range(0, len(image_files), 4)]

    summarized_steps = describe_frame_batches(batches)

    if not summarized_steps.strip() or any(term in summarized_steps.lower() for term in ["unclear", "can't tell", "unknown", "no food"]):
        fallback_prompt = [
            {"role": "system", "content": "You are a recipe assistant. Your job is to generate a structured JSON recipe from the provided frames. Even if parts of the process are unclear, make your best guess. You must respond ONLY with valid JSON, no commentary, no apologies, no preamble."},
            {"role": "user", "content": [
                {"type": "text", "text": "Here are frames from a cooking video. Return your response ONLY in this exact format — without any explanation or markdown."},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}} for f in image_files]
            ]}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=fallback_prompt,
            max_tokens=1000
        )
        raw_output = clean_json_output(response.choices[0].message.content)
    else:
        structured_prompt = f"""You are watching a cooking video, broken into described visual segments. Your task is to infer the dish that was made and write the recipe that was actually followed — not a general version. Use domain knowledge only to fill minor gaps (e.g. estimate cook time, typical quantities). Always prioritize what is visible. Respond with valid JSON only.

Visual breakdown:
{summarized_steps}

Respond in this format only:
{{
  "title": str,
  "ingredients": [{{"name": str, "quantity": str}}],
  "steps": [str],
  "cook_time_minutes": int
}}
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a structured recipe writer."},
                {"role": "user", "content": structured_prompt}
            ],
            max_tokens=1000
        )
        raw_output = clean_json_output(response.choices[0].message.content)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from GPT-4 Vision: {raw_output}")

    parsed_minutes = safe_parse_minutes(data.get("cook_time_minutes"))
    data["cook_time_minutes"] = parsed_minutes if parsed_minutes is not None else estimate_cook_time(data["title"], data["steps"])

    validate_recipe_fields(data)

    return Recipe(
        id=str(uuid.uuid4()),
        title=data["title"],
        ingredients=fix_ingredients(data["ingredients"]),
        steps=data["steps"],
        cook_time_minutes=data["cook_time_minutes"]
    )

def sync_user_to_airtable(user_id: str, email: str, name: str = "Guest", provider: str = "Guest"):
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
    requests.post(url, headers=headers, json=data)

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
            "Created At": str(datetime.utcnow().date())
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
    requests.post(url, headers=headers, json=data)

@app.post("/upload-video", response_model=Recipe)
def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
            recipe = use_gpt4_vision_on_frames(frame_dir)

        db_recipe = RecipeDB(
            id=recipe.id,
            title=recipe.title,
            ingredients=json.dumps([i.dict() for i in recipe.ingredients]),
            steps=json.dumps(recipe.steps),
            cook_time_minutes=recipe.cook_time_minutes
        )
        db.add(db_recipe)
        db.commit()

        sync_recipe_to_airtable(recipe)

        return recipe

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recipes", response_model=List[Recipe])
def list_recipes(db: Session = Depends(get_db)):
    rows = db.query(RecipeDB).all()
    return [
        Recipe(
            id=r.id,
            title=r.title,
            ingredients=json.loads(r.ingredients),
            steps=json.loads(r.steps),
            cook_time_minutes=r.cook_time_minutes
        ) for r in rows
    ]

@app.post("/signup")
def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    user_id = str(uuid.uuid4())
    sync_user_to_airtable(user_id=user_id, email=email, name=name, provider="email")
    return {"success": True, "user_id": user_id}

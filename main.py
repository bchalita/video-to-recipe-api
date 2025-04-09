import os
import uuid
import json
import shutil
import smtplib
import tempfile
import subprocess
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Column, String, Integer, Text, ForeignKey, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from email.mime.text import MIMEText
from datetime import datetime
import requests
import openai

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing:
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
    return {"success": True, "user_id": new_user.id}

@app.post("/upload-video")
def upload_video(file: UploadFile = File(...), user_id: Optional[str] = Form(None), db: Session = Depends(get_db)):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vf", "fps=1/2",
            os.path.join(frames_dir, "frame_%04d.jpg")
        ], check=True)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to extract frames from video")

    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not frames:
        raise HTTPException(status_code=500, detail="No frames extracted")

    vision_prompt = """Extract the recipe being prepared in these frames. Return JSON with:
  - title (string)
  - ingredients (list of {name, quantity})
  - steps (list of strings)
  - cook_time_minutes (integer)
"""
    messages = [
        {"role": "system", "content": vision_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": vision_prompt},
            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}} for f in frames]
        ]}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=1000
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT did not return JSON: {e}")

    recipe_id = str(uuid.uuid4())
    db_recipe = RecipeDB(
        id=recipe_id,
        title=parsed["title"],
        ingredients=json.dumps(parsed["ingredients"]),
        steps=json.dumps(parsed["steps"]),
        cook_time_minutes=parsed.get("cook_time_minutes", 30),
        user_id=user_id
    )
    db.add(db_recipe)
    db.commit()

    shutil.rmtree(temp_dir)

    return {"id": recipe_id, **parsed}

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

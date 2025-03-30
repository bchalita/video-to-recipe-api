# main.py — deployable FastAPI backend for recipe extraction with fallback upload option and GPT-4 Vision for silent videos

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uuid
from openai import OpenAI
import tempfile
import subprocess
import os
import json
import shutil
import base64

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VideoRequest(BaseModel):
    url: str

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None

class Recipe(BaseModel):
    id: str
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: Optional[int]

def use_gpt4_vision_on_frames(frames_dir: str) -> Recipe:
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(img, 'rb').read()).decode()}"}}
        for img in image_files
    ]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are a recipe analysis assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "These are frames from a recipe video. Please extract a recipe with title, ingredients (with quantities), steps, and estimated cook time."},
                *image_messages
            ]}
        ],
        max_tokens=1000
    )

    data = json.loads(response.choices[0].message.content)

    return Recipe(
        id=str(uuid.uuid4()),
        title=data["title"],
        ingredients=[Ingredient(**item) for item in data["ingredients"]],
        steps=data["steps"],
        cook_time_minutes=data["cook_time_minutes"]
    )

def extract_recipe_from_file(file_path: str) -> Recipe:
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        ).text

    if not transcript.strip():
        print("No transcript found, using GPT-4 Vision fallback")
        with tempfile.TemporaryDirectory() as frame_dir:
            subprocess.run([
                "ffmpeg", "-i", file_path,
                "-vf", "fps=1/2",
                os.path.join(frame_dir, "frame_%03d.jpg")
            ], check=True)
            return use_gpt4_vision_on_frames(frame_dir)

    prompt = f"""
    Extract a recipe in structured JSON format from the following transcript:

    {transcript}

    Format:
    {{
      "title": str,
      "ingredients": [{{"name": str, "quantity": str}}],

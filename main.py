# main.py — deployable FastAPI backend for recipe extraction

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import uuid
import openai
import tempfile
import subprocess
import os
import json

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

@app.post("/generate-recipe", response_model=Recipe)
def generate_recipe(video: VideoRequest):
    if not video.url:
        raise HTTPException(status_code=400, detail="Missing video URL")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            subprocess.run([
                "yt-dlp", "-f", "mp4", "-o", video_path, video.url
            ], check=True)

            with open(video_path, "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)["text"]

        prompt = f"""
        Extract a recipe in structured JSON format from the following transcript:

        {transcript}

        Format:
        {{
          "title": str,
          "ingredients": [{{"name": str, "quantity": str}}],
          "steps": [str],
          "cook_time_minutes": int
        }}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a recipe parser."},
                {"role": "user", "content": prompt}
            ]
        )
        data = json.loads(response.choices[0].message.content)

        return Recipe(
            id=str(uuid.uuid4()),
            title=data["title"],
            ingredients=[Ingredient(**item) for item in data["ingredients"]],
            steps=data["steps"],
            cook_time_minutes=data["cook_time_minutes"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# main.py — fallback to GPT-3.5 with dummy transcript while waiting for OpenAI Tier 1 unlock

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uuid
from openai import OpenAI
import tempfile
import subprocess
import os
import json

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

def extract_recipe_from_file(file_path: str) -> Recipe:
    # Temporarily bypass Whisper/Vision; use dummy transcript
    transcript = "Today we're making a simple pasta. Boil pasta, sauté garlic in olive oil, mix together with parmesan and parsley."

    prompt = f"""Extract a recipe in structured JSON format from the following transcript:

{transcript}

Format:
{{
  "title": str,
  "ingredients": [{{"name": str, "quantity": str}}],
  "steps": [str],
  "cook_time_minutes": int
}}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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

@app.post("/generate-recipe", response_model=Recipe)
def generate_recipe(video: VideoRequest):
    if not video.url:
        raise HTTPException(status_code=400, detail="Missing video URL")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            result = subprocess.run(
                ["yt-dlp", "-f", "mp4", "-o", video_path, video.url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("yt-dlp stdout:", result.stdout)
            print("yt-dlp stderr:", result.stderr)
            result.check_returncode()

            return extract_recipe_from_file(video_path)

    except subprocess.CalledProcessError as e:
        print("yt-dlp failed. Asking for manual upload.")
        raise HTTPException(
            status_code=422,
            detail="This video requires manual upload. Please download the video and use /upload-video instead."
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-video", response_model=Recipe)
def upload_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        return extract_recipe_from_file(tmp_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

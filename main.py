# main.py — deployable FastAPI backend for recipe extraction with fallback upload option and GPT-4 Vision for silent videos

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uuid
import openai
import tempfile
import subprocess
import os
import json
import shutil
import base64

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

def use_gpt4_vision_on_frames(frames_dir: str) -> Recipe:
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(img, 'rb').read()).decode()}"}}
        for img in image_files
    ]

    response = openai.ChatCompletion.create(
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
        transcript = openai.Audio.transcribe("whisper-1", f)["text"]

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

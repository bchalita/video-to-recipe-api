# main.py — harden fallback prompt to force JSON from GPT-4 Vision

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import uuid
from openai import OpenAI
import tempfile
import subprocess
import os
import json
import shutil
import base64
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None

class Recipe(BaseModel):
    id: str
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: Optional[int]

def clean_json_output(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()

def safe_parse_minutes(value) -> Optional[int]:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def validate_recipe_fields(data: dict):
    required = ["title", "ingredients", "steps"]
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
        if any(x in raw_output.lower() for x in ["does not contain a recipe", "cannot extract", "doesn't include ingredients"]):
            raise HTTPException(status_code=422, detail="Video appears to contain no recognizable recipe content.")
        raise HTTPException(status_code=500, detail=f"Invalid JSON from GPT-4 Vision: {raw_output}")

    validate_recipe_fields(data)

    return Recipe(
        id=str(uuid.uuid4()),
        title=data["title"],
        ingredients=[Ingredient(**item) for item in data["ingredients"]],
        steps=data["steps"],
        cook_time_minutes=safe_parse_minutes(data.get("cook_time_minutes"))
    )

@app.post("/upload-video", response_model=Recipe)
def upload_video(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        with tempfile.TemporaryDirectory() as frame_dir:
            subprocess.run([
                "ffmpeg", "-i", tmp_path,
                "-vf", "fps=1/2",
                os.path.join(frame_dir, "frame_%03d.jpg")
            ], check=True)
            return use_gpt4_vision_on_frames(frame_dir)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

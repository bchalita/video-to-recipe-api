# main.py — enhanced GPT vision fallback with robustness against weak frame output

from fastapi import FastAPI, HTTPException, UploadFile, File
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

def clean_json_output(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()

def safe_parse_minutes(value) -> Optional[int]:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def describe_frame_batches(frames: list[list[str]]) -> str:
    descriptions = []
    for i, batch in enumerate(frames):
        messages = [
            {"role": "system", "content": "You describe what is visually happening in a batch of video frames from a cooking video."},
            {"role": "user", "content": [
                {"type": "text", "text": f"What’s happening in this sequence of frames? Describe only the cooking actions, ingredients, and transitions."},
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

    # fallback if GPT sees nothing useful in vision summaries
    if not summarized_steps.strip() or any(term in summarized_steps.lower() for term in ["unclear", "can't tell", "unknown", "no food"]):
        fallback_prompt = [
            {"role": "system", "content": "You are a recipe generation assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "These are frames from a cooking video. Generate a structured JSON recipe based on what is seen."},
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

    return Recipe(
        id=str(uuid.uuid4()),
        title=data["title"],
        ingredients=[Ingredient(**item) for item in data["ingredients"]],
        steps=data["steps"],
        cook_time_minutes=safe_parse_minutes(data.get("cook_time_minutes"))
    )

# main.py — stronger fallback prompting and graceful handling of ambiguous frame content

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
            {"role": "system", "content": "You are a recipe assistant. Your job is to generate a structured JSON recipe from the images provided. Even if parts of the process are unclear, make your best guess based on visual evidence and cooking knowledge. Do not reject the task."},
            {"role": "user", "content": [
                {"type": "text", "text": "Here are frames from a cooking video. Provide a structured recipe in JSON format that represents what is most likely being prepared."},
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
        # check for polite refusals
        if any(x in raw_output.lower() for x in ["does not contain a recipe", "cannot extract", "doesn't include ingredients"]):
            raise HTTPException(status_code=422, detail="Video appears to contain no recognizable recipe content.")
        raise HTTPException(status_code=500, detail=f"Invalid JSON from GPT-4 Vision: {raw_output}")

    return Recipe(
        id=str(uuid.uuid4()),
        title=data["title"],
        ingredients=[Ingredient(**item) for item in data["ingredients"]],
        steps=data["steps"],
        cook_time_minutes=safe_parse_minutes(data.get("cook_time_minutes"))
    )

def extract_recipe_from_file(file_path: str) -> Recipe:
    transcript = ""
    try:
        with open(file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            ).text
    except Exception as e:
        print("Whisper failed or unsupported audio. Switching to GPT-4 Vision fallback.")

    if not transcript.strip():
        print("Transcript is empty or unavailable. Switching to GPT-4 Vision fallback.")
        with tempfile.TemporaryDirectory() as frame_dir:
            subprocess.run([
                "ffmpeg", "-i", file_path,
                "-vf", "fps=1/2",
                os.path.join(frame_dir, "frame_%03d.jpg")
            ], check=True)
            return use_gpt4_vision_on_frames(frame_dir)

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
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a recipe parser."},
            {"role": "user", "content": prompt}
        ]
    )
    raw_output = clean_json_output(response.choices[0].message.content)
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from GPT-4: {raw_output}")

    return Recipe(
        id=str(uuid.uuid4()),
        title=data["title"],
        ingredients=[Ingredient(**item) for item in data["ingredients"]],
        steps=data["steps"],
        cook_time_minutes=safe_parse_minutes(data.get("cook_time_minutes"))
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

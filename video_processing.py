# video_processing.py

import os, json, tempfile, shutil, subprocess, base64, time, re, sqlite3
import yt_dlp, torch
from PIL import Image
from torchvision import transforms, models
from fastapi import HTTPException
from openai_client import client

ingredient_db_path = "ingredients.db"  # or import from your db module

def classify_image_multiple(images: list[str]) -> int:
    print(f"[DEBUG] Classifying {len(images)} images")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    probabilities = None
    for image_path in images:
        img = Image.open(image_path).convert("RGB")
        inp = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
        prob = torch.nn.functional.softmax(out[0], dim=0)
        probabilities = prob if probabilities is None else probabilities + prob
    return probabilities.argmax().item()

def get_known_ingredients_and_dishes() -> tuple[list[str], list[str]]:
    conn = sqlite3.connect(ingredient_db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM ingredients")
    ingredients = [r[0] for r in c.fetchall()]
    c.execute("SELECT name FROM dishes")
    dishes = [r[0] for r in c.fetchall()]
    conn.close()
    return ingredients, dishes

def process_video_to_recipe(file, tiktok_url: str, user_id: str | None = None):
    """
    Encapsulate everything from download → frame extraction → GPT recipe JSON → summary call
    → build_airtable_fields payload. Return:
      (title, ingredients, steps, cook_time, video_url, summary, airtable_payload)
    """
    # ... copy your entire upload_video logic here, minus @app.post decorator ...
    # at the end instead of POST'ing to Airtable, just return
    #   recipe_title, ingredients, steps, cook_time_minutes, video_url_field, recipe_summary
    # (the router will call build_airtable_fields and then POST)
    temp_dir = None
    frames = []
    description = ""
    try:
        # Download or save video
        if tiktok_url:
            temp_dir = tempfile.mkdtemp()
            ydl_opts = {"quiet": True, "skip_download": False,
                        "outtmpl": os.path.join(temp_dir, "tiktok.%(ext)s"),
                        "format": "mp4"}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(tiktok_url, download=True)
                video_path = ydl.prepare_filename(info)
        elif file:
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, file.filename)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            raise HTTPException(status_code=400, detail="Must provide a TikTok URL or video file")

        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vf", "fps=1,scale=128:-1",
            os.path.join(frames_dir, "frame_%04d.jpg")
        ], check=True)

        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith(".jpg")
        ])
        if not frames:
            raise HTTPException(status_code=500, detail="No frames extracted")

        guess_id = classify_image_multiple(frames) if frames else None

        # ─── SAMPLE & LIMIT FRAMES TO AVOID TOKEN OVERFLOW ──────────────────────────
        # only send at most 20 images to GPT
        max_frames = 20
        step = max(1, len(frames) // max_frames)
        selected = frames[::step][:max_frames]
        mid = len(selected) // 2

        def gpt_prompt(frames_subset: List[str]):
            parts = []
            if description:
                parts.append(f"Here is the video description:\n{description}\n")
            parts.append(
                "Você é um especialista em extrair receitas. "
                "Com base nas imagens, retorne JSON com chaves: "
                "titulo, ingredientes (lista de {nome,quantidade}), passos (lista), "
                "tempo_preparo_minutos (inteiro). "
                "Todo o texto deve estar em português."
            )

            system_msg = {"role": "system", "content": "\n\n".join(parts)}

            user_list = [{"type": "text", "text": "Extract recipe JSON from these frames:"}]
            for fpath in frames_subset:
                b64 = base64.b64encode(open(fpath, "rb").read()).decode()
                user_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })

            return [system_msg, {"role": "user", "content": user_list}]

        def safe_create(messages):
            for attempt in range(3):
                try:
                    return client.chat.completions.create(
                        model="gpt-4o", messages=messages, max_tokens=1000
                    )
                except Exception as e:
                    if "429" in str(e):
                        time.sleep(2 ** attempt)
                        continue
                    raise
            # last-ditch
            return client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=1000
            )

        first_pass = safe_create(gpt_prompt(selected[:mid]))
        second_pass = safe_create(gpt_prompt(selected[mid:]))

        combined = first_pass.choices[0].message.content.strip() + "\n" + second_pass.choices[0].message.content.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", combined, re.DOTALL)
        parsed = json.loads(match.group(1).strip() if match else combined)

        # Build recipe fields including video_url
        recipe_title      = parsed.get("title") or "Recipe"
        ingredients       = parsed.get("ingredients", [])
        steps             = parsed.get("steps", [])
        cook_time_minutes = parsed.get("cook_time_minutes")
        video_url_field   = tiktok_url or None


        summary_prompt = [
            {"role": "system", "content": "Você é um assistente que cria resumos de receitas em português do Brasil."},
            {"role": "user", "content":
                f"Crie um resumo caloroso em 2–3 frases para esta receita:\n"
                f"Título: {recipe_title}\n"
                f"Ingredientes: {', '.join(i['name'] for i in ingredients)}\n"
                f"Passos: {'; '.join(steps)}"
            }
        ]
        try:
            summary_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=summary_prompt,
                max_tokens=100
            )
            recipe_summary = summary_resp.choices[0].message.content.strip()

        except Exception as e:
          logger.warning("Summary generation failed, continuing without it", e)
          recipe_summary = ""
        
        new_id = resp.json()["records"][0]["id"]
        return {
            "id": new_id,
            "title": recipe_title,
            "ingredients": ingredients,
            "steps": steps,
            "cook_time_minutes": cook_time_minutes,
            "video_url": video_url_field,
            "summary": recipe_summary,
            "debug": {"frames_processed": len(frames)},
        }

    except Exception as e:
        import traceback, sys
        tb = traceback.format_exc()
        print(f"[upload-video] FULL TRACEBACK:\n{tb}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


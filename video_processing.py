import os, json, tempfile, shutil, subprocess, base64, time, re
import yt_dlp, requests
from PIL import Image
from torchvision import transforms, models
from your_openai_client import client  # however you import OpenAI
from fastapi import HTTPException

def process_video_to_recipe(file, tiktok_url):
    # ————————————— DOWNLOAD / SAVE VIDEO —————————————
    temp_dir = tempfile.mkdtemp()
    try:
        if tiktok_url:
            ydl_opts = {
                "quiet": True,
                "outtmpl": os.path.join(temp_dir, "tiktok.%(ext)s"),
                "format": "mp4"
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(tiktok_url, download=True)
                video_path = ydl.prepare_filename(info)
                description = info.get("description", "")
        elif file:
            video_path = os.path.join(temp_dir, file.filename)
            with open(video_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)
            description = ""
        else:
            raise HTTPException(400, "Need file or TikTok URL")

        # ————————————— EXTRACT FRAMES —————————————
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vf", "fps=1,scale=128:-1",
            os.path.join(frames_dir, "frame_%04d.jpg")
        ], check=True)
        frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if not frames:
            raise HTTPException(500, "No frames extracted")

        # ————————— CLASSIFY / SAMPLE FRAMES —————————
        def classify_imgs(img_paths):
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model.eval()
            tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
            probs = None
            for p in img_paths:
                img = Image.open(os.path.join(frames_dir, p)).convert("RGB")
                inp = tf(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(inp)
                prob = torch.nn.functional.softmax(out[0], dim=0)
                probs = prob if probs is None else probs + prob
            return probs.argmax().item()

        # limit to 20 frames
        step = max(1, len(frames)//20)
        sample = frames[::step][:20]
        mid = len(sample)//2

        def make_prompt(subset):
            msgs = []
            if description:
                msgs.append({"role":"system","content":description})
            msgs.append({
              "role":"system",
              "content":(
                "You are an expert recipe extractor. "
                "Return JSON with keys: title, ingredients(list of {name,quantity}), "
                "steps(list of strings), cook_time_minutes(int)."
              )
            })
            user = []
            for f in subset:
                b64 = base64.b64encode(open(os.path.join(frames_dir,f),"rb").read()).decode()
                user.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
            return [*msgs, {"role":"user","content":user}]

        def call_gpt(prompt):
            for i in range(3):
                try:
                    return client.chat.completions.create(model="gpt-4o", messages=prompt, max_tokens=1000)
                except Exception as e:
                    if "429" in str(e):
                        time.sleep(2**i)
                        continue
                    raise
            return client.chat.completions.create(model="gpt-4o", messages=prompt, max_tokens=1000)

        # two-pass for overflow
        p1 = call_gpt(make_prompt(sample[:mid]))
        p2 = call_gpt(make_prompt(sample[mid:]))
        combined = p1.choices[0].message.content + "\n" + p2.choices[0].message.content
        js = re.search(r"```(?:json)?\s*(.*?)\s*```", combined, re.DOTALL)
        parsed = json.loads(js.group(1) if js else combined)

        # ————— EXTRACT FIELDS —————
        recipe_title      = parsed.get("title","Recipe")
        ingredients       = parsed.get("ingredients",[])
        steps             = parsed.get("steps",[])
        cook_time_minutes = parsed.get("cook_time_minutes")
        video_url_field   = tiktok_url or None

        # ————— SUMMARY IN PORTUGUÊS —————
        sum_prompt = [
          {"role":"system","content":"You are a culinary assistant."},
          {"role":"user","content":(
             "Resuma esta receita em um parágrafo em português baseado em:\n"
             f"Título: {recipe_title}\n"
             f"Ingredientes: {ingredients}\n"
             f"Passos: {steps}\n"
          )}
        ]
        sum_resp = client.chat.completions.create(model="gpt-4o", messages=sum_prompt, max_tokens=200)
        recipe_summary = sum_resp.choices[0].message.content.strip()

        return recipe_title, ingredients, steps, cook_time_minutes, video_url_field, recipe_summary

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

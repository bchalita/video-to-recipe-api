# routers/public_recipes.py

from fastapi import APIRouter, HTTPException, Query
from typing import List
import requests, json

from schemas import RecipeOut
from config import AIRTABLE_BASE_ID, AIRTABLE_API_KEY, AIRTABLE_RECIPES_TABLE, AIRTABLE_RECIPES_FEED_TABLE

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Optional
from video_processing import process_video_to_recipe
from helpers import build_airtable_fields
from config import RECIPES_ENDPOINT, HEADERS


router = APIRouter()


@router.get("", response_model=List[RecipeOut])
def list_feed(offset: str = Query(None, description="Airtable paging offset")):
    """
    OLD: GET /recipes-feed  
      ➞ NEW: GET /recipes?offset=…
    """
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {}
    if offset:
        params["offset"] = offset

    resp = requests.get(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_FEED_TABLE}",
        headers=headers,
        params=params
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Error fetching feed")

    data = resp.json()
    out: List[RecipeOut] = []
    for rec in data.get("records", []):
        f = rec["fields"]
        out.append(RecipeOut(
            id=rec["id"],
            title=f.get("Title"),
            ingredients=json.loads(f.get("Ingredients","[]")),
            steps=json.loads(f.get("Steps","[]")),
            cook_time_minutes=f.get("Cook Time Minutes"),
            video_url=f.get("Video_URL"),
            summary=f.get("Recipe Summary")
        ))
    # if Airtable returned an offset, client can pass it back
    # via ?offset=… to page
    return out


@router.get("/{recipe_id}", response_model=RecipeOut)
def get_one(recipe_id: str):
    """
    OLD: GET /recipes/{recipe_id}  ➞ NEW: GET /recipes/{recipe_id}
    """
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    resp = requests.get(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}/{recipe_id}",
        headers=headers
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Recipe not found")

    f = resp.json().get("fields", {})
    return RecipeOut(
        id=recipe_id,
        title=f.get("Title"),
        ingredients=json.loads(f.get("Ingredients","[]")),
        steps=json.loads(f.get("Steps","[]")),
        cook_time_minutes=f.get("Cook Time Minutes"),
        video_url=f.get("Video_URL"),
        summary=f.get("Recipe Summary")
    )


@router.get("/recent", response_model=List[RecipeOut])
def recent(user_id: str = Query(..., description="Airtable record ID returned from /login")):
    """
    OLD: GET /recent-recipes?user_id=…  ➞ NEW: GET /recipes/recent?user_id=…
    """
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    # 1) Fetch user → grab linked Recipes
    user_resp = requests.get(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/Users/{user_id}",
        headers=headers
    )
    if user_resp.status_code == 404:
        return []  # no uploads yet
    if user_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Error fetching user")

    recipe_ids = user_resp.json().get("fields", {}).get("Recipes", [])
    if not recipe_ids:
        return []

    # 2) Batch-fetch latest 5 by Created Time desc
    or_formula = ",".join(f"RECORD_ID()='{rid}'" for rid in recipe_ids)
    params = {
        "filterByFormula": f"OR({or_formula})",
        "sort[0][field]": "Created Time",
        "sort[0][direction]": "desc",
        "pageSize": 5
    }
    rec_resp = requests.get(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}",
        headers=headers,
        params=params
    )
    if rec_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch recent recipes")

    out = []
    for rec in rec_resp.json().get("records", []):
        f = rec["fields"]
        out.append(RecipeOut(
            id=rec["id"],
            title=f.get("Title"),
            ingredients=json.loads(f.get("Ingredients","[]")),
            steps=json.loads(f.get("Steps","[]")),
            cook_time_minutes=f.get("Cook Time Minutes"),
            video_url=f.get("Video_URL"),
            summary=f.get("Recipe Summary")
        ))
    return out


@router.post("/upload-video")
async def upload_video(
    file: UploadFile = File(None),
    tiktok_url: str = Form(None),
    user_id: Optional[str] = Form(None),
):
    try:
        title, ingredients, steps, cook_time, video_url, summary = process_video_to_recipe(
            file, tiktok_url, user_id
        )
        payload = build_airtable_fields(
            title, ingredients, steps, cook_time, video_url, recipe_summary=summary, user_id=user_id
        )
        resp = requests.post(RECIPES_ENDPOINT, headers=HEADERS, json=payload)
        if resp.status_code not in (200, 201):
            raise HTTPException(500, "Failed to save recipe")
        new_id = resp.json()["records"][0]["id"]
        return {
            "id": new_id,
            "title": title,
            "ingredients": ingredients,
            "steps": steps,
            "cook_time_minutes": cook_time,
            "video_url": video_url,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(500, str(e))

from fastapi import APIRouter, Header, File, UploadFile, Form, Depends, HTTPException
import requests

from helpers.video import process_video_to_recipe
from helpers.airtable import build_airtable_fields
from config import HEADERS, AIRTABLE_BASE_ID, AIRTABLE_API_KEY, AIRTABLE_RECIPES_TABLE, AIRTABLE_RECIPES_FEED_TABLE, ADMIN_API_KEY
from auth import get_admin_key  # your simple dependency

router = APIRouter()

@router.post("/upload-video")
def admin_upload_video(
    x_api_key: str = Header(None),
    file: UploadFile = File(None),
    tiktok_url: str = Form(None),
    _admin_key: str = Depends(get_admin_key)
):

      # 1) your existing video→recipe logic in a helper:
    title, ingredients, steps, cook_time, video_url, summary = \
        process_video_to_recipe(file, tiktok_url)

    # 2) build payload once (writes both tables)
    payload = build_airtable_fields(
      title, ingredients, steps, cook_time, video_url,
      summary=summary, user_id=None
    )

    # 3a) write to Recipes
    resp_main = requests.post(RECIPES_ENDPOINT, headers=HEADERS, json=payload)
    if resp_main.status_code not in (200,201):
        raise HTTPException(500, "Failed to save to Recipes")

    # 3b) write to RecipesFeed
    resp_feed = requests.post(FEED_ENDPOINT, headers=HEADERS, json=payload)
    if resp_feed.status_code not in (200,201):
        raise HTTPException(500, "Failed to save to RecipesFeed")

    new_id = resp_main.json()["records"][0]["id"]
    return {"id": new_id, **payload["records"][0]["fields"]}

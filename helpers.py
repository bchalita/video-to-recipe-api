import json
from typing import List, Optional, Dict
import requests
from fastapi import HTTPException


def build_airtable_fields(
    recipe_title: str,
    ingredients: List[Dict],
    steps: List[str],
    cook_time_minutes: Optional[int],
    video_url: Optional[str],
    recipe_summary: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict:
    """
    Construct the payload for creating/updating a record in Airtable.
    """
    fields: Dict = {
        "Title": recipe_title,
        "Ingredients": json.dumps(ingredients),
        "Steps": json.dumps(steps),
        "Cook Time Minutes": cook_time_minutes,
        "Video_URL": video_url,
    }
    if recipe_summary:
        fields["Recipe Summary"] = recipe_summary
    if user_id:
        # Airtable expects linked-record IDs as a list
        fields["User ID"] = [user_id]

    return {"records": [{"fields": fields}]}


def post_to_airtable(endpoint: str, payload: Dict, headers: Dict) -> Dict:
    """
    Helper to POST JSON payload to Airtable and raise if it fails.
    Returns the parsed JSON response.
    """
    resp = requests.post(endpoint, headers=headers, json=payload)
    if resp.status_code not in (200, 201):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save to Airtable ({endpoint}): {resp.text}"
        )
    return resp.json()

from fastapi import APIRouter, HTTPException
import requests, json, logging
from config import AIRTABLE_BASE_ID, AIRTABLE_API_KEY, AIRTABLE_USERS_TABLE, AIRTABLE_SAVED_RECIPES_TABLE, AIRTABLE_INTERACTIONS_TABLE

router = APIRouter()

@router.post("/save-recipe")
def save_recipe(payload: dict):
    user_id = payload.get("user_id")
    recipe = payload.get("recipe")

    if not user_id or not recipe:
        raise HTTPException(status_code=400, detail="Missing user_id or recipe")

    logger.info(f"[save-recipe] payload: {payload}")

    # Lookup User record in Airtable by ID
    user_lookup_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/Users"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }

    user_response = requests.get(
        user_lookup_url,
        headers=headers,
        params={"filterByFormula": f"RECORD_ID() = '{user_id}'"}
    )

    user_data = user_response.json()
    if "records" not in user_data or not user_data["records"]:
        logger.error("[save-recipe] No matching user in Airtable")
        raise HTTPException(status_code=404, detail="User not found in Airtable")

    airtable_user_id = user_data["records"][0]["id"]  # should be the same as user_id

    recipe_data = {
        "fields": {
            "User ID": [airtable_user_id],
            "Recipe JSON": json.dumps(recipe),
            "Title": recipe.get("title"),
            "Cook Time (Minutes)": recipe.get("cook_time_minutes"),
            "Ingredients": "\n".join([
                f"{i['quantity']} {i['name']}" for i in recipe.get("ingredients", [])
            ]),
            "Steps": "\n".join(recipe.get("steps", [])),
        }
    }

    save_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/SavedRecipes"
    save_response = requests.post(
        save_url,
        headers={
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        },
        json=recipe_data
    )

    if save_response.status_code != 200:
        logger.error(f"[save-recipe] Airtable error {save_response.status_code}: {save_response.text}")
        raise HTTPException(status_code=500, detail="Failed to save recipe")

    return {"status": "success"}

@router.get("/saved-recipes/{user_id}")
def get_saved_recipes(user_id: str):
    # 1) Load the user
    user_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}/{user_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    user_resp = requests.get(user_url, headers=headers)
    if user_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch user record")
    user_fields = user_resp.json().get("fields", {})

    # 2) Grab the list of saved-recipe record IDs
    saved_ids = user_fields.get("SavedRecipes", [])
    if not saved_ids:
        return []

    # 3) Batch fetch those recipes
    recipes_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_SAVED_RECIPES_TABLE}"
    # use the `records[]` param to pull by record ID
    params = [("records[]", rid) for rid in saved_ids]
    # only need the JSON + Title fields
    params += [("fields[]", "Recipe JSON"), ("fields[]", "Title")]
    resp = requests.get(recipes_url, headers=headers, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch saved recipes")

    # 4) Parse and return
    output = []
    for rec in resp.json().get("records", []):
        f = rec.get("fields", {})
        try:
            parsed = json.loads(f.get("Recipe JSON", "{}"))
        except Exception:
            parsed = {}
        output.append({
            "id": rec["id"],
            "title": parsed.get("title", f.get("Title")),
            "cook_time_minutes": parsed.get("cook_time_minutes"),
            "ingredients": parsed.get("ingredients"),
            "steps": parsed.get("steps")
        })
    return output


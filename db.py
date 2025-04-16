import os
import requests
import json
from fastapi import HTTPException

# Airtable configuration
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")

# Define your table names (ensure these match your Airtable schema)
USERS_TABLE = os.getenv("AIRTABLE_USERS_TABLE", "Users")
SAVED_RECIPES_TABLE = os.getenv("AIRTABLE_SAVED_RECIPES_TABLE", "SavedRecipes")

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}


def airtable_get(table: str, filter_formula: str = None):
    """
    Generic GET from Airtable with optional filterByFormula.
    """
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table}"
    params = {"filterByFormula": filter_formula} if filter_formula else {}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Airtable GET failed: {resp.text}")
    return resp.json().get("records", [])


def airtable_create(table: str, fields: dict):
    """
    Generic POST to create a record in Airtable.
    """
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table}"
    payload = {"fields": fields}
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Airtable POST failed: {resp.text}")
    return resp.json()


def check_user_exists(email: str):
    """
    Check if a user exists by email. Returns record if found, else None.
    """
    filter_formula = f"{{Email}} = '{email}'"
    records = airtable_get(USERS_TABLE, filter_formula)
    return records[0] if records else None


def create_user(email: str, name: str, password_hash: str, auth_provider: str = None):
    """
    Create a new user in Airtable. Raises if user exists.
    """
    if check_user_exists(email):
        raise HTTPException(status_code=400, detail="Email already registered")
    fields = {
        "Email": email,
        "Name": name,
        "Password": password_hash,
        "Auth Provider": auth_provider,
        "Registration Date": datetime.utcnow().isoformat()
    }
    return airtable_create(USERS_TABLE, fields)


def save_recipe(user_id: str, recipe_json: dict):
    """
    Save a recipe JSON for a user in Airtable.
    """
    fields = {
        "User ID": user_id,
        "Recipe JSON": json.dumps(recipe_json)
    }
    return airtable_create(SAVED_RECIPES_TABLE, fields)


def get_saved_recipes(user_id: str):
    """
    Retrieve saved recipes for a user from Airtable.
    """
    filter_formula = f"{{User ID}} = '{user_id}'"
    return airtable_get(SAVED_RECIPES_TABLE, filter_formula)

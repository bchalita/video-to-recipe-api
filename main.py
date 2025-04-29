import os
import re
import json
import base64
import shutil
import smtplib
import tempfile
import subprocess
from datetime import date, datetime
from typing import List, Optional
import sqlite3
import hashlib
import yt_dlp
from bs4 import BeautifulSoup
from unidecode import unidecode
from rapidfuzz import fuzz
from typing import Dict
from fastapi import Query

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker, relationship, declarative_base
from sqlalchemy import Column, String, Integer, Text, ForeignKey, create_engine
from pydantic import BaseModel
from starlette.responses import JSONResponse
from email.mime.text import MIMEText
import requests
from openai import OpenAI
import uuid
from uuid import uuid4
from schemas import UserLogin  # make sure you have UserLogin in schemas.py
import logging
import traceback, sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")

AUTH_UID_MAP: Dict[str, str] = {}


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_RECIPES_TABLE = "Recipes"
AIRTABLE_USERS_TABLE = "Users"
AIRTABLE_INTERACTIONS_TABLE = "UserInteractions"
AIRTABLE_SAVED_RECIPES_TABLE = "SavedRecipes"
AIRTABLE_RECIPES_FEED_TABLE = "RecipesFeed"

RECIPES_ENDPOINT = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
FEED_ENDPOINT = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_FEED_TABLE}"

# Common headers for Airtable
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

class RecipeIn(BaseModel):
    id: str
    title: str
    ingredients: list[str]
    steps: list[str]
    cook_time_minutes: int
    video_url: str | None = None

class RecipeOut(BaseModel):
    id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    cook_time_minutes: Optional[int]
    video_url: Optional[str]
    summary: Optional[str]

class Recipe(BaseModel):
     id: str
     title: str
     ingredients: list[str]
     steps: List[str]
     cook_time_minutes: int
     user_id: Optional[str] = None

class UserDB(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    auth_provider = Column(String, default="email")
    registration_date = Column(String)
    last_login = Column(String)
    uploaded_count = Column(Integer, default=0)
    saved_count = Column(Integer, default=0)
    recipes = relationship("RecipeDB", back_populates="user")

class RecipeDB(Base):
    __tablename__ = "recipes"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    ingredients = Column(Text, nullable=False)
    steps = Column(Text, nullable=False)
    cook_time_minutes = Column(Integer)
    user_id = Column(String, ForeignKey("users.id"))
    user = relationship("UserDB", back_populates="recipes")

Base.metadata.create_all(bind=engine)

ingredient_db_path = "ingredients.db"

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None

class UserInteraction(BaseModel):
    user_id: str
    recipe_id: str
    action: str
    timestamp: Optional[str] = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(user: UserLogin = Body(...)):
    """
    Authenticate a user against Airtable.
    """
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params  = {"filterByFormula": f"{{Email}} = '{user.email}'"}
    url     = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    resp    = requests.get(url, headers=headers, params=params)

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch user")
    records = resp.json().get("records", [])
    if not records:
        raise HTTPException(status_code=404, detail="User not found")

    airtable_record = records[0]
    record          = airtable_record["fields"]

    # verify password
    hashed_input = hashlib.sha256(user.password.encode()).hexdigest()
    if record.get("Password") != hashed_input:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # grab your “external” UUID from the linked User ID field
    real_uuid = record.get("User ID")
    if not real_uuid:
        raise HTTPException(status_code=500, detail="No external UID on user record")

    # map the Airtable recordID → your external UUID
    AUTH_UID_MAP[airtable_record["id"]] = real_uuid
    logging.info(f"[login] mapped {airtable_record['id']} → {real_uuid}")

    return {
        "success": True,
        "user_id": airtable_record["id"],
        "name": record.get("Name")
    }


    # 6. Return Airtable record ID (front-end’s user_id) and name

class UserSignup(BaseModel):
    name: str
    email: str
    password: str

@app.post("/signup")
def signup(user: UserSignup):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }
    params = {
        "filterByFormula": f"{{Email}} = '{user.email}'"
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    check = requests.get(url, headers=headers, params=params)

    if check.status_code == 200 and check.json().get("records"):
        raise HTTPException(status_code=400, detail="Email already registered")

    post_headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    user_id = str(uuid.uuid4())
    payload = {
        "fields": {
            "User ID": user_id,
            "Name": user.name,
            "Email": user.email,
            "Password": hashlib.sha256(user.password.encode()).hexdigest(),
            "Authentication Provider": "email",
            "Registration Date": str(date.today()),
            "Number of Uploaded Recipes": 0
        }
    }

    response = requests.post(url, headers=post_headers, json=payload)

    if response.status_code != 200:
        logger.error(f"[signup] Airtable error: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to create user")

    logger.info(f"[signup] New user created with ID: {user_id}")
    return {"success": True, "user_id": user_id}

@app.post("/recipes", response_model=RecipeOut)
async def upload_recipe(recipe: RecipeIn):
    # Prepare payload for Airtable
    record = {"fields": recipe.dict()}

    # Create in main Recipes table
    resp_main = requests.post(RECIPES_ENDPOINT, headers=HEADERS, json={"records": [record]})
    if resp_main.status_code != 200 and resp_main.status_code != 201:
        raise HTTPException(status_code=resp_main.status_code, detail="Failed to write to Recipes table")
    main_data = resp_main.json().get("records", [])[0]
    new_id = main_data.get("id")

    # Create in RecipesFeed table
    resp_feed = requests.post(FEED_ENDPOINT, headers=HEADERS, json={"records": [record]})
    if resp_feed.status_code != 200 and resp_feed.status_code != 201:
        # Rollback main table if desired or log error
        raise HTTPException(status_code=resp_feed.status_code, detail="Failed to write to RecipesFeed table")
    feed_data = resp_feed.json().get("records", [])[0]

    # Return the record from main table with its id
    return RecipeOut(id=new_id, **recipe.dict())
    

@app.get("/recipes/{recipe_id}", response_model=RecipeOut)
async def get_recipe(recipe_id: str):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}/{recipe_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Recipe not found")
    fields = resp.json().get("fields", {})
    return RecipeOut(
        id=recipe_id,
        title=fields.get("Title"),
        ingredients=json.loads(fields.get("Ingredients", "[]")),
        steps=json.loads(fields.get("Steps", "[]")),
        cook_time_minutes=fields.get("Cook Time Minutes"),
        video_url=fields.get("Video_URL"),
        summary=fields.get("Recipe Summary")
    )

@app.get("/recipes-feed", response_model=List[RecipeOut])
async def get_recipes_feed():
    params = {}
    records: List[RecipeOut] = []
    while True:
        resp = requests.get(FEED_ENDPOINT, headers=HEADERS, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Error fetching RecipesFeed data")
        data = resp.json()
        for rec in data.get("records", []):
            f = rec.get("fields", {})
            records.append(RecipeOut(
                title=f.get("Title"),
                ingredients=json.loads(f.get("Ingredients", "[]")),
                steps=json.loads(f.get("Steps", "[]")),
                cook_time_minutes=f.get("Cook Time Minutes"),
                video_url=f.get("Video_URL"),
                summary=f.get("Recipe Summary")
            ))
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset
    return records


@app.get("/recent-recipes")
def get_recent_recipes(
    user_id: str = Query(..., description="Airtable record ID returned from /login")
):
    """
    Fetch the last 5 recipes uploaded by this user.
    """
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    # 1) Grab the user record by its Airtable record ID
    user_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}/{user_id}"
    user_resp = requests.get(user_url, headers=headers)
    if user_resp.status_code == 404:
        raise HTTPException(404, detail="User not found")
    if user_resp.status_code != 200:
        raise HTTPException(500, detail="Error fetching user from Airtable")

    user_fields = user_resp.json().get("fields", {})
    # The linked-record field in your Users table that holds recipe-record IDs
    recipe_ids: list[str] = user_fields.get("Recipes", [])
    if not recipe_ids:
        return []  # no uploads yet

    # 2) Fetch those recipe records in one shot, sorted by creation time desc, limit 5
    recipes_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
    # Build an OR(...) formula on RECORD_ID()
    or_clauses = ",".join(f"RECORD_ID()='{rid}'" for rid in recipe_ids)
    params = {
        "filterByFormula": f"OR({or_clauses})",
        "sort[0][field]": "Created Time",
        "sort[0][direction]": "desc",
        "pageSize": 5
    }
    rec_resp = requests.get(recipes_url, headers=headers, params=params)
    if rec_resp.status_code != 200:
        raise HTTPException(500, detail="Failed to fetch recipes")

    output = []
    for record in rec_resp.json().get("records", []):
        fields = record.get("fields", {})
        # assume you still store your full JSON under “Recipe JSON”
        try:
            parsed = json.loads(fields.get("Recipe JSON", "{}"))
        except json.JSONDecodeError:
            parsed = {}
        output.append({
            "id": record["id"],
            "title": parsed.get("title", fields.get("Title", "")),
            "cook_time_minutes": parsed.get("cook_time_minutes"),
            "ingredients": parsed.get("ingredients"),
            "steps": parsed.get("steps")
            
        })

    logging.info(f"[recent-recipes] returning {len(output)} records for user {user_id}")
    return output

@app.post("/save-recipe")
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

@app.get("/saved-recipes/{user_id}")
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

@app.post("/interact")
def save_interaction(interaction: UserInteraction):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }

    # Fetch Airtable record IDs for user and recipe
    user_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    recipe_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"

    user_resp = requests.get(user_url, headers=headers, params={"filterByFormula": f'{{User ID}} = "{interaction.user_id}"'})
    recipe_resp = requests.get(recipe_url, headers=headers, params={"filterByFormula": f'{{Recipe ID}} = "{interaction.recipe_id}"'})

    if user_resp.status_code != 200 or recipe_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to lookup user or recipe record")

    user_records = user_resp.json().get("records", [])
    recipe_records = recipe_resp.json().get("records", [])

    if not user_records or not recipe_records:
        raise HTTPException(status_code=404, detail="User or Recipe not found")

    airtable_user_id = user_records[0]["id"]
    airtable_recipe_id = recipe_records[0]["id"]

    post_headers = headers.copy()
    post_headers["Content-Type"] = "application/json"

    now = interaction.timestamp or datetime.utcnow().isoformat()
    payload = {
        "fields": {
            "User ID": [airtable_user_id],
            "Recipe ID": [airtable_recipe_id],
            "Action": interaction.action,
            "Timestamp": now,
            "Unique Key": f"{interaction.user_id} - {interaction.recipe_id} - {interaction.action} - {now[:16]}"
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_INTERACTIONS_TABLE}"
    r = requests.post(url, headers=post_headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to sync interaction: {r.text}")

    return {"success": True, "airtable_id": r.json().get("id")}


@app.get("/interactions/{user_id}")
def get_user_interactions(user_id: str):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }
    params = {
        "filterByFormula": f"{{User ID}} = '{user_id}'"
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_INTERACTIONS_TABLE}"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch interactions")

    records = response.json().get("records", [])
    interactions = [
        {
            "recipe_id": record["fields"].get("Recipe ID", [None])[0],
            "action": record["fields"].get("Action"),
            "timestamp": record["fields"].get("Timestamp")
        }
        for record in records if "Recipe ID" in record["fields"]
    ]

    return {"user_id": user_id, "interactions": interactions}


def clean_gpt_json_response(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    logger.warning(f"[rappi-cart] Could not extract JSON from GPT response: {text}")
    raise ValueError("Could not extract JSON array from GPT response")

def format_unit_display(qty, unit_type):
    unit_map = {
        "kg": "kg", "g": "g", "l": "L", "ml": "ml", "unit": "un", "un": "un", "unidade": "un", "": "un"
    }
    if unit_type == "" and qty >= 50:
        return f"{qty}g"
    return f"{qty}{unit_map.get(unit_type.lower(), unit_type)}"


cached_cart_result = None
cached_last_payload = None
cached_user_id = None

# @app.post("/rappi-cart")
# def rappi_cart_search(
#     ingredients: List[str] = Body(..., embed=True),
#     recipe_title: Optional[str] = Body(None),
#     quantities: Optional[List[str]] = Body(None),
#     user_id: Optional[str] = Body(None)
# ):
#     global cached_cart_result, cached_last_payload, cached_user_id
#     cached_last_payload = {
#         "ingredients": ingredients,
#         "recipe_title": recipe_title,
#         "quantities": quantities,
#         "user_id": user_id
#     }
#     cached_user_id = user_id

#     try:
#         ingredient_override_map = {
#             "tuna": "lombo de atum"
#         }

#         if recipe_title and "tartare" in recipe_title.lower():
#             ingredients = [ingredient_override_map.get(ing.lower(), ing) for ing in ingredients]

#         prompt = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a translation assistant for a grocery shopping app.\n"
#                     "Your job is to take each English ingredient name and produce:\n"
#                     "  - A Portuguese translation suitable for Zona Sul.\n"
#                     "  - A 'search_base' — the generic noun in Portuguese to use for broader searches.\n"
#                     "  - A list of any qualifiers (adjectives or preparations) to refine searches.\n\n"
#                     "Instructions:\n"
#                     "- Preserve important adjectives (e.g., fresco, integral, vegano).\n"
#                     "- Remove true parentheticals (e.g., '(for greasing the dish)').\n"
#                     "- Extract preparations like 'minced', 'diced' into qualifiers.\n"
#                     "- If the input has no adjectives or preparation terms, qualifiers list can be empty.\n\n"
#                     "Output strict JSON array. Each item must be:\n"
#                     "{\n"
#                     '  "original": "<exact English input>",\n'
#                     '  "translated": "<full Portuguese phrase>",\n'
#                     '  "search_base": "<noun in Portuguese>",\n'
#                     '  "qualifiers": ["<qualifier1>", "<qualifier2>", ...]\n'
#                     "}\n"
#                     "Do not include any text before or after the JSON."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": f"Translate and structure the following ingredients: {json.dumps(ingredients)}"
#             }
#         ]
#         translation_response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=prompt,
#             max_tokens=300
#         )

#         translated_text = translation_response.choices[0].message.content.strip()
#         translated_list = json.loads(clean_gpt_json_response(translated_text))

#         logger.info(f"[rappi-cart] Translated ingredients: {translated_list}")

#         headers = {"User-Agent": "Mozilla/5.0"}
        
        
#         #####HIDDEN FOR NOW (JUST ADD BACK TO DICTIONARY)

#         # "Rappi - Zona Sul": "https://www.rappi.com.br/lojas/900498307-zona-sul-rio-de-janeiro/s",
#         # "Rappi - Pão de Açúcar": "https://www.rappi.com.br/lojas/900014202-pao-de-acucar-rio-de-janeiro/s",
        
        
#         store_urls = {
#             "Zona Sul": "https://www.zonasul.com.br"  # ← just the base, term will be appended later
#         }

#         store_carts = {store: [] for store in store_urls.keys()}

#         def parse_required_quantity(qty_str):
#             if not qty_str:
#                 return None, ""
#             match = re.match(r"(\d+)(\.?\d*)\\s*(g|kg|ml|l|un|unid|unidade|tbsp|tsp|cup|clove)?", qty_str.lower())
#             if match:
#                 value = float(match.group(1) + match.group(2))
#                 unit = match.group(3) or "un"
#                 factor = {"g": 1, "kg": 1000, "ml": 1, "l": 1000, "un": 1, "unid": 1, "unidade": 1, "tbsp": 1, "tsp": 1, "cup": 1, "clove": 1}.get(unit, 1)
#                 std_val = value * factor
#                 return std_val, unit
#             return None, ""

#         def extract_next_data_json(soup):
#             script = soup.find("script", {"id": "__NEXT_DATA__", "type": "application/json"})
#             if not script:
#                 logger.warning("[rappi-cart] Could not find __NEXT_DATA__ script tag")
#                 return None
#             try:
#                 return json.loads(script.string)
#             except Exception as e:
#                 logger.warning(f"[rappi-cart] Failed to parse NEXT_DATA: {e}")
#                 return None

#         def iterate_fallback_products(fallback):
#             for key, val in fallback.items():
#                 if isinstance(val, dict) and "products" in val:
#                     yield from val["products"]

#         def estimate_mass(ingredient_name, unit, value):
#             key = ingredient_name.lower()
#             table = {
#                 "un": {"onion": 200, "garlic": 5, "egg": 50, "lime": 65, "shallot": 50, "nori": 5},
#                 "tbsp": {"butter": 14, "olive oil": 13, "avocado oil": 13, "sugar": 12, "flour": 8},
#                 "tsp": {"salt": 6, "pepper": 2, "sugar": 4, "mirin": 5, "soy sauce": 5, "rice wine vinegar": 5},
#                 "clove": {"garlic": 5},
#                 "cup": {"milk": 240, "heavy cream": 240, "water": 240, "cherry tomatoes": 150}
#             }.get(unit, {})
#             return value * table.get(key, 1)
        
#         seen_items = set()

#         for idx, (original, translated) in enumerate(zip(ingredients, translated_list)):
#             if original.lower() in ["water", "água"]:
#                 continue

#             base_term = translated.split()[0]
#             search_terms = [translated]
#             # 🔍 Log the full search terms being used for this ingredient
#             logger.info(f"[rappi-cart][{original}] Search base: {search_base}, Qualifiers: {qualifiers}")

#             search_base = translated["search_base"]
#             qualifiers = translated.get("qualifiers", [])
            
#             fallback_prompt = [
#                 {
#                     "role": "system",
#                     "content": (
#                         "System: You are a search‐term generator for Zona Sul grocery items.\n"
#                         "Input is a JSON object with:\n"
#                         "  • \"search_base\" (generic noun, e.g. \"alho\")\n"
#                         "  • \"qualifiers\" (list, e.g. [\"fresco\", \"picado\"])\n\n"
#                         "Rules:\n"
#                         "  1. First term must be the **exact** full Portuguese phrase:\n"
#                         "     search_base + all qualifiers joined in natural order.\n"
#                         "  2. Second term should be **only** the search_base noun.\n"
#                         "  3. Then generate up to three more combinations of search_base + single qualifiers,\n"
#                         "     ordered by likely availability (e.g., “creme de leite fresco”, “creme de leite integral”).\n"
#                         "  4. No hard‐coded overrides—entirely driven by input fields.\n\n"
#                         "Return a JSON array of up to 5 strings."
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": json.dumps({
#                         "search_base": search_base,
#                         "qualifiers": qualifiers
#                     }, ensure_ascii=False)
#                 }
#             ]

#             fallback_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=fallback_prompt,
#                 max_tokens=100
#             )
#             fallback_text = fallback_response.choices[0].message.content.strip()
#             logger.info(f"[rappi-cart] Fallback GPT response: {fallback_text}")
            
#             try:
#                 fallback_list = json.loads(clean_gpt_json_response(fallback_text))
#                 if not isinstance(fallback_list, list):
#                     raise ValueError("Fallback is not a JSON list.")
#             except Exception as e:
#                 logger.error(f"[rappi-cart] Error parsing fallback JSON: {str(e)}")
#                 fallback_list = []
            
#             search_terms.extend(fallback_list)
            
#             # Corrected name: use search_base not base_term
#             if search_base.lower() not in [term.lower() for term in search_terms]:
#                 search_terms.append(search_base)


#             quantity_needed_raw = quantities[idx] if quantities and idx < len(quantities) else ""
#             quantity_needed_val, quantity_needed_unit = parse_required_quantity(quantity_needed_raw)
#             estimated_needed_val = estimate_mass(original, quantity_needed_unit, quantity_needed_val) if quantity_needed_val else None


#             #### NEW VERSION OF LOOP
#             for store, url in store_urls.items():
#                 found = False
        
#                 # try each search term in turn until we append one product
#                 for term in search_terms:
#                     if found:
#                         break
        
#                     # 1️⃣ Build the list of raw product dicts:
#                     product_candidates = []

#                     ### USE ZONA SUL ONLY FOR NOW
#                     if "zonasul.com.br" in url:
#                         search_url = f"https://www.zonasul.com.br/{term.replace(' ', '%20')}?_q={term.replace(' ', '%20')}&map=ft"
#                         logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] ➤ Full URL: {search_url}")
#                         response = requests.get(search_url, headers=headers, timeout=10)
#                         soup = BeautifulSoup(response.text, "html.parser")
#                         found = False
                
#                         product_blocks = soup.select("article.vtex-product-summary-2-x-element")
#                         logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] 🧱 Found {len(product_blocks)} product blocks")
                
#                         product_candidates = []
                        
#                         for product in product_blocks[:5]:
#                             name_elem = product.select_one("span.vtex-product-summary-2-x-productBrand")
#                             image_elem = product.select_one("img.vtex-product-summary-2-x-imageNormal")
#                             price_int = product.select_one("span.zonasul-zonasul-store-1-x-currencyInteger")
#                             price_frac = product.select_one("span.zonasul-zonasul-store-1-x-currencyFraction")
                        
#                             if not name_elem or not price_int:
#                                 continue
                        
#                             full_name = name_elem.get_text(strip=True)
#                             full_price = float(f"{price_int.text.strip()}.{price_frac.text.strip() if price_frac else '00'}")
#                             description = full_name.lower()
                        
#                             product_candidates.append({
#                                 "name": full_name,
#                                 "price": f"R$ {full_price:.2f}",
#                                 "description": description,
#                                 "image_url": image_elem.get("src") if image_elem else None,
#                                 "raw_block": product  # Keep the full block for later reuse
#                             })
                        
#                         if not product_candidates:
#                             logger.warning(f"[rappi-cart][{original} @ Zona Sul Direct] ❌ No viable products to evaluate with GPT.")



#                     ##### COMMENTING ON RAPPI SEARCH TO OPTIMIZE TESTING. WILL REVERT BACK LATER
#                     # else:
#                     #     # — your existing Rappi JSON logic (Pão de Açúcar) —
#                     #     response = requests.get(url, params={"term": term}, headers=headers, timeout=10)
#                     #     json_data = extract_next_data_json(BeautifulSoup(response.text, "html.parser"))
#                     #     if not json_data:
#                     #         continue
#                     #     fallback = json_data["props"]["pageProps"]["fallback"]
#                     #     for p in iterate_fallback_products(fallback):
#                     #         name = p["name"].strip()
#                     #         price = float(str(p["price"]).replace(",", "."))
#                     #         description = name.lower()
#                     #         image_raw = p.get("image", "")
#                     #         image_url = (
#                     #             image_raw
#                     #             if image_raw.startswith("http")
#                     #             else f"https://images.rappi.com.br/products/{image_raw}?e=webp&q=80&d=130x130"
#                     #             if image_raw else None
#                     #         )
#                     #         product_candidates.append({
#                     #             "name": name,
#                     #             "price": f"R$ {price:.2f}",
#                     #             "description": description,
#                     #             "image_url": image_url,
#                     #             "raw_block": p
#                     #         })
#                     #     # (log & continue if product_candidates is empty)
        
#                     if not product_candidates:
#                         logger.warning(f"[rappi-cart][{original} @ {store}] ❌ no candidates for term '{term}'")
#                         continue
        
#                     # 2️⃣ Ask GPT to pick exactly one of them:
#                     EVALUATION_PROMPT = """
#                     You are a product evaluator for a grocery shopping app.
#                     You receive:
#                     - candidates: a list of product objects, each with:
#                       - "id": string
#                       - "title": string
#                       - "department": string (example: "Frios & Laticínios")
#                     - search_base: the generic noun expected (example: "creme de leite")
#                     - qualifiers: list of descriptors (example: ["fresco"])
                    
#                     Instructions:
#                     1. Accept only products whose title contains search_base.
#                     2. If qualifiers exist, prefer products whose title contains at least one.
#                     3. Department must make sense (e.g., "creme de leite" should be in dairy, not personal care).
#                     4. Reject any sponsored, irrelevant, or obviously wrong products (e.g., toothpaste, detergent).
#                     5. If multiple good matches, pick the best overlap with search_base + qualifiers.
#                     6. If nothing fits well, return null.
                    
#                     Output strict JSON:
#                     { "chosen_id": "<best id>" }
#                     or
#                     { "chosen_id": null }
#                     """

#                     try:
#                         raw_trans = translation_response.choices[0].message.content.strip()
#                         logger.debug(f"[rappi-cart] Raw translation response: {raw_trans}")
#                         translations = json.loads(clean_gpt_json_response(raw_trans))
#                         # Expecting something like:
#                         # [
#                         #   {"original":"Heavy cream", "translated":"Creme de leite fresco",
#                         #    "search_base":"creme de leite","qualifiers":["fresco"]},
#                         #   …
#                         # ]
#                     except Exception as e:
#                         logger.error(f"[rappi-cart] ⚠️ Failed to parse translation JSON: {e}")
#                         raise HTTPException(status_code=500, detail="Translation JSON malformed")
            
#                     # Now zip over translations
#                     for idx, trans_obj in enumerate(translations):
#                         original    = trans_obj["original"]
#                         translated  = trans_obj["translated"]
#                         search_base = trans_obj["search_base"]
#                         qualifiers  = trans_obj.get("qualifiers", [])
            
#                         # … your quantity‐parsing, skip water, etc. …
            
#                         # 2️⃣ Use the JSON evaluator to pick one candidate by index
#                         eval_messages = [
#                             {"role": "system",  "content": EVALUATION_PROMPT},
#                             {"role": "user",    "content": json.dumps({
#                                 "candidates": [
#                                     {"id": i, "title": c["name"], "department": c.get("department","")}
#                                     for i, c in enumerate(product_candidates)
#                                 ],
#                                 "search_base": search_base,
#                                 "qualifiers": qualifiers
#                             })}
#                         ]
#                         eval_resp = client.chat.completions.create(
#                             model="gpt-4o",
#                             messages=eval_messages,
#                             temperature=0,
#                             max_tokens=150
#                         )
#                         raw_eval = eval_resp.choices[0].message.content.strip()
#                         logger.info(f"[{original} @ {store}] 🧠 GPT raw evaluation reply: {raw_eval}")
            
#                         try:
#                             evj = json.loads(clean_gpt_json_response(raw_eval))
#                             chosen_idx = evj.get("chosen_id")
#                             if chosen_idx is None:
#                                 raise ValueError("chosen_id null")
#                             chosen_idx = int(chosen_idx)
#                             if not (0 <= chosen_idx < len(product_candidates)):
#                                 raise IndexError(f"{chosen_idx} out of range")
#                             chosen_product = product_candidates[chosen_idx]
#                         except Exception as e:
#                             logger.warning(
#                                 f"[{original} @ {store}] ❌ Eval parse error ({e}); falling back to top result"
#                             )
#                             chosen_product = product_candidates[0]
            
#                         # 3️⃣ Compute quantity & total cost exactly as before
#                         product_name = chosen_product["name"]
#                         price = float(chosen_product["price"].replace("R$", "").replace(",", "."))
            
#                         qm = re.search(r"(\d+(?:[.,]\d+)?)(kg|g|unidade|un)", product_name.lower())
#                         if qm:
#                             val, unit = float(qm.group(1).replace(",", ".")), qm.group(2)
#                             factor = {"kg": 1000, "g": 1, "unidade": 1}.get(unit, 1)
#                             quantity_per_unit = int(val * factor)
#                         else:
#                             quantity_per_unit = 500
            
#                         if estimated_needed_val is not None:
#                             units_needed = max(1, int(estimated_needed_val // quantity_per_unit + 0.999))
#                             excess = total_quantity - estimated_needed_val
#                             needed_display = (
#                                 format_unit_display(quantity_needed_val, quantity_needed_unit)
#                                 + f" (~{int(estimated_needed_val)}g)"
#                             )
#                         else:
#                             units_needed = 1
#                             excess = None
#                             needed_display = quantity_needed_raw or ""
            
#                         total_cost = units_needed * price
#                         total_quantity = units_needed * quantity_per_unit
            
#                         key = (store, translated, product_name.lower())
#                         if key in seen_items:
#                             logger.info(f"[rappi-cart][{original} @ {store}] 🔁 Already seen: {product_name}")
#                             continue
#                         seen_items.add(key)
            
#                         store_carts[store].append({
#                             "ingredient": original,
#                             "translated": translated,
#                             "product_name": product_name,
#                             "price": f"R$ {price:.2f}",
#                             "image_url": chosen_product["image_url"],
#                             "quantity_needed": quantity_needed_raw,
#                             "quantity_needed_display": needed_display,
#                             "quantity_unit": "",
#                             "quantity_per_unit": quantity_per_unit,
#                             "display_quantity_per_unit": format_unit_display(quantity_per_unit, "g"),
#                             "units_to_buy": units_needed,
#                             "total_quantity_added": total_quantity,
#                             "total_cost": f"R$ {total_cost:.2f}",
#                             "excess_quantity": excess
#                         })
#                         logger.info(f"[rappi-cart][{original} @ {store}] ✅ Added: {product_name}")
            
#                         found = True


    
#             # 5️⃣ After we’ve exhausted all terms for this store…
#             if not found:
#                 logger.warning(f"[rappi-cart][{original} @ {store}] ⚠️ No acceptable product found for any term")
        
#         for store, items in store_carts.items():
#             logger.info(f"[Cart] Final cart for {store}: {json.dumps(items, indent=2, ensure_ascii=False)}")

#         # ▶️ Logging number of items per store
#         for store, items in store_carts.items():
#             logger.info(f"[Cart] {store} has {len(items)} items")

#         # ▶️ Fix #1 & #3: assign to local var, cache and return
#         final_cart_result = {"carts_by_store": store_carts}
#         cached_cart_result = final_cart_result
#         logger.info("[rappi-cart] Cart result cached and returned (id=%s)", id(final_cart_result))
#         return final_cart_result

#     except Exception as e:
#         logger.error(f"[rappi-cart] Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# --- Prompts ---
TRANSLATION_PROMPT = """
You are a translation assistant for a grocery shopping app.
Your job is to take each English ingredient name and produce:
  - A Portuguese translation suitable for Zona Sul.
  - A "search_base" — the generic noun in Portuguese to use for broader searches.
  - A list of any qualifiers (adjectives or preparations) to refine searches.

Instructions:
- Preserve important adjectives (e.g., fresco, integral, vegano).
- Remove true parentheticals (e.g., "(for greasing the dish)").
- Extract preparations like "minced", "diced" into qualifiers.
- If there are no adjectives or prep terms, qualifiers list can be empty.

Output strict JSON array, where each item is:
{
  "original": "<exact English input>",
  "translated": "<full Portuguese phrase>",
  "search_base": "<noun in Portuguese>",
  "qualifiers": ["<qual1>", "<qual2>", ...]
}
Do NOT wrap your output in triple-backtick code fences or annotate it with json.
Return only the raw JSON array.
"""

SEARCH_TERM_PROMPT = """
You are a search-term generator for Zona Sul grocery items.
Input is a JSON object with:
  • "search_base": generic noun, e.g. "alho"
  • "qualifiers": list of qualifiers, e.g. ["fresco", "picado"]

Rules:
1. First term must be the exact full phrase: search_base + qualifiers in natural order.
2. Second term must be only the search_base.
3. Then up to three more combos: search_base + single qualifiers,
   ordered by likely availability.
4. No hard-coded overrides.

Return a JSON array of up to 5 strings.
"""

EVALUATION_PROMPT = """
You are a product evaluator for a grocery shopping app.
You receive:
- candidates: list of objects {id, title, department}
- search_base: the noun expected, e.g. "creme de leite"
- qualifiers: list of descriptors, e.g. ["fresco"]

Instructions:
1. Accept only products whose title contains search_base.
2. If qualifiers exist, prefer those matching at least one qualifier.
3. Department must align (e.g. dairy vs personal care).
4. Reject sponsored or irrelevant items.
5. If multiple, pick best overlap with search_base + qualifiers.
6. If none, chosen_id = null.

Output strict JSON: { "chosen_id": <index|null> }
"""

# --- Helpers ---
def clean_gpt_json_response(text: str) -> str:
    # strip code fences and surrounding text
    text = text.strip()
    # remove triple backticks
    if text.startswith("```"):
        text = text.strip('`')
    return text


def parse_required_quantity(qty_str: str) -> (Optional[float], str):
    if not qty_str:
        return None, ""
    pattern = r"(\d+)(\.?\d*)\s*(g|kg|ml|l|un|unid|unidade|tbsp|tsp|cup|clove)?"
    match = re.match(pattern, qty_str.lower())
    if not match:
        return None, ""
    val = float(match.group(1) + match.group(2))
    unit = match.group(3) or "un"
    factor = {"g":1, "kg":1000, "ml":1, "l":1000, "un":1,
              "unid":1, "unidade":1, "tbsp":1, "tsp":1, "cup":1, "clove":1}[unit]
    return val * factor, unit


def estimate_mass(name: str, unit: str, value: float) -> float:
    table = {
        "un": {"onion":200, "garlic":5, "egg":50},
        "tbsp": {"butter":14, "olive oil":13},
        # add more as needed
    }
    return value * table.get(unit, {}).get(name.lower(), 1)

# --- Endpoint ---
@app.post("/rappi-cart")
def rappi_cart_search(
    ingredients: List[str] = Body(..., embed=True),
    quantities: Optional[List[str]] = Body(None),
    user_id: Optional[str] = Body(None)
):
    # 1️⃣ Translate ingredients
    payload = {"ingredients": ingredients}
    translation_resp = client.chat.completions.create(        
        model="gpt-4o",
        messages=[
            {"role": "system", "content": TRANSLATION_PROMPT},
            {
              "role": "user",
              "content": (
                  "Translate and structure the following ingredients into the JSON schema I described:\n"
                  + json.dumps(ingredients, ensure_ascii=False)
              )
            }
        ],
        max_tokens=800
    )
    raw_trans = translation_resp.choices[0].message.content.strip()
    if not raw_trans:
        logger.error("[rappi-cart] ⚠️ Empty translation response from GPT")
        raise HTTPException(status_code=500, detail="No translation returned from GPT")
    try:
        translations = json.loads(clean_gpt_json_response(raw_trans))
    except json.JSONDecodeError:
        logger.error(f"[rappi-cart] ⚠️ Failed to parse translation JSON:\n{raw_trans}")
        raise HTTPException(status_code=500, detail="Invalid JSON from translation step")
 
    logger.info(f"[rappi-cart] Translations: {translations}")

    # 2️⃣ Prepare stores
    store_urls = {"Zona Sul":"https://www.zonasul.com.br"}
    store_carts = {store: [] for store in store_urls}
    seen = set()
        # … after store_urls/store_carts …
    headers = {"User-Agent": "Mozilla/5.0"}


    # 3️⃣ Loop each translated ingredient
    for idx, trans in enumerate(translations):
        orig = trans["original"]
        full_pt = trans["translated"]
        search_base = trans["search_base"]
        qualifiers = trans.get("qualifiers", [])

        # skip water
        if orig.lower() in ["water","água"]:
            continue

        # build search terms
        search_terms = []
        # 1. full phrase
        search_terms.append(full_pt)
        # then via GPT fallback
        fallback_prompt = [
            {"role":"system","content":SEARCH_TERM_PROMPT},
            {"role":"user","content":json.dumps({"search_base":search_base,"qualifiers":qualifiers}, ensure_ascii=False)}
        ]
        fb = client.chat.completions.create(
            model="gpt-4o", messages=fallback_prompt, max_tokens=100
        ).choices[0].message.content.strip()
        try:
            fb_list = json.loads(clean_gpt_json_response(fb))
            if isinstance(fb_list,list):
                search_terms.extend(fb_list)
        except:
            logger.warning(f"Fallback parse failed for {orig}")
        # ensure search_base is present
        if search_base.lower() not in [t.lower() for t in search_terms]:
            search_terms.append(search_base)

        # parse needed mass
        qty_raw = quantities[idx] if quantities and idx < len(quantities) else ""
        quantity_needed_val, quantity_needed_unit = parse_required_quantity(qty_raw)
        estimated_needed_val = estimate_mass(orig, quantity_needed_unit, quantity_needed_val) if quantity_needed_val else None

        # 4️⃣ For each store, try terms
        for store, url in store_urls.items():
            found = False
            added = False
        
            for term in search_terms:
                if found:
                    break
        
                # 1️⃣ Build the list of raw product dicts:
                product_candidates = []
        
                if "zonasul.com.br" in url:
                    # 1️⃣ Build & log the search URL
                    search_url = f"https://www.zonasul.com.br/{term.replace(' ','%20')}?_q={term.replace(' ','%20')}&map=ft"
                    logger.info(f"[rappi-cart][{orig} @ {store}] ➤ Full URL: {search_url}")
                
                    r = requests.get(search_url, headers=headers, timeout=10)
                    soup = BeautifulSoup(r.text, "html.parser")
                
                    # 2️⃣ Grab the VTEX cards
                    cards = soup.select("article.vtex-product-summary-2-x-element")
                    logger.info(f"[rappi-cart][{orig} @ {store}] 🧱 Found {len(cards)} product cards")
                
                    if not cards:
                        logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ No cards at all – page may be JS-rendered")
                        # let fallback or next term handle it
                    for idx, card in enumerate(cards[:5]):
                        # 3️⃣ Extract name
                        name_el = (
                            card.select_one("span.vtex-product-summary-2-x-brandName") 
                            or card.select_one("h2.vtex-product-summary-2-x-productNameContainer span")
                        )
                        logger.debug(f"[{orig} @ {store}] card #{idx} — raw HTML snippet:\n{card.prettify()}")

                        if not name_el:
                            logger.debug(f"[{orig} @ {store}] card #{idx} → no name element, skipping")
                            continue
                        name = name_el.get_text(strip=True)
                        logger.debug(f"[{orig} @ {store}] card #{idx} → name: {name!r}")
                
                        # 4️⃣ Extract price parts (Zona Sul custom first, then VTEX fallback)
                        int_el = (
                            card.select_one("span.zonasul-zonasul-store-1-x-currencyInteger")
                            or card.select_one("span.vtex-product-summary-2-x-currencyInteger")
                        )
                        frac_el = (
                            card.select_one("span.zonasul-zonasul-store-1-x-currencyFraction")
                            or card.select_one("span.vtex-product-summary-2-x-currencyFraction")
                        )
                        if not int_el:
                            logger.warning(f"[{orig} @ {store}] card #{idx} → missing integer price part, skipping")
                            continue
                        int_txt = int_el.get_text(strip=True)
                        frac_txt = frac_el.get_text(strip=True) if frac_el else "00"
                        try:
                            price = float(f"{int_txt}.{frac_txt}")
                        except Exception as e:
                            logger.error(f"[{orig} @ {store}] card #{idx} → bad price parse '{int_txt}.{frac_txt}': {e}")
                            continue
                        logger.debug(f"[{orig} @ {store}] card #{idx} → price: R$ {price:.2f}")
                
                        # 5️⃣ Extract image
                        img_el = card.select_one("img.vtex-product-summary-2-x-imageNormal")
                        img = img_el["src"] if (img_el and img_el.has_attr("src")) else None
                
                        # 6️⃣ Add to candidates
                        product_candidates.append({
                            "name": name,
                            "price": f"R$ {price:.2f}",
                            "description": name.lower(),
                            "image_url": img,
                            "raw": card
                        })
                        logger.info(f"[{orig} @ {store}] candidate #{idx}: {name} — R$ {price:.2f}")
                
                    if not product_candidates:
                        logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ no candidates after scraping")
                        continue



                    
                    # — FILTER BY search_base + qualifiers —
                    filtered = []
                    sb = search_base.lower()
                    quals = [q.lower() for q in qualifiers]
                    for c in product_candidates:
                        n = c["name"].lower()
                        if sb not in n:
                            logger.debug(f"[{orig} @ {store}] dropping '{c['name']}' (no '{sb}' in name)")
                            continue
                        if quals and not any(q in n for q in quals):
                            logger.debug(f"[{orig} @ {store}] dropping '{c['name']}' (none of {quals} present)")
                            continue
                        filtered.append(c)
                    logger.info(f"[{orig} @ {store}] ▶️ {len(filtered)}/{len(product_candidates)} remain after filtering")
                    product_candidates = filtered

        
                else:
                    # — Rappi / Pão de Açúcar JSON logic goes here —
                    resp = requests.get(url, params={"term": term}, headers=headers, timeout=10)
                    json_data = extract_next_data_json(BeautifulSoup(resp.text, "html.parser"))
                    if not json_data:
                        continue
                    for p in iterate_fallback_products(json_data["props"]["pageProps"]["fallback"]):
                        name = p["name"].strip()
                        price = float(str(p["price"]).replace(",", "."))
                        image_raw = p.get("image", "")
                        image_url = (
                            image_raw
                            if image_raw.startswith("http")
                            else f"https://images.rappi.com.br/products/{image_raw}?e=webp&q=80&d=130x130"
                        )
                        product_candidates.append({
                            "name": name,
                            "price": f"R$ {price:.2f}",
                            "description": name.lower(),
                            "image_url": image_url
                        })
        
                if not product_candidates:
                    logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ no candidates for term '{term}'")
                    continue


                phase1 = [
                    c for c in product_candidates
                    if search_base.lower() in c["name"].lower()
                ]
                #   * only on the exact full-phrase pass do we enforce qualifiers
                if term == full_pt and qualifiers:
                    phase2 = [
                        c for c in phase1
                        if any(q.lower() in c["name"].lower() for q in qualifiers)
                    ]
                    product_candidates = phase2 or phase1
                else:
                    product_candidates = phase1
        
                # 2️⃣ Use the same evaluator for either source:
                eval_messages = [
                    {"role": "system", "content": EVALUATION_PROMPT},
                    {"role": "user",   "content": json.dumps({
                        "candidates": [
                            {"id": i, "title": c["name"], "department": c.get("department", "")}
                            for i, c in enumerate(product_candidates)
                        ],
                        "search_base": search_base,
                        "qualifiers": qualifiers
                    }, ensure_ascii=False)}
                ]
                eval_resp = client.chat.completions.create(
                    model="gpt-4o", messages=eval_messages, temperature=0, max_tokens=300
                )
                raw_eval = eval_resp.choices[0].message.content.strip()
                try:
                    evj = json.loads(clean_gpt_json_response(raw_eval))
                    idx = evj.get("chosen_id")
                    chosen_idx = int(idx) if idx is not None else None
                    if chosen_idx is None or not (0 <= chosen_idx < len(product_candidates)):
                        raise ValueError("no valid choice")
                    chosen_product = product_candidates[chosen_idx]
                except Exception:
                    logger.warning(f"[{orig} @ {store}] ❌ Eval failed or null – falling back to top result")
                    chosen_product = product_candidates[0]

                    # ─── extract quantity_per_unit via regex ────────────────────────────────────
                qm = re.search(r"(\d+(?:[.,]\d+)?)(kg|g|unidade|un)", chosen_product["name"].lower())
                if qm:
                    val = float(qm.group(1).replace(",", "."))
                    unit = qm.group(2)
                    factor = {"kg": 1000, "g": 1, "unidade": 1, "un": 1}.get(unit, 1)
                    quantity_per_unit = int(val * factor)
                else:
                    quantity_per_unit = 500
            
                # ─── compute how many units to buy & display string ─────────────────────────
                if estimated_needed_val is not None:
                    units_needed = max(1, int(estimated_needed_val // quantity_per_unit + 0.999))
                    needed_display = (
                        format_unit_display(quantity_needed_val, quantity_needed_unit)
                        + f" (~{int(estimated_needed_val)}g)"
                    )
                else:
                    units_needed = 1
                    needed_display = qty_raw or ""
            
                # ─── totals & excess ───────────────────────────────────────────────────────
                total_cost = units_needed * float(chosen_product["price"]
                                                  .replace("R$", "")
                                                  .replace(",", "."))
                total_quantity = units_needed * quantity_per_unit
                if estimated_needed_val is not None:
                    excess = total_quantity - estimated_needed_val
                else:
                    excess = None

                key = (store, orig, chosen_product['name'])
                if key in seen: break
                seen.add(key)

                store_carts[store].append({
                    "ingredient":orig,
                    "translated":full_pt,
                    "product_name":chosen_product['name'],
                    "price":chosen_product['price'],
                    "image_url":chosen_product['image_url'],
                    "quantity_needed":qty_raw,
                    "quantity_needed_display":f"{units_needed} x {quantity_per_unit}g",
                    "quantity_per_unit":quantity_per_unit,
                    "units_to_buy":units_needed,
                    "total_quantity_added":units_needed*quantity_per_unit,
                    "total_cost":f"R$ {total_cost:.2f}",
                    "excess_quantity":excess
                })

                logger.info(f"[rappi-cart][{orig} @ {store}] ✅ Added: {chosen_product['name']}")
                added = True
                break
            if not added:
                logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ No match found")

    # 5️⃣ Return combined carts
    return {"carts_by_store": store_carts}

        
@app.get("/rappi-cart/view")
async def view_rappi_cart():
    logger.info("[rappi-cart][view] called")
    logger.info("[rappi-cart][view] cached_cart_result: %s (id=%s)",
                "EXISTS" if cached_cart_result is not None else "NONE",
                id(cached_cart_result) if cached_cart_result is not None else "N/A")
    if cached_cart_result is None:
        logger.warning("[rappi-cart][view] No cached cart available!")
        raise HTTPException(status_code=404, detail="No cart data available.")
    # Validate structure
    carts = cached_cart_result.get("carts_by_store")
    if not carts or not isinstance(carts, dict):
        logger.warning("[rappi-cart][view] Invalid cache structure: %s", type(cached_cart_result))
        raise HTTPException(status_code=404, detail="Invalid cart data.")
    total_items = sum(len(items) for items in carts.values())
    logger.info("[rappi-cart][view] Returning cached cart: %d stores, %d total items", len(carts), total_items)
    return cached_cart_result


@app.post("/rappi-cart/reset")
def reset_rappi_cart():
    global cached_cart_result, cached_last_payload, cached_user_id
    cached_cart_result = None
    cached_last_payload = None
    cached_user_id = None
    logger.info("[rappi-cart] Cache reset")
    return {"status": "cleared"}

@app.post("/rappi-cart/resend")
def resend_rappi_cart():
    if not cached_last_payload:
        raise HTTPException(status_code=400, detail="No previous payload available")
    # Re-run with last payload
    return rappi_cart_search(**cached_last_payload)


def classify_image_multiple(images):
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
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        probabilities = prob if probabilities is None else probabilities + prob
    class_id = probabilities.argmax().item()
    print(f"[DEBUG] Voted class ID: {class_id}")
    return class_id

def get_known_ingredients_and_dishes():
    conn = sqlite3.connect(ingredient_db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM ingredients")
    ingredients = [row[0] for row in c.fetchall()]
    c.execute("SELECT name FROM dishes")
    dishes = [row[0] for row in c.fetchall()]
    conn.close()
    return ingredients, dishes

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(None),
    tiktok_url: str = Form(None),
    user_id: Optional[str] = Form(None)
):
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
        summary_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=summary_prompt,
            max_tokens=100
        )
        recipe_summary = summary_resp.choices[0].message.content.strip()
        
        fields = {
            "Title": recipe_title,
            "Ingredients": json.dumps(ingredients),
            "Steps": json.dumps(steps),
            "Cook Time Minutes": cook_time_minutes,
            "Video_URL": video_url_field,
            "Recipe JSON": json.dumps(parsed)
            "Recipe Summary": recipe_summary 
        }
        if user_id:
            fields["User ID"] = [user_id]

        payload = {"records": [{"fields": fields}]}

        # Save to Recipes table
        resp_main = requests.post(RECIPES_ENDPOINT, headers=HEADERS, json=payload)
        if resp_main.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail="Failed to save recipe to Recipes table")
        # Save to RecipesFeed table
        # parse out the new record’s ID
        main_record = resp_main.json().get("records", [])[0]
        new_id = main_record.get("id")
        
        # then write to the feed table as before
        resp_feed = requests.post(FEED_ENDPOINT, headers=HEADERS, json=payload)
        if resp_feed.status_code not in (200, 201):
            raise HTTPException(...)

        # Return without summary; frontend can fetch summary in GET
    # … after resp_feed check …
        return {
            "id": new_id,
            "title": recipe_title,
            "ingredients": ingredients,
            "steps": steps,
            "cook_time_minutes": cook_time_minutes,
            "video_url": video_url_field,
            "debug": {"frames_processed": len(frames)}
        }

    except Exception as e:
        import traceback, sys
        tb = traceback.format_exc()
        print(f"[upload-video] FULL TRACEBACK:\n{tb}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def sync_user_to_airtable(user: UserDB):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "fields": {
            "User ID": user.id,
            "Name": user.name,
            "Email": user.email,
            "Registration Date": str(user.registration_date),
            "Authentication Provider": user.auth_provider,
            "Last Login": str(user.last_login),
            "Number of Uploaded Recipes": user.uploaded_count
        }
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    r = requests.post(url, headers=headers, json=payload)
    print("Airtable user sync status:", r.status_code, r.text)

def sync_recipe_to_airtable(recipe: Recipe):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    users_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    params = {"filterByFormula": f'{{User ID}} = "{recipe.user_id}"'}
    user_response = requests.get(users_url, headers=headers, params=params)
    user_data = user_response.json()
    user_airtable_id = user_data["records"][0]["id"] if user_data.get("records") else None

    recipe_payload = {
        "fields": {
            "Title": recipe.title,
            "Cook Time (Minutes)": recipe.cook_time_minutes,
            "Ingredients": json.dumps([i.dict() for i in recipe.ingredients]),
            "Steps": json.dumps(recipe.steps),
            "Recipe ID": recipe.id,
            "User ID": [user_airtable_id] if user_airtable_id else None,
        }
    }

    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
    r = requests.post(url, headers=headers, json=recipe_payload)
    print("Airtable recipe sync status:", r.status_code, r.text)

print("✅ Tables ensured on startup")

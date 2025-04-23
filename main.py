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
from schemas import UserLogin  # make sure you have UserLogin in schemas.py
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_RECIPES_TABLE = "Recipes"
AIRTABLE_USERS_TABLE = "Users"
AIRTABLE_INTERACTIONS_TABLE = "UserInteractions"
AIRTABLE_SAVED_RECIPES_TABLE = "SavedRecipes"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

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

class Recipe(BaseModel):
    id: str
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: int
    user_id: Optional[str] = None

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
    # 1. Fetch user record by email
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }
    params = {
        "filterByFormula": f"{{Email}} = '{user.email}'"
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch user")
    records = resp.json().get("records", [])
    if not records:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Verify password
    record = records[0]["fields"]
    hashed_input = hashlib.sha256(user.password.encode()).hexdigest()
    if record.get("Password") != hashed_input:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # 3. Return success with Airtable record ID
    return {
        "success": True,
        "user_id": records[0]["id"],
        "name": record.get("Name")
    }

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
            "Cook Time (Minutes)": recipe.get("cookTimeMinutes"),
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

@app.post("/rappi-cart")
def rappi_cart_search(
    ingredients: List[str] = Body(..., embed=True),
    recipe_title: Optional[str] = Body(None),
    quantities: Optional[List[str]] = Body(None),
    user_id: Optional[str] = Body(None)
):
    global cached_cart_result, cached_last_payload, cached_user_id
    cached_last_payload = {
        "ingredients": ingredients,
        "recipe_title": recipe_title,
        "quantities": quantities,
        "user_id": user_id
    }
    cached_user_id = user_id

    try:
        ingredient_override_map = {
            "tuna": "lombo de atum"
        }

        if recipe_title and "tartare" in recipe_title.lower():
            ingredients = [ingredient_override_map.get(ing.lower(), ing) for ing in ingredients]

        prompt = [
            {"role": "system", "content": (
                "You are a food translation expert. Translate each ingredient into the common name as used in Brazilian supermarkets. "
                "Use product terminology that aligns with shopping categories (e.g., 'pasta' should become 'macarrão', not 'massa'). "
                "Return only a JSON array."
            )},
            {"role": "user", "content": f"Translate to Portuguese: {json.dumps(ingredients)}"}
        ]

        translation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=300
        )

        translated_text = translation_response.choices[0].message.content.strip()
        translated_list = json.loads(clean_gpt_json_response(translated_text))

        logger.info(f"[rappi-cart] Translated ingredients: {translated_list}")

        headers = {"User-Agent": "Mozilla/5.0"}
        store_urls = {
            "Zona Sul": "https://www.rappi.com.br/lojas/900498307-zona-sul-rio-de-janeiro/s",
            "Pão de Açúcar": "https://www.rappi.com.br/lojas/900014202-pao-de-acucar-rio-de-janeiro/s",
            "Zona Sul Direct": "https://www.zonasul.com.br"  # ← just the base, term will be appended later
        }

        store_carts = {store: [] for store in store_urls.keys()}

        def parse_required_quantity(qty_str):
            if not qty_str:
                return None, ""
            match = re.match(r"(\d+)(\.?\d*)\\s*(g|kg|ml|l|un|unid|unidade|tbsp|tsp|cup|clove)?", qty_str.lower())
            if match:
                value = float(match.group(1) + match.group(2))
                unit = match.group(3) or "un"
                factor = {"g": 1, "kg": 1000, "ml": 1, "l": 1000, "un": 1, "unid": 1, "unidade": 1, "tbsp": 1, "tsp": 1, "cup": 1, "clove": 1}.get(unit, 1)
                std_val = value * factor
                return std_val, unit
            return None, ""

        def extract_next_data_json(soup):
            script = soup.find("script", {"id": "__NEXT_DATA__", "type": "application/json"})
            if not script:
                logger.warning("[rappi-cart] Could not find __NEXT_DATA__ script tag")
                return None
            try:
                return json.loads(script.string)
            except Exception as e:
                logger.warning(f"[rappi-cart] Failed to parse NEXT_DATA: {e}")
                return None

        def iterate_fallback_products(fallback):
            for key, val in fallback.items():
                if isinstance(val, dict) and "products" in val:
                    yield from val["products"]

        def estimate_mass(ingredient_name, unit, value):
            key = ingredient_name.lower()
            table = {
                "un": {"onion": 200, "garlic": 5, "egg": 50, "lime": 65, "shallot": 50, "nori": 5},
                "tbsp": {"butter": 14, "olive oil": 13, "avocado oil": 13, "sugar": 12, "flour": 8},
                "tsp": {"salt": 6, "pepper": 2, "sugar": 4, "mirin": 5, "soy sauce": 5, "rice wine vinegar": 5},
                "clove": {"garlic": 5},
                "cup": {"milk": 240, "heavy cream": 240, "water": 240, "cherry tomatoes": 150}
            }.get(unit, {})
            return value * table.get(key, 1)

        def clean_tokens(text):
            return set(
                unidecode(text.lower())
                .replace("-", " ")
                .replace(",", " ")
                .replace(".", " ")
                .split()
            )
        
        def normalize(text):
            return unidecode(text).lower().strip()
        
        def score_match(product_name, search_terms):
            product_name_norm = normalize(product_name)
        
            if any(x in product_name_norm for x in [
                "tempero", "mistura", "combo", "kit", "pronto", "congelado", "empanado", "frito", "defumado", "instantaneo"
            ]):
                return -1  # hard reject
        
            # Generic herb check: reject mixes
            if re.search(r'\b(cheiro verde|ervas finas|ervas aromatizadas|tempero de ervas)\b', product_name_norm):
                return -1
        
            # Garlic-specific: reject 'alho poró', powders
            if "alho" in [normalize(term) for term in search_terms]:
                if "poró" in product_name_norm or "po" in product_name_norm:
                    return -1
        
            # Parsley: reject compound or dehydrated if fresh mentioned
            if "salsa" in [normalize(term) for term in search_terms]:
                if "cheiro verde" in product_name_norm or "ervas" in product_name_norm:
                    return -1
        
            # Vinegar: prefer pure types
            if "vinagre" in [normalize(term) for term in search_terms]:
                if "balsamico" in product_name_norm:
                    return 3
                if "vinho branco" in product_name_norm:
                    return 4
                if "alcool" in product_name_norm:
                    return 5
                return 2  # fallback vinegar
        
            # Olive oil
            if "azeite" in [normalize(term) for term in search_terms]:
                if "extra virgem" in product_name_norm:
                    return 5
                if "virgem" in product_name_norm:
                    return 3
                return -1  # reject other types
        
            # Score match based on overlap
            term_scores = []
            for term in search_terms:
                term_norm = normalize(term)
                if term_norm in product_name_norm:
                    term_scores.append(5)
                elif all(word in product_name_norm for word in term_norm.split()):
                    term_scores.append(3)
                elif any(word in product_name_norm for word in term_norm.split()):
                    term_scores.append(1)
        
            if not term_scores:
                return 0
            return max(term_scores)

        seen_items = set()

        for idx, (original, translated) in enumerate(zip(ingredients, translated_list)):
            if original.lower() in ["water", "água"]:
                continue

            base_term = translated.split()[0]
            search_terms = [translated]
            # 🔍 Log the full search terms being used for this ingredient
            logger.info(f"[rappi-cart][{original}] Search terms (base + fallback): {search_terms}")


            fallback_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a food product expert fluent in Brazilian Portuguese. "
                        "Strictly follow these fallback rules when given an ingredient name. "
                        "Output only a raw JSON list (no explanation, no formatting) with up to 5 product name alternatives in Brazilian Portuguese. If none apply, return [].\n\n"
            
                        "General rules:\n"
                        "- Prefer: fresh > refrigerated > shelf-stable > frozen > canned > powdered.\n"
                        "- Reject any item with: 'mistura', 'tempero', 'kit', 'combo', or seasoning blends unless explicitly requested.\n"
                        "- Reject: frozen, chopped, pre-cooked, or powdered unless context clearly allows it.\n"
                        "- Accept singular/plural/root variants (e.g., 'alho' for 'dentes de alho', 'cebola' for 'cebolas').\n"
            
                        "Category rules:\n\n"
            
                        "[HERBS]\n"
                        "- Only pure herb items. Reject: 'cheiro verde', 'ervas finas', mixed herbs.\n"
                        "- Reject powdered/dried unless fallback allows.\n"
                        "- Parsley → 'salsinha'.\n"
                        "- Basil → 'manjericão fresco' only.\n"
                        "- Thyme → only pure 'tomilho'.\n"
                        "- Rosemary, cilantro, oregano, bay leaf → same rules apply: single-ingredient, unblended.\n\n"
            
                        "[MEAT]\n"
                        "- Steak: only 'bife de contrafilé', 'filé mignon', 'alcatra', 'coxão mole'. Reject: chicken/pork/processed.\n"
                        "- Ground meat: default to 'carne moída bovina'. Accept 'suína' or 'frango' if specified.\n"
                        "- Pancetta: fallback to 'bacon em cubos' or 'fatiado'. Reject: 'presunto', 'linguiça'.\n"
                        "- Shrimp: 'camarão rosa' or 'cinza', peeled preferred. Reject: breaded/fried/precooked.\n\n"
            
                        "[DAIRY]\n"
                        "- Cream: only 'creme de leite fresco'.\n"
                        "- Sour cream → 'nata' or 'creme de leite com limão'.\n"
                        "- Heavy/double cream → 'creme de leite fresco'.\n"
                        "- Cream cheese → only if labeled. Fallback: 'requeijão cremoso' for spreading.\n"
                        "- Parmesan rind: must be solid cheese with rind. Reject grated or powder.\n\n"
            
                        "[VEGETABLES]\n"
                        "- Onion: 'cebola amarela' > 'cebola branca'. Only use 'cebola roxa' in cold recipes.\n"
                        "- Garlic: only whole cloves. Reject: 'alho poró', powder.\n"
                        "- Potatoes: only fresh whole 'batata inglesa'. Reject: frozen fries or mashed.\n"
                        "- Mushrooms: 'cogumelo paris' > 'portobello' > 'shitake' > 'ostra'. Reject 'champignon'.\n\n"
            
                        "[PANTRY ITEMS]\n"
                        "- Olive oil: only 'azeite de oliva extra virgem'.\n"
                        "- Vinegar: fallback order: 'vinagre de álcool branco' > 'vinagre de vinho branco' > 'vinagre balsâmico'.\n"
                        "- Cacao powder: only 'cacau 100%' or 'cacau alcalino'. Reject: 'achocolatado', 'Nescau'.\n"
                        "- Flour: only 'farinha de trigo'. Reject: cake mixes, 'para empanar', 'farinha de rosca'.\n"
                        "- Pasta: match format (e.g. 'espaguete', 'penne', 'fusilli'). Fallback: 'massa tipo espaguete'.\n\n"
            
                        "[CUISINE CONTEXT]\n"
                        "- Asian: prioritize 'shoyu', 'gengibre', 'óleo de gergelim', 'arroz japonês'.\n"
                        "- Italian: prioritize 'parmesão', 'muçarela', 'azeite', 'manjericão fresco'."
                    )
                }
            ]


            fallback_prompt.append({"role": "user", "content": f"Term: '{translated}'"})


            fallback_response = client.chat.completions.create(
                model="gpt-4o",
                messages=fallback_prompt,
                max_tokens=100
            )
            fallback_text = fallback_response.choices[0].message.content.strip()
            logger.info(f"[rappi-cart] Fallback GPT response: {fallback_text}")
            
            try:
                fallback_list = json.loads(clean_gpt_json_response(fallback_text))
                if not isinstance(fallback_list, list):
                    raise ValueError("Fallback is not a JSON list.")
            except Exception as e:
                logger.error(f"[rappi-cart] Error parsing fallback JSON: {str(e)}")
                fallback_list = []
            search_terms.extend(fallback_list)

            if base_term.lower() not in [term.lower() for term in search_terms]:
                search_terms.append(base_term)

            quantity_needed_raw = quantities[idx] if quantities and idx < len(quantities) else ""
            quantity_needed_val, quantity_needed_unit = parse_required_quantity(quantity_needed_raw)
            estimated_needed_val = estimate_mass(original, quantity_needed_unit, quantity_needed_val) if quantity_needed_val else None

            for store, url in store_urls.items():
                found = False
                for term in search_terms:
                    if found:
                        break
                    if "zonasul.com.br" in url:
                        try:
                            search_url = f"https://www.zonasul.com.br/{term.replace(' ', '%20')}?_q={term.replace(' ', '%20')}&map=ft"
                            logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] ➤ Full URL: {search_url}")
                            response = requests.get(search_url, headers=headers, timeout=10)
                            soup = BeautifulSoup(response.text, "html.parser")
                            found = False
                    
                            product_blocks = soup.select("article.vtex-product-summary-2-x-element")
                            logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] 🧱 Found {len(product_blocks)} product blocks")
                    
                            for product in product_blocks[:5]:
                                name_elem = product.select_one("span.vtex-product-summary-2-x-productBrand")
                                image_elem = product.select_one("img.vtex-product-summary-2-x-imageNormal")
                                price_int = product.select_one("span.zonasul-zonasul-store-1-x-currencyInteger")
                                price_frac = product.select_one("span.zonasul-zonasul-store-1-x-currencyFraction")
                    
                                if not name_elem or not price_int:
                                    logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] ⛔️ Skipping: incomplete price or name")
                                    continue
                    
                                product_name = name_elem.get_text(strip=True)
                                score = score_match(product_name, [term])


                                if score < 1:
                                    logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] ❌ Low match score ({score}) for: {product_name}")
                                    continue

                                    
                                price = float(f"{price_int.text.strip()}.{price_frac.text.strip() if price_frac else '00'}")
                                image_url = image_elem.get("src") if image_elem else None
                    
                                quantity_match = re.search(r"(\d+(?:[.,]\d+)?)(\s?)(kg|g|unidade|un)", product_name.lower())
                                if quantity_match:
                                    val = float(quantity_match.group(1).replace(",", "."))
                                    unit = quantity_match.group(3).lower()
                                    factor = {"kg": 1000, "g": 1, "un": 1, "unidade": 1}.get(unit, 1)
                                    quantity_per_unit = int(val * factor)
                                    logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] 📦 Parsed quantity: {quantity_per_unit}g from '{quantity_match.group(0)}'")
                                else:
                                    quantity_per_unit = 500
                                    logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] ⚠️ No quantity match, using default: {quantity_per_unit}g")
                    
                                units_needed = max(1, int(estimated_needed_val // quantity_per_unit + 0.999)) if estimated_needed_val else 1
                                total_cost = units_needed * price
                                total_quantity = units_needed * quantity_per_unit
                    
                                needed_display = (
                                    format_unit_display(quantity_needed_val, quantity_needed_unit)
                                    + f" (~{int(estimated_needed_val)}g)"
                                    if quantity_needed_val else quantity_needed_raw or ""
                                )
                    
                                key = (store, translated, product_name.lower())
                                if key in seen_items:
                                    logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] 🔁 Already seen: {product_name}")
                                    continue
                                seen_items.add(key)
                    
                                store_carts[store].append({
                                    "ingredient": original,
                                    "translated": translated,
                                    "product_name": product_name,
                                    "price": f"R$ {price:.2f}",
                                    "image_url": image_url,
                                    "quantity_needed": quantity_needed_raw,
                                    "quantity_needed_display": needed_display,
                                    "quantity_unit": "",
                                    "quantity_per_unit": quantity_per_unit,
                                    "display_quantity_per_unit": format_unit_display(quantity_per_unit, "g"),
                                    "units_to_buy": units_needed,
                                    "total_quantity_added": total_quantity,
                                    "total_cost": f"R$ {total_cost:.2f}",
                                    "excess_quantity": (total_quantity - estimated_needed_val) if estimated_needed_val else None
                                })
                                logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] ✅ Added: {product_name}")
                                found = True
                                break
                            if not found:
                                logger.warning(f"[rappi-cart][{original} @ Zona Sul Direct] ⚠️ No acceptable product found for term '{term}'")

                        except Exception as e:
                            logger.warning(f"[rappi-cart][{original} @ Zona Sul Direct] ❌ Error fetching products: {e}")
                        continue

                    response = requests.get(url, params={"term": term}, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, "html.parser")
                    json_data = extract_next_data_json(soup)

                    if json_data:
                        try:
                            fallback = json_data.get("props", {}).get("pageProps", {}).get("fallback", {})
                            # 🛒 Log first 5 product titles for this search term
                            preview = list(iterate_fallback_products(fallback))[:5]
                            logger.info(f"[rappi-cart][{original} @ {store} / term: {term}] First 5 fallback results: {[p.get('name') for p in preview]}")

                            for product in iterate_fallback_products(fallback):
                                title = product.get("name", "").lower()
                                price = float(str(product.get("price", "0")).replace(",", "."))
                                unit_type = product.get("unitType", "")
                                quantity_per_unit = product.get("quantity", 1)

                                term_words = term.lower().split()
                                title = product.get("name", "").lower()
                                
                                # Primary: all words must match (strict)
                                if not all(word in title for word in term_words):
                                    # Secondary: noun-only relaxed match
                                    if not term_words or term_words[0] not in title:
                                        continue

                                product_name = product.get("name", "").strip()
                                score = score_match(product_name, [term])
                                if score < 1:
                                    logger.info(f"[rappi-cart][{original} @ {store}] ❌ Low match score ({score}) for: {product_name}")
                                    continue

                                key = (store, translated, product_name)
                                if key in seen_items:
                                    continue
                                seen_items.add(key)

                                image_raw = product.get("image")
                                image_url = image_raw if image_raw and image_raw.startswith("http") else f"https://images.rappi.com.br/products/{image_raw}?e=webp&q=80&d=130x130" if image_raw else None

                                if estimated_needed_val and unit_type in ["kg", "g", "ml", "l"]:
                                    units_needed = max(1, int(estimated_needed_val // quantity_per_unit + 0.999))
                                else:
                                    units_needed = 1

                                total_cost = units_needed * price
                                total_quantity = units_needed * quantity_per_unit

                                # ----- BEGIN: normalized needed display -----
                                # Normalize unit display and estimate grams if possible
                                if quantity_needed_val is not None:
                                    estimated_needed_val = estimated_needed_val or estimate_mass(original, quantity_needed_unit, quantity_needed_val)
                                    needed_display = format_unit_display(quantity_needed_val, quantity_needed_unit)
                                    if quantity_needed_unit in ["un", "tbsp", "tsp", "cup", "clove"] and estimated_needed_val:
                                        needed_display += f" (~{int(estimated_needed_val)}g)"
                                else:
                                    needed_display = quantity_needed_raw or ""

                    # ----- END: normalized needed display -----
                                store_carts[store].append({
                                    "ingredient": original,
                                    "translated": translated,
                                    "product_name": product.get("name"),
                                    "price": f"R$ {price:.2f}",
                                    "image_url": image_url,
                                    "quantity_needed": quantity_needed_raw,
                                    "quantity_needed_display": needed_display,
                                    "quantity_unit": unit_type,
                                    "quantity_per_unit": quantity_per_unit,
                                    "display_quantity_per_unit": format_unit_display(quantity_per_unit, unit_type),
                                    "units_to_buy": units_needed,
                                    "total_quantity_added": total_quantity,
                                    "total_cost": f"R$ {total_cost:.2f}",
                                    "excess_quantity": (total_quantity - estimated_needed_val) if estimated_needed_val else None
                                })
                                # ✅ Selected product match
                                logger.info(f"[rappi-cart][{original} @ {store}] → Selected match: {product.get('name')}")
                                if not found:
                                    logger.warning(f"[rappi-cart][{original} @ {store}] ⚠️ No match found. Terms tried: {search_terms}")
                                found = True
                                break
                        except Exception as e:
                            logger.warning(f"[rappi-cart] Failed to parse fallback product info: {e}")
        
        for store, items in store_carts.items():
            logger.info(f"[rappi-cart] Final cart for {store}: {json.dumps(items, indent=2, ensure_ascii=False)}")

        # ▶️ Logging number of items per store
        for store, items in store_carts.items():
            logger.info(f"[rappi-cart] {store} has {len(items)} items")

        # ▶️ Fix #1 & #3: assign to local var, cache and return
        final_cart_result = {"carts_by_store": store_carts}
        cached_cart_result = final_cart_result
        logger.info("[rappi-cart] Cart result cached and returned (id=%s)", id(final_cart_result))
        return final_cart_result

    except Exception as e:
        logger.error(f"[rappi-cart] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

        
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

@app.get("/recent-recipes")
def get_recent_recipes(user_id: str):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {
        "filterByFormula": f"FIND('{user_id}', ARRAYJOIN({{User ID}}))",
        "sort[0][field]": "Created Time",
        "sort[0][direction]": "desc",
        "pageSize": 5
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch recent recipes")

    records = response.json().get("records", [])
    output = []
    for r in records:
        fields = r.get("fields", {})
        try:
            parsed_json = json.loads(fields.get("Recipe JSON", "{}"))
            output.append({
                "id": r["id"],
                "title": parsed_json.get("title", fields.get("Title")),
                "cook_time_minutes": parsed_json.get("cookTimeMinutes"),
                "ingredients": parsed_json.get("ingredients")
            })
        except Exception as e:
            logger.warning(f"[recent-recipes] Failed to parse recipe JSON: {e}")
            continue
    logger.info(f"[recent-recipes] Parsed output: {json.dumps(output, indent=2, ensure_ascii=False)}")
    return output

@app.get("/saved-recipes/{user_id}")
def get_saved_recipes(user_id: str):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_SAVED_RECIPES_TABLE}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"filterByFormula": f"{{User ID}} = '{user_id}'"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch saved recipes")

    records = response.json().get("records", [])
    output = []
    for r in records:
        fields = r.get("fields", {})
        try:
            parsed_json = json.loads(fields.get("Recipe JSON", "{}"))
            output.append({
                "id": r["id"],
                "title": parsed_json.get("title", fields.get("Title")),
                "cook_time_minutes": parsed_json.get("cookTimeMinutes"),
                "ingredients": parsed_json.get("ingredients")
            })
        except Exception as e:
            logger.warning(f"[saved-recipes] Failed to parse recipe JSON: {e}")
            continue
    return output

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
    description = ""
    frames = []

    try:
        if tiktok_url:
            temp_dir = tempfile.mkdtemp()
            ydl_opts = {
                "quiet": True,
                "skip_download": False,
                "outtmpl": os.path.join(temp_dir, "tiktok.%(ext)s"),
                "format": "mp4"
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(tiktok_url, download=True)
                description = info.get("description", "") or ""
                video_path = ydl.prepare_filename(info)

        if file:
            if not temp_dir:
                temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, file.filename)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        if not (tiktok_url or file):
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
        selected = frames[:90]
        mid = len(selected) // 2

        def gpt_prompt(frames_subset: List[str]):
            parts = []
            if description:
                parts.append(f"Here is the video description:\n{description}\n")
            parts.append(
                "You are an expert recipe extractor. "
                "Based on the images, output valid JSON with keys: "
                "title, ingredients (list of {name,quantity}), steps (list), cook_time_minutes (int)."
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

        first_pass = client.chat.completions.create(
            model="gpt-4o",
            messages=gpt_prompt(selected[:mid]),
            max_tokens=1000
        )
        second_pass = client.chat.completions.create(
            model="gpt-4o",
            messages=gpt_prompt(selected[mid:]),
            max_tokens=1000
        )

        combined = first_pass.choices[0].message.content.strip() + "\n" + second_pass.choices[0].message.content.strip()
        print(f"[DEBUG] Raw GPT response (first 500 chars):\n{combined[:500]}")
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", combined, re.DOTALL)

        try:
            parsed = json.loads(match.group(1).strip() if match else combined)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse GPT output: {str(e)}")

        recipe_title = parsed.get("title") or "Recipe"
        ingredients = parsed.get("ingredients")
        steps = parsed.get("steps")
        cook_time_minutes = parsed.get("cook_time_minutes")

        try:
            filename = f"{str(uuid.uuid4())}.mp4"
            shutil.copy(video_path, filename)

            payload = {
                "fields": {
                    "Title": recipe_title,
                    "Steps": json.dumps(steps),
                    "Ingredients": json.dumps(ingredients),
                    "Recipe JSON": json.dumps(parsed)
                }
            }

            if user_id:
                print(f"[DEBUG] Looking for user with UUID: {user_id}")

                headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
                params = {"filterByFormula": f"{{User ID}} = '{user_id}'"}
                user_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
                user_response = requests.get(user_url, headers=headers, params=params)

                try:
                    response_json = user_response.json()
                    print("[DEBUG] Airtable user lookup result:", json.dumps(response_json, indent=2))
                except Exception as e:
                    print(f"[ERROR] Failed to parse user response JSON: {str(e)}")

                if user_response.status_code == 200 and response_json.get("records"):
                    airtable_record_id = response_json["records"][0]["id"]
                    payload["fields"]["User ID"] = [airtable_record_id]  # ✅ Properly include linked User ID
                    print(f"[DEBUG] Found Airtable user record ID: {airtable_record_id}")
                else:
                    print("[DEBUG] No matching user found in Airtable or bad response.")

            url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
            headers = {
                "Authorization": f"Bearer {AIRTABLE_API_KEY}",
                "Content-Type": "application/json"
            }
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code not in (200, 201):
                logger.warning(f"[upload-video] Failed to save recipe to Airtable: {response.text}")
                raise HTTPException(status_code=500, detail="Failed to save recipe")

            return {
                "title": recipe_title,
                "ingredients": ingredients,
                "steps": steps,
                "cook_time_minutes": cook_time_minutes,
                "debug": {
                    "frames_processed": len(frames),
                    "model_hint": guess_id or "n/a"
                }
            }

        except Exception as e:
            logger.error(f"[upload-video] Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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

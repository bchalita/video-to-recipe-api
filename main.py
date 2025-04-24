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
        
        
        #####HIDDEN FOR NOW (JUST ADD BACK TO DICTIONARY)

        # "Rappi - Zona Sul": "https://www.rappi.com.br/lojas/900498307-zona-sul-rio-de-janeiro/s",
        # "Rappi - Pão de Açúcar": "https://www.rappi.com.br/lojas/900014202-pao-de-acucar-rio-de-janeiro/s",
        
        
        store_urls = {
            "Zona Sul": "https://www.zonasul.com.br"  # ← just the base, term will be appended later
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
                                "You are a grocery search expert fluent in Brazilian Portuguese. "
                                "Your task is to translate each English ingredient into up to 5 highly relevant Brazilian Portuguese product search terms. "
                                "Think like a picky customer typing into the search bar — you want the most accurate, buyable item on the first try. "
                                "- **First term must be the singular root** of the ingredient (e.g. 'alho' for 'garlic').  \n"
                                "- Then more specific forms ('alho fresco', 'alho descascado') if needed.  \n"
                                "- **Do not** include words that clash with non-food categories (e.g. 'dentes').  \n"
                                "Use context from the recipe to avoid irrelevant or generic items. " 
                                "For example, you know from a steak frittes recipe that you are looking for a good cut of meat "
                                "For example you know that a Tuna tartar will use the fresh tuna fish and not 'atum em lata'"
                                "For example, always prioritize the item on its own, avoiding 'mixturas' or item combinations unless explicitily prompted to"
                                "For example: do not return 'molho de tomate' for 'tomato', or 'alho poró' for 'garlic'. "
                                "Stick to fresh, common supermarket terms unless otherwise clear from context. "
                                "Given the 5 selections, be wise on which are the most relevant. "
                                "Output only a valid JSON list of 5 or fewer strings. No explanation, no formatting, no commentary."
                            )
                },
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


            #### NEW VERSION OF LOOP
            for store, url in store_urls.items():
                found = False
        
                # try each search term in turn until we append one product
                for term in search_terms:
                    if found:
                        break
        
                    # 1️⃣ Build the list of raw product dicts:
                    product_candidates = []

                    ### USE ZONA SUL ONLY FOR NOW
                    if "zonasul.com.br" in url:
                        search_url = f"https://www.zonasul.com.br/{term.replace(' ', '%20')}?_q={term.replace(' ', '%20')}&map=ft"
                        logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] ➤ Full URL: {search_url}")
                        response = requests.get(search_url, headers=headers, timeout=10)
                        soup = BeautifulSoup(response.text, "html.parser")
                        found = False
                
                        product_blocks = soup.select("article.vtex-product-summary-2-x-element")
                        logger.info(f"[rappi-cart][{original} @ Zona Sul Direct] 🧱 Found {len(product_blocks)} product blocks")
                
                        product_candidates = []
                        
                        for product in product_blocks[:5]:
                            name_elem = product.select_one("span.vtex-product-summary-2-x-productBrand")
                            image_elem = product.select_one("img.vtex-product-summary-2-x-imageNormal")
                            price_int = product.select_one("span.zonasul-zonasul-store-1-x-currencyInteger")
                            price_frac = product.select_one("span.zonasul-zonasul-store-1-x-currencyFraction")
                        
                            if not name_elem or not price_int:
                                continue
                        
                            full_name = name_elem.get_text(strip=True)
                            full_price = float(f"{price_int.text.strip()}.{price_frac.text.strip() if price_frac else '00'}")
                            description = full_name.lower()
                        
                            product_candidates.append({
                                "name": full_name,
                                "price": f"R$ {full_price:.2f}",
                                "description": description,
                                "image_url": image_elem.get("src") if image_elem else None,
                                "raw_block": product  # Keep the full block for later reuse
                            })
                        
                        if not product_candidates:
                            logger.warning(f"[rappi-cart][{original} @ Zona Sul Direct] ❌ No viable products to evaluate with GPT.")



                    ##### COMMENTING ON RAPPI SEARCH TO OPTIMIZE TESTING. WILL REVERT BACK LATER
                    # else:
                    #     # — your existing Rappi JSON logic (Pão de Açúcar) —
                    #     response = requests.get(url, params={"term": term}, headers=headers, timeout=10)
                    #     json_data = extract_next_data_json(BeautifulSoup(response.text, "html.parser"))
                    #     if not json_data:
                    #         continue
                    #     fallback = json_data["props"]["pageProps"]["fallback"]
                    #     for p in iterate_fallback_products(fallback):
                    #         name = p["name"].strip()
                    #         price = float(str(p["price"]).replace(",", "."))
                    #         description = name.lower()
                    #         image_raw = p.get("image", "")
                    #         image_url = (
                    #             image_raw
                    #             if image_raw.startswith("http")
                    #             else f"https://images.rappi.com.br/products/{image_raw}?e=webp&q=80&d=130x130"
                    #             if image_raw else None
                    #         )
                    #         product_candidates.append({
                    #             "name": name,
                    #             "price": f"R$ {price:.2f}",
                    #             "description": description,
                    #             "image_url": image_url,
                    #             "raw_block": p
                    #         })
                    #     # (log & continue if product_candidates is empty)
        
                    if not product_candidates:
                        logger.warning(f"[rappi-cart][{original} @ {store}] ❌ no candidates for term '{term}'")
                        continue
        
                    # 2️⃣ Ask GPT to pick exactly one of them:
                    gpt_prompt = [
                        {"role": "system", "content": (
                            "You're helping someone shop online for groceries in Brazil. "
                            "From the list of available products, select the **single** best match "
                            "for the ingredient mentioned. Reply only with the product name. "
                            "If none are acceptable, return 'REJECT'."
                        )},
                        {"role": "user", "content": (
                            f"Ingredient: {original}\n\n"
                            "Available products:\n" +
                            "\n".join(
                                f"{i+1}. {c['name']} — {c['price']} — {c['description']}"
                                for i, c in enumerate(product_candidates)
                            )
                        )}
                    ]
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=gpt_prompt,
                        temperature=0,
                        max_tokens=50
                    )
                    chosen_name = response.choices[0].message.content.strip()
                    logger.info(f"[{original} @ {store}] 🧠 GPT chose: {chosen_name}")
                    
                    # If GPT rejects everything, or we can't match its exact string,
                    # fall back to the very first candidate in your scraped list:
                    if chosen_name == "REJECT" or not any(p["name"] == chosen_name for p in product_candidates):
                        if chosen_name == "REJECT":
                            logger.warning(f"[{original} @ {store}] ❌ GPT rejected all products – falling back to top result")
                        else:
                            logger.warning(f"[{original} @ {store}] ⚠️ GPT result '{chosen_name}' not found – falling back to top result")
                        chosen_product = product_candidates[0]
                    else:
                        chosen_product = next(p for p in product_candidates if p["name"] == chosen_name)
                    if not chosen_product:
                        # fallback fuzzy:
                        chosen_product = next(
                            (c for c in product_candidates if chosen_name.lower() in c["name"].lower()),
                            None
                        )
                        if chosen_product:
                            logger.warning(
                                f"[rappi-cart][{original} @ {store}] ⚠️ using fuzzy fallback for '{chosen_name}'"
                            )
                        else:
                            logger.warning(
                                f"[rappi-cart][{original} @ {store}] ❌ GPT result '{chosen_name}' not found"
                            )
                            continue
        
                    # 4️⃣ Now compute quantity & cost exactly as before:
                    product_name = chosen_product["name"]
                    price = float(chosen_product["price"].replace("R$", "").replace(",", "."))
                    
                    # extract quantity_per_unit via regex on product_name or raw_block…
                    qm = re.search(r"(\d+(?:[.,]\d+)?)(kg|g|unidade|un)", product_name.lower())
                    if qm:
                        val, unit = float(qm.group(1).replace(",", ".")), qm.group(2)
                        factor = {"kg": 1000, "g": 1, "unidade": 1}.get(unit, 1)
                        quantity_per_unit = int(val * factor)
                    else:
                        quantity_per_unit = 500
                    
                    # --- EXPLICIT None check here ---
                    if estimated_needed_val is not None:
                        units_needed = max(1, int(estimated_needed_val // quantity_per_unit + 0.999))
                        excess = total_quantity - estimated_needed_val  # use below
                        # build display with the "~g" suffix
                        needed_display = (
                            format_unit_display(quantity_needed_val, quantity_needed_unit)
                            + f" (~{int(estimated_needed_val)}g)"
                        )
                    else:
                        units_needed = 1
                        excess = None
                        needed_display = quantity_needed_raw or ""
                    
                    total_cost = units_needed * price
                    total_quantity = units_needed * quantity_per_unit
                    
                    key = (store, translated, product_name.lower())
                    if key in seen_items:
                        logger.info(f"[rappi-cart][{original} @ {store}] 🔁 Already seen: {product_name}")
                        continue
                    seen_items.add(key)
                    
                    store_carts[store].append({
                        "ingredient": original,
                        "translated": translated,
                        "product_name": product_name,
                        "price": f"R$ {price:.2f}",
                        "image_url": chosen_product["image_url"],
                        "quantity_needed": quantity_needed_raw,
                        "quantity_needed_display": needed_display,
                        "quantity_unit": "",
                        "quantity_per_unit": quantity_per_unit,
                        "display_quantity_per_unit": format_unit_display(quantity_per_unit, "g"),
                        "units_to_buy": units_needed,
                        "total_quantity_added": total_quantity,
                        "total_cost": f"R$ {total_cost:.2f}",
                        "excess_quantity": excess
                    })
                    logger.info(f"[rappi-cart][{original} @ {store}] ✅ Added: {product_name}")
                    
                    found = True

        
                # 5️⃣ After we’ve exhausted all terms for this store…
                if not found:
                    logger.warning(f"[rappi-cart][{original} @ {store}] ⚠️ No acceptable product found for any term")
        
        for store, items in store_carts.items():
            logger.info(f"[Cart] Final cart for {store}: {json.dumps(items, indent=2, ensure_ascii=False)}")

        # ▶️ Logging number of items per store
        for store, items in store_carts.items():
            logger.info(f"[Cart] {store} has {len(items)} items")

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
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    # 1️⃣ Find the Airtable Users record whose "User ID" (text) = the passed UUID
    user_lookup = requests.get(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}",
        headers=headers,
        params={"filterByFormula": f"{{User ID}} = '{user_id}'"}
    )
    user_lookup.raise_for_status()
    users = user_lookup.json().get("records", [])
    if not users:
        return []   # no such user → no recipes

    airtable_user_record_id = users[0]["id"]

    # 2️⃣ Now fetch recipes whose linked-record field contains that record ID
    params = {
        "filterByFormula": f"FIND('{airtable_user_record_id}', ARRAYJOIN({{User ID}}))",
        "sort[0][field]": "Created Time",
        "sort[0][direction]": "desc",
        "pageSize": 5
    }
    resp = requests.get(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}",
        headers=headers,
        params=params
    )
    resp.raise_for_status()
    records = resp.json().get("records", [])

    output = []
    for r in records:
        fields = r["fields"]
        parsed = json.loads(fields.get("Recipe JSON","{}"))
        output.append({
            "id": r["id"],
            "title": parsed.get("title", fields.get("Title")),
            "cook_time_minutes": parsed.get("cookTimeMinutes"),
            "ingredients": parsed.get("ingredients")
        })
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

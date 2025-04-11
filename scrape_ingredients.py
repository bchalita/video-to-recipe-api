import os
import sqlite3
import requests
from bs4 import BeautifulSoup

ingredient_db_path = "ingredients.db"

if not os.path.exists(ingredient_db_path):
    print("[INIT] Creating and populating ingredients.db")
    conn = sqlite3.connect(ingredient_db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE ingredients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        type TEXT,
        common_uses TEXT
    )""")
    c.execute("""
    CREATE TABLE dishes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        region TEXT,
        notes TEXT
    )""")
    c.execute("""
    CREATE TABLE dish_ingredients (
        dish_id INTEGER,
        ingredient_id INTEGER,
        quantity TEXT,
        FOREIGN KEY (dish_id) REFERENCES dishes(id),
        FOREIGN KEY (ingredient_id) REFERENCES ingredients(id)
    )""")

    base_ingredients = [
        ("chicken", "protein", "grilled, fried, baked"),
        ("rice", "grain", "boiled, steamed"),
        ("garlic", "aromatic", "sautéing, seasoning"),
        ("ginger", "aromatic", "grated, sliced"),
        ("turmeric", "spice", "powdered, coloring"),
        ("paprika", "spice", "sprinkling, marinating"),
        ("salt", "seasoning", "universal"),
        ("pepper", "seasoning", "universal"),
        ("butter", "fat", "sautéing, baking"),
        ("olive oil", "fat", "drizzling, frying"),
        ("cream", "dairy", "sauces, soups"),
        ("coriander", "herb", "garnish, flavoring"),
        ("mayo", "condiment", "sauces, spreads"),
        ("onion", "vegetable", "sautéed, raw"),
        ("tomato", "vegetable", "raw, sauce"),
        ("avocado", "fruit", "sliced, mashed")
    ]
    c.executemany("INSERT INTO ingredients (name, type, common_uses) VALUES (?, ?, ?)", base_ingredients)

    sample_dishes = [
        ("Butter Chicken", "Indian", "Classic creamy chicken curry"),
        ("Avocado Toast", "American", "Simple breakfast dish with avocado on toast"),
        ("Fried Rice", "Chinese", "Stir-fried rice with vegetables and optionally meat or egg")
    ]
    c.executemany("INSERT INTO dishes (name, region, notes) VALUES (?, ?, ?)", sample_dishes)

    conn.commit()
    conn.close()
    print("[INIT] ingredients.db created with base schema")
else:
    print("[INIT] ingredients.db already exists")

def scrape_additional_ingredients():
    url = "https://www.bbcgoodfood.com/recipes/collection/easy-ingredient"
    print(f"[SCRAPE] Fetching: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    found = set()

    for tag in soup.find_all(['h3', 'h2']):
        text = tag.get_text(strip=True).lower()
        for word in text.split():
            if len(word) > 2:
                found.add(word)

    conn = sqlite3.connect(ingredient_db_path)
    c = conn.cursor()
    inserted = 0
    for name in found:
        try:
            c.execute("INSERT INTO ingredients (name, type, common_uses) VALUES (?, ?, ?)", (name, None, None))
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    print(f"[SCRAPE] Inserted {inserted} new ingredients")

scrape_additional_ingredients()

print("✅ Scraping and DB population complete")

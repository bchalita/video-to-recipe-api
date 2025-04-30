# config.py

import os

# ───── Database ─────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")

# ───── Airtable ─────
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_RECIPES_TABLE = "Recipes"
AIRTABLE_USERS_TABLE = "Users"
AIRTABLE_INTERACTIONS_TABLE = "UserInteractions"
AIRTABLE_SAVED_RECIPES_TABLE = "SavedRecipes"
AIRTABLE_RECIPES_FEED_TABLE = "RecipesFeed"
# (add other table names here if you use them)
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type":  "application/json",
}
RECIPES_ENDPOINT = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECIPES_TABLE}"
FEED_ENDPOINT    = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_FEED_TABLE}"

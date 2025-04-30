# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# these constants you pull from config.py
from config import (
    RECIPES_ENDPOINT,
    FEED_ENDPOINT,
    HEADERS,
    AIRTABLE_API_KEY,
    AIRTABLE_BASE_ID,
    AIRTABLE_USERS_TABLE,
    AUTH_UID_MAP,
)

# shared Pydantic types
from schemas import (
    Ingredient,
    RecipeIn,
    RecipeOut,
    UserLogin,
    UserSignup,
)

# core video→recipe logic
from video_processing import process_video_to_recipe

# helpers for building Airtable payloads
from helpers import build_airtable_fields

# your routers (each should define and export a `router` instance)
from routers.auth import router as auth_router
from routers.public_recipes import router as public_recipes_router
from routers.admin_uploads import router as admin_uploads_router
from routers.user_actions import router as user_actions_router
from routers.rappi_cart import router as rappi_cart_router

app = FastAPI()

# allow your frontend to call any endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount all of your routers
app.include_router(auth_router)             # /login, /signup
app.include_router(public_recipes_router)   # /recipes, /recipes-feed, /upload-video
app.include_router(admin_uploads_router)    # /admin/upload-video
app.include_router(user_actions_router)     # /save-recipe, /recent-recipes, /interactions, /saved-recipes
app.include_router(rappi_cart_router)       # /rappi-cart/*

# simple health-check
@app.get("/")
async def root():
    return {"status": "ok"}


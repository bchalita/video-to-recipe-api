# schemas.py

from pydantic import BaseModel
from typing import List, Optional

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None

class RecipeIn(BaseModel):
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: int
    video_url: Optional[str] = None

class RecipeOut(BaseModel):
    id: str
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: Optional[int]
    video_url: Optional[str]
    summary: Optional[str]

class UserLogin(BaseModel):
    email: str
    password: str

class UserSignup(BaseModel):
    name: str
    email: str
    password: str

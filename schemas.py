from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# User authentication schemas
class UserSignup(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# Ingredient and Recipe schemas
class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None

class RecipeSchema(BaseModel):
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: Optional[int] = None

# Payload for saving a recipe
class SaveRecipePayload(BaseModel):
    user_id: str
    recipe: RecipeSchema

# Generic Airtable record schema for responses
class AirtableRecord(BaseModel):
    id: str
    fields: Dict[str, Optional[str]]

# Schema for returned saved recipe records
class SavedRecipeRecord(AirtableRecord):
    fields: Dict[str, Optional[str]]  # includes User ID, Recipe JSON, Created At

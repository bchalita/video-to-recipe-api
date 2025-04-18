from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from db import Base
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# SQLAlchemy ORM models

class UserDB(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=True)
    auth_provider = Column(String, nullable=True)
    registration_date = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    uploaded_count = Column(Integer, default=0)
    saved_count = Column(Integer, default=0)

    recipes = relationship("RecipeDB", back_populates="user", cascade="all, delete-orphan")
    saved_recipes = relationship("SavedRecipeDB", back_populates="user", cascade="all, delete-orphan")


class RecipeDB(Base):
    __tablename__ = "recipes"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    ingredients = Column(Text, nullable=False)
    steps = Column(Text, nullable=False)
    cook_time_minutes = Column(Integer, nullable=True)
    user_id = Column(String, ForeignKey("users.id"))
    user = relationship("UserDB", back_populates="recipes")


class SavedRecipeDB(Base):
    __tablename__ = "saved_recipes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    recipe_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserDB", back_populates="saved_recipes")


# Pydantic schemas

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None


class Recipe(BaseModel):
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time_minutes: Optional[int] = None


class SavedRecipe(BaseModel):
    id: int
    user_id: str
    recipe_json: dict
    created_at: datetime


class User(BaseModel):
    id: str
    name: str
    email: str
    auth_provider: Optional[str] = None
    registration_date: Optional[datetime] = None
    last_login: Optional[datetime] = None
    uploaded_count: Optional[int] = 0
    saved_count: Optional[int] = 0


class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str

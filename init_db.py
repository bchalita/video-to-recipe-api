from scrape_ingredients import init_ingredients_db, ingredient_db_path

if __name__ == "__main__":
    init_ingredients_db(ingredient_db_path)
    print("✅ ingredients.db initialized or already exists.")

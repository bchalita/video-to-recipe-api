import os
import sqlite3

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
        notes TEXT,
        prep_time_minutes INTEGER
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
        ("beef", "protein", "grilled, minced, stewed"),
        ("salmon", "protein", "grilled, baked, pan-seared"),
        ("egg", "protein", "boiled, scrambled, poached"),
        ("rice", "grain", "boiled, steamed, fried"),
        ("pasta", "grain", "boiled, baked"),
        ("onion", "vegetable", "sautéed, raw"),
        ("garlic", "aromatic", "sautéing, seasoning"),
        ("ginger", "aromatic", "grated, sliced"),
        ("tomato", "vegetable", "raw, sauce"),
        ("lettuce", "vegetable", "salads"),
        ("avocado", "fruit", "sliced, mashed"),
        ("cucumber", "vegetable", "sliced, pickled"),
        ("carrot", "vegetable", "raw, shredded"),
        ("cheese", "dairy", "grated, melted"),
        ("cream", "dairy", "sauces, soups"),
        ("butter", "fat", "sautéing, baking"),
        ("olive oil", "fat", "drizzling, frying"),
        ("salt", "seasoning", "universal"),
        ("pepper", "seasoning", "universal"),
        ("turmeric", "spice", "powdered, coloring"),
        ("paprika", "spice", "sprinkling, marinating"),
        ("soy sauce", "condiment", "marinades, dipping"),
        ("teriyaki sauce", "condiment", "glazing, stir-fry"),
        ("mayo", "condiment", "sauces, spreads"),
        ("coriander", "herb", "garnish, flavoring"),
        ("basil", "herb", "garnish, sauces"),
        ("oregano", "herb", "pasta, pizza"),
        ("eggplant", "vegetable", "grilled, baked"),
        ("zucchini", "vegetable", "sliced, roasted"),
        ("potato", "vegetable", "boiled, mashed, fried"),
        ("bell pepper", "vegetable", "grilled, stir-fried"),
        ("lime", "fruit", "juiced, garnish"),
        ("cilantro", "herb", "fresh, garnish"),
        ("chili", "spice", "hot dishes"),
        ("yogurt", "dairy", "marinade, sauces"),
        ("spinach", "vegetable", "steamed, sautéed"),
        ("bread", "grain", "toast, sandwiches"),
        ("mushroom", "vegetable", "sautéed, baked"),
        ("parsley", "herb", "garnish, flavoring"),
        ("mustard", "condiment", "dressing, sauces"),
        ("lemon", "fruit", "zest, juice"),
        ("anchovy", "fish", "sauces, umami enhancer"),
        ("capers", "condiment", "salty, garnish"),
        ("sour cream", "dairy", "topping, sauces")
    ]
    c.executemany("INSERT INTO ingredients (name, type, common_uses) VALUES (?, ?, ?)", base_ingredients)

    extended_dishes = [
        ("Pad Thai", "Thai", "Stir-fried rice noodle dish with egg, tofu, and peanuts", 35),
        ("Fish and Chips", "British", "Fried battered fish with chips", 30),
        ("Lasagna", "Italian", "Layered pasta with cheese and meat sauce", 50),
        ("Chicken Alfredo", "Italian-American", "Creamy pasta with chicken", 30),
        ("Lamb Korma", "Indian", "Mild curry with yogurt and cream", 45),
        ("Pho", "Vietnamese", "Noodle soup with herbs and beef", 60),
        ("Ceviche", "Peruvian", "Raw fish marinated in citrus juice", 25),
        ("Miso Soup", "Japanese", "Soup with miso paste, tofu and seaweed", 15),
        ("Tuna Salad", "American", "Cold salad with tuna and vegetables", 10),
        ("Pancakes", "American", "Fluffy breakfast cakes", 20),
        ("Chicken Parmesan", "Italian-American", "Breaded chicken with marinara and cheese", 35),
        ("Quiche Lorraine", "French", "Egg pie with bacon and cheese", 45),
        ("Fettuccine Carbonara", "Italian", "Creamy pasta with egg and bacon", 25),
        ("Moussaka", "Greek", "Layered eggplant and meat casserole", 60),
        ("Gazpacho", "Spanish", "Cold tomato soup", 15),
        ("Biryani", "Indian", "Spiced rice dish with meat or vegetables", 60),
        ("Croque Monsieur", "French", "Grilled ham and cheese sandwich", 15),
        ("Pulled Pork Sandwich", "American", "Slow cooked pork in a sandwich", 50),
        ("Tempura", "Japanese", "Lightly battered and fried seafood and vegetables", 30),
        ("Paella", "Spanish", "Rice dish with seafood and spices", 60),
        ("Shawarma", "Middle Eastern", "Sliced meat wrapped in pita", 40),
        ("Tuna Poke Bowl", "Hawaiian", "Rice bowl with raw tuna and toppings", 20),
        ("Beef Wellington", "British", "Beef fillet baked in puff pastry", 75),
        ("Chicken Shawarma Bowl", "Middle Eastern", "Chicken with rice, salad, and sauces", 35),
        ("Risotto", "Italian", "Creamy rice dish with broth and cheese", 40),
        ("Huevos Rancheros", "Mexican", "Eggs on tortilla with salsa", 20),
        ("Banh Mi", "Vietnamese", "Baguette sandwich with meat and pickled veggies", 25),
        ("Okonomiyaki", "Japanese", "Cabbage pancake with meat or seafood", 30),
        ("Chili Con Carne", "Tex-Mex", "Spicy stew with beans and beef", 45),
        ("Sweet and Sour Chicken", "Chinese", "Fried chicken with sweet sauce", 30),
        ("Mac and Cheese", "American", "Baked pasta with cheese sauce", 30),
        ("Spaghetti Bolognese", "Italian", "Meat-based pasta sauce", 40),
        ("Beef Stroganoff", "Russian", "Beef in creamy mushroom sauce", 35),
        ("Tacos", "Mexican", "Corn tortillas with meat and toppings", 25),
        ("Gnocchi", "Italian", "Soft dough dumplings with sauce", 30),
        ("Chicken Tikka Masala", "Indian", "Marinated chicken in spiced curry sauce", 45),
        ("Shakshuka", "Middle Eastern", "Eggs poached in tomato sauce", 25),
        ("Falafel Wrap", "Middle Eastern", "Chickpea patties in wrap", 20),
        ("Stuffed Peppers", "Mediterranean", "Peppers filled with rice and meat", 50),
        ("Ratatouille", "French", "Stewed vegetable dish", 40),
        ("Clam Chowder", "American", "Creamy seafood soup", 40),
        ("Enchiladas", "Mexican", "Rolled tortillas with sauce", 45),
        ("Katsu Curry", "Japanese", "Breaded cutlet with curry sauce", 35),
        ("Bibimbap", "Korean", "Rice bowl with vegetables and egg", 35),
        ("Roast Chicken", "Universal", "Oven-roasted whole chicken", 90),
        ("Goulash", "Hungarian", "Stew of meat and vegetables", 60),
        ("Fried Chicken", "American", "Deep-fried seasoned chicken", 40),
        ("Pasta Primavera", "Italian-American", "Pasta with fresh vegetables", 30),
        ("Egg Fried Rice", "Chinese", "Stir-fried rice with egg", 20)
    ]
    c.executemany("INSERT INTO dishes (name, region, notes, prep_time_minutes) VALUES (?, ?, ?, ?)", extended_dishes)

    conn.commit()
    conn.close()
    print("[INIT] ingredients.db created with base schema and extended recipes")
else:
    print("[INIT] ingredients.db already exists")

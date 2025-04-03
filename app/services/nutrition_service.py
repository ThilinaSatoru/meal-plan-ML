import csv

from app.database import db
from app.models import Nutrition, Meal


def get_all():
    results = Nutrition.query.all()
    return [nutrition.to_dict() for nutrition in results]


def insert_from_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # Clean the keys by stripping extra spaces
            row = {key.strip(): value for key, value in row.items()}

            # First, check if the MealID exists in the Meal table
            meal = Meal.query.get(row['MealID'])

            if meal:
                # Create a new Nutrition record if the MealID exists
                new_nutrition = Nutrition(
                    meal_id=row['MealID'],
                    glycemic_index=int(row['GlycemicIndex']),
                    calories=int(row['Calories']),
                    carbohydrates=float(row['Carbohydrates']),
                    protein=float(row['Protein']),
                    fats=float(row['Fats']),
                    fiber=float(row['Fiber']),
                    sodium=int(row['Sodium']),
                    sugar=float(row['Sugar'])
                )
                # Add the new nutrition record to the session
                db.session.add(new_nutrition)
            else:
                print(f"Meal with ID {row['MealID']} does not exist. Skipping nutrition entry.")

        # Commit all nutrition entries at once
        db.session.commit()

    return "Nutrition data added successfully from CSV!"

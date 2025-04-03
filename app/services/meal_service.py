import csv

from app.database import db
from app.models import Meal


def get_all_meals():
    results = Meal.query.all()
    return [meal.to_dict() for meal in results]


def add_meals_from_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # Clean the keys by stripping extra spaces
            row = {key.strip(): value for key, value in row.items()}
            # Create a new meal from each row in the CSV
            new_meal = Meal(
                meal_id=row['MealID'],
                meal_name=row['MealName'],
                meal_type=row['Type'],
                meal_details=row['MealDetails'],
                calories=int(row['Calories']),
                carbs=float(row['Carbs(g)']),
                protein=float(row['Protein(g)']),
                fiber=float(row['Fiber(g)']),
                glycemic_load=row['GlycemicLoad'],
                allergy_status=row['AllergyStatus'],
                preferences=row['Preferences']
            )
            print(new_meal)
            # Add the new meal to the session
            db.session.add(new_meal)

        # Commit all meals at once
        db.session.commit()

    return "Meals added successfully from CSV!"

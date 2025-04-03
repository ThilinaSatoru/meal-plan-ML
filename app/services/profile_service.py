import csv

from app.database import db
from app.models import Profile, Meal


def get_all():
    profiles = Profile.query.all()
    return [profile.to_dict() for profile in profiles]


def insert_from_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # Clean the keys by stripping extra spaces
            row = {key.strip(): value for key, value in row.items()}

            # First, check if the MealID exists in the Meal table
            meal = Meal.query.get(row['MealID'])

            if meal:
                # Create a new Profile if the MealID exists
                new_profile = Profile(
                    age=int(row['Age']),
                    gender=row['Gender'],
                    height=float(row['Height']),
                    weight=float(row['Weight']),
                    location=row['Location'],
                    fbs=row['FBS'],
                    bmi=float(row['BMI']),
                    hba1c=row['HbA1c'],
                    diagnosed_years_ago=int(row['DiagnosedYearsAgo']),
                    fasting_glucose=int(row['FastingGlucose']),
                    postprandial_glucose=int(row['PostprandialGlucose']),
                    other_conditions=row['OtherConditions'],
                    favorite_foods=row['FavoriteFoods'],
                    foods_avoided=row['FoodsAvoided'],
                    diet_followed=row['DietFollowed'],
                    trigger_foods=row['TriggerFoods'],
                    allergies=row['Allergies'],
                    traditional_foods=row['TraditionalFoods'],
                    cooking_frequency=row['CookingFrequency'],
                    cooking_methods=row['CookingMethods'],
                    meal_id=row['MealID']  # Link to the Meal table
                )
                # Add the new profile to the session
                db.session.add(new_profile)

            else:
                print(f"Meal with ID {row['MealID']} does not exist.")

        # Commit all profiles at once
        db.session.commit()

    return "Profiles added successfully from CSV!"

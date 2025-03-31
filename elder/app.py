import pickle
import random
import warnings
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Constants
ENCODER_PATH = 'encoder_meal.pkl'
MODEL_PATH = 'model_meal.pkl'
MEAL_DATA_PATH = 'sri_lankan_meal_dataset.csv'
USER_DATA_PATH = 'diabetes_user_profiles_with_mealID.csv'
NUTRITION_PATH = 'E:\\PartProjects\\MealPlan\\meal plan recommendation\\elder\\nutrition.csv'

# Match the feature definitions exactly with the training code
USER_CAT_FEATURES = [
    'Gender', 'Location', 'FBS', 'BMI', 'HbA1c',
    'FavoriteFoods', 'TraditionalFoods',
    'CookingFrequency', 'CookingMethods'
]

USER_NUM_FEATURES = [
    'Age', 'Height', 'Weight', 'DiagnosedYearsAgo',
    'FastingGlucose', 'PostprandialGlucose'
]

MEAL_FEATURES = [
    'Calories', 'Carbs(g)', 'Protein(g)', 'Fiber(g)',
    'GlycemicLoad', 'AllergyStatus', 'Preferences'
]

app = Flask(__name__)
swagger = Swagger(app)


class MealRecommender:
    def __init__(self):
        self.encoder = self._load_encoder()
        self.model = self._load_model()
        self.meal_plans = self._load_meal_plans()
        self.user_data = self._load_user_data()
        # Get default values from user data for missing features
        self.default_values = self._get_default_values()

    def _load_encoder(self) -> defaultdict:
        with open(ENCODER_PATH, 'rb') as f:
            return defaultdict(LabelEncoder, pickle.load(f))

    def _load_model(self):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)

    def _load_nutrition_data(self) -> Dict[int, Dict]:
        df = pd.read_csv(NUTRITION_PATH)
        return df.set_index('MealID').to_dict('index')

    def _load_meal_plans(self) -> Dict[int, List[Dict]]:
        df = pd.read_csv(MEAL_DATA_PATH)
        column_mapping = {
            'CalorieCount': 'Calories',
            'Carbohydrates': 'Carbs(g)',
            'Fiber': 'Fiber(g)',
            'Protein': 'Protein(g)',
            'Fats': 'Fat(g)',
            # Add any other needed mappings
        }
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # Group by MealID and convert to dictionary
        return df.groupby('MealID').apply(lambda x: x.to_dict('records')).to_dict()

    def _load_user_data(self) -> pd.DataFrame:
        return pd.read_csv(USER_DATA_PATH)

    def _get_default_values(self) -> Dict:
        """Get most common values from training data to use as defaults"""
        defaults = {}
        for col in USER_CAT_FEATURES:
            if col in self.user_data.columns:
                # Get most common non-empty value
                valid_vals = self.user_data[col][self.user_data[col].notna() & (self.user_data[col] != '')]
                if not valid_vals.empty:
                    defaults[col] = valid_vals.mode()[0]
                else:
                    defaults[col] = self.encoder[col].classes_[0] if col in self.encoder else 'Unknown'

        for col in USER_NUM_FEATURES:
            if col in self.user_data.columns:
                defaults[col] = float(self.user_data[col].median())

        return defaults

    def preprocess_user(self, user_input: Dict) -> pd.DataFrame:
        """Convert raw user input to model-ready format"""
        # Create a copy of user_input to avoid modifying the original
        processed_input = user_input.copy()

        # Create DataFrame with all required columns
        df = pd.DataFrame([processed_input])

        # Handle numeric features first - CRITICAL for CatBoost
        for feature in USER_NUM_FEATURES:
            if feature not in processed_input or processed_input.get(feature) == '' or pd.isna(
                    processed_input.get(feature)):
                df[feature] = self.default_values.get(feature, 0.0)
            else:
                # Ensure numeric features are float values
                try:
                    df[feature] = float(processed_input.get(feature, 0.0))
                except (ValueError, TypeError):
                    df[feature] = self.default_values.get(feature, 0.0)

        # Calculate BMI if needed and possible
        if ('BMI' not in df.columns or pd.isna(df['BMI'][0]) or df['BMI'][
            0] == '') and 'Height' in df.columns and 'Weight' in df.columns:
            if df['Height'][0] > 0:  # Avoid division by zero
                df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
            else:
                df['BMI'] = self.default_values.get('BMI', 25.0)  # Default BMI

        # Now handle categorical features
        for feature in USER_CAT_FEATURES:
            if feature == 'BMI' and feature in df.columns and not pd.isna(df[feature][0]):
                # BMI is special - it's categorical in the model but numeric in nature
                continue

            if feature not in processed_input or processed_input.get(feature) == '' or pd.isna(
                    processed_input.get(feature)):
                # Use default value
                if feature in self.default_values:
                    value = self.default_values[feature]
                elif feature in self.encoder and len(self.encoder[feature].classes_) > 0:
                    # Use first class from encoder
                    value = self.encoder[feature].classes_[0]
                else:
                    # Last resort - this shouldn't happen if encoders are properly set up
                    value = "Unknown"
            else:
                value = processed_input.get(feature)

            # Encode the value
            if feature in self.encoder:
                if value in self.encoder[feature].classes_:
                    df[feature] = self.encoder[feature].transform([value])[0]
                else:
                    # Use the first class if value is not in known classes
                    df[feature] = self.encoder[feature].transform([self.encoder[feature].classes_[0]])[0]

        # Verify all required columns are present
        for col in USER_NUM_FEATURES + USER_CAT_FEATURES:
            if col not in df.columns:
                if col in USER_NUM_FEATURES:
                    df[col] = self.default_values.get(col, 0.0)
                else:
                    if col in self.encoder and len(self.encoder[col].classes_) > 0:
                        df[col] = self.encoder[col].transform([self.encoder[col].classes_[0]])[0]
                    else:
                        # We should never reach here, but just in case
                        df[col] = 0

        # Ensure all numeric features are float
        for col in USER_NUM_FEATURES:
            df[col] = df[col].astype(float)

        # Final check - fill any remaining NaNs
        df = df.fillna(0)

        return df[USER_NUM_FEATURES + USER_CAT_FEATURES]

    def filter_allergies(self, meal: Dict, user_allergies: List[str]) -> bool:
        """Check if meal contains user's allergens"""
        if not user_allergies or not meal or meal.get('AllergyStatus', 'None') == 'None':
            return True
        meal_allergens = [a.strip() for a in str(meal.get('AllergyStatus', '')).split(',')]
        return not any(allergen in meal_allergens for allergen in user_allergies)

    def recommend(self, user_input: Dict) -> Dict[str, Any]:
        try:
            # Preprocess
            user_df = self.preprocess_user(user_input)
            nutrition_data = self._load_nutrition_data()

            # Debug info
            print(f"Preprocessed user data: {user_df.to_dict()}")
            print(f"Data types: {user_df.dtypes}")

            # Predict
            proba = self.model.predict_proba(user_df)[0]

            # Get meal IDs and make sure they're properly encoded
            meal_ids = list(self.meal_plans.keys())
            suitable_plans = []

            for meal_id in meal_ids:
                try:
                    encoded_id = self.encoder['MealID'].transform([meal_id])[0]
                    if encoded_id < len(proba) and proba[encoded_id] >= 0.1:
                        suitable_plans.append(meal_id)
                except:
                    # Skip meal IDs that can't be encoded
                    continue

            if not suitable_plans:
                suitable_plans = list(self.meal_plans.keys())[:3]  # Fallback

            # Filter by preferences/allergies
            user_allergies = [a.strip() for a in user_input.get('Allergies', '').split(',') if a.strip()]
            user_diet = user_input.get('DietFollowed', '')

            valid_plans = []
            for plan_id in suitable_plans:
                if plan_id not in self.meal_plans:
                    continue

                meals = self.meal_plans[plan_id]
                if all(
                        self.filter_allergies(m, user_allergies) and
                        (not user_diet or str(m.get('Preferences', '')).lower() == user_diet.lower())
                        for m in meals
                ):
                    valid_plans.append(plan_id)

            # Select best match
            if valid_plans:
                chosen_plan = random.choice(valid_plans)
            elif suitable_plans:
                chosen_plan = random.choice(suitable_plans)
            elif self.meal_plans:
                chosen_plan = list(self.meal_plans.keys())[0]  # Last resort
            else:
                return {"error": "No meal plans available"}

            plan_meals = self.meal_plans[chosen_plan]

            print("Available meal data fields:")
            if plan_meals and len(plan_meals) > 0:
                print(f"Sample meal keys: {list(plan_meals[0].keys())}")
                print(f"Sample meal data: {plan_meals[0]}")

            # Calculate nutrition safely
            nutrition = {
                'TotalCalories': sum(
                    float(m.get('CalorieCount', nutrition_data.get(m.get('MealID', 0), {}).get('Calories', 0))) for m in
                    plan_meals),
                'NetCarbs(g)': sum(float(nutrition_data.get(m.get('MealID', 0), {}).get('Carbohydrates', 0) -
                                         nutrition_data.get(m.get('MealID', 0), {}).get('Fiber', 0)) for m in
                                   plan_meals),
                'Protein(g)': sum(
                    float(nutrition_data.get(m.get('MealID', 0), {}).get('Protein', 0)) for m in plan_meals)
            }

            print(f"Nutrition calculation: {nutrition}")
            print(f"Raw meal data: {plan_meals}")

            # Get plan name safely
            plan_name = "Meal Plan"
            if plan_meals and 'MealName' in plan_meals[0]:
                meal_name = plan_meals[0]['MealName']
                if isinstance(meal_name, str) and ' Plan' in meal_name:
                    plan_name = meal_name.split(' Plan')[0]
                else:
                    plan_name = str(meal_name)

            return {
                'MealPlanID': chosen_plan,
                'PlanName': plan_name,
                'Nutrition': nutrition,
                'Meals': plan_meals,
                'UserProfileMatch': {
                    'GlycemicLoad': user_input.get('GlycemicPreference', 'Medium'),
                    'AllergiesAvoided': user_allergies,
                    'DietaryPreference': user_diet
                }
            }

        except Exception as e:
            import traceback
            print(f"Error in recommend: {e}")
            print(traceback.format_exc())
            return {"error": str(e), "details": traceback.format_exc()}


recommender = MealRecommender()


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommend personalized meal plan based on user profile
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Age:
              type: number
              example: 45
            Gender:
              type: string
              example: Male
            Height:
              type: number
              example: 175
            Weight:
              type: number
              example: 80
            FastingGlucose:
              type: number
              example: 130
            Allergies:
              type: string
              example: Dairy,Nuts
            DietFollowed:
              type: string
              example: Low-carb
    responses:
      200:
        description: Successful response
      400:
        description: Bad request
      500:
        description: Internal server error
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input provided'}), 400

        # Add default values for required fields
        data.setdefault('Allergies', '')
        data.setdefault('DietFollowed', '')

        # Print request data for debugging
        print(f"Received request data: {data}")

        result = recommender.recommend(data)
        return jsonify(result)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {e}\nDetails: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


if __name__ == '__main__':
    # Print loaded model and encoder info on startup
    print("Initializing meal recommender service...")
    try:
        print(f"Encoder loaded with {len(recommender.encoder)} features")
        print(f"Categorical features: {USER_CAT_FEATURES}")
        print(f"Numerical features: {USER_NUM_FEATURES}")
        print(f"Default values: {recommender.default_values}")
        print(f"Model type: {type(recommender.model).__name__}")
        print("Service initialized successfully!")
    except Exception as e:
        print(f"Error during initialization: {e}")

    app.run(host='0.0.0.0', port=5000)
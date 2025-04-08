import pickle
import pickle
import random
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from flask import jsonify
from sklearn.preprocessing import LabelEncoder

# Constants
ENCODER_PATH = 'E:\\PartProjects\\MealPlan\\meal plan recommendation\\app\\services\\AI\\encoder_meal.pkl'
MODEL_PATH = 'E:\\PartProjects\\MealPlan\\meal plan recommendation\\app\\services\\AI\\model_meal.pkl'
MEAL_DATA_PATH = 'E:\\PartProjects\\MealPlan\\meal plan recommendation\\dataset\\new_meals.csv'
USER_DATA_PATH = 'E:\\PartProjects\\MealPlan\\meal plan recommendation\\dataset\\diabetes_user_profiles_with_mealID.csv'
NUTRITION_PATH = 'E:\\PartProjects\\MealPlan\\meal plan recommendation\\dataset\\nutrition.csv'

# Match the feature definitions exactly with the training code
USER_CAT_FEATURES = [
    'Gender', 'Location', 'FBS', 'BMI', 'HbA1c',
    'FavoriteFoods', 'TraditionalFoods',
    'CookingFrequency', 'CookingMethods', 'Allergies'
]

USER_NUM_FEATURES = [
    'Age', 'Height', 'Weight', 'DiagnosedYearsAgo',
    'FastingGlucose', 'PostprandialGlucose'
]

MEAL_FEATURES = [
    'Calories', 'Carbs(g)', 'Protein(g)', 'Fiber(g)',
    'GlycemicLoad', 'AllergyStatus', 'Preferences'
]


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

    def sanitize_meal_data(self, meals):
        """Convert non-JSON-serializable values to appropriate formats"""
        sanitized_meals = []
        for meal in meals:
            sanitized_meal = {}
            for key, value in meal.items():
                # Handle NaN values
                if isinstance(value, float) and pd.isna(value):
                    sanitized_meal[key] = None
                else:
                    sanitized_meal[key] = value
            sanitized_meals.append(sanitized_meal)
        return sanitized_meals

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
        # for feature in USER_CAT_FEATURES:
        #     if feature == 'BMI' and feature in df.columns and not pd.isna(df[feature][0]):
        #         # BMI is special - it's categorical in the model but numeric in nature
        #         continue
        #
        #     if feature not in processed_input or processed_input.get(feature) == '' or pd.isna(processed_input.get(feature)):
        #         # Use default value
        #         if feature in self.default_values:
        #             value = self.default_values[feature]
        #         elif feature in self.encoder and len(self.encoder[feature].classes_) > 0:
        #             # Use first class from encoder
        #             value = self.encoder[feature].classes_[0]
        #         else:
        #             # Last resort - this shouldn't happen if encoders are properly set up
        #             value = "Unknown"
        #     else:
        #         value = processed_input.get(feature)
        #
        #     user_allergies = user_input.get('Allergies', [])
        #     if isinstance(user_allergies, str):
        #         user_allergies = user_allergies.split(',')
        #     user_allergies = [a.strip() for a in user_allergies if a.strip()]
        #
        #     if feature == 'Allergies':
        #         # If allergies exist in encoder, encode them properly
        #         if feature in self.encoder:
        #             allergy_value = processed_input.get(feature, '')
        #             # Check if this exact allergy is in the encoder classes
        #             if allergy_value in self.encoder[feature].classes_:
        #                 df[feature] = self.encoder[feature].transform([allergy_value])[0]
        #             else:
        #                 # Use a default value from the encoder
        #                 df[feature] = self.encoder[feature].transform([self.encoder[feature].classes_[0]])[0]
        #         else:
        #             # If not in encoder, use a numeric placeholder
        #             df[feature] = 0
        #
        #     # Encode the value
        #     if feature in self.encoder:
        #         if value in self.encoder[feature].classes_:
        #             df[feature] = self.encoder[feature].transform([value])[0]
        #         else:
        #             # Use the first class if value is not in known classes
        #             df[feature] = self.encoder[feature].transform([self.encoder[feature].classes_[0]])[0]
        #
        #     # NEW CODE: Handle health conditions
        #     # Extract conditions from OtherConditions field
        #     conditions = str(user_input.get('OtherConditions', '')).lower().split(',')
        #     conditions = [c.strip() for c in conditions]
        #
        #     # Add binary columns for each condition the model expects
        #     health_conditions = ['High cholesterol', 'Hypertension']
        #     for condition in health_conditions:
        #         df[condition] = 1 if condition.lower() in conditions else 0
        #
        #     # Make sure all required features exist
        #     for feature_name in self.model.feature_names_:
        #         if feature_name not in df.columns:
        #             # Add missing features with default values
        #             if feature_name in USER_NUM_FEATURES:
        #                 df[feature_name] = self.default_values.get(feature_name, 0.0)
        #             else:
        #                 df[feature_name] = 0  # Default for binary/categorical features

            # Handle categorical features
        for feature in USER_CAT_FEATURES:
                if feature == 'BMI' and feature in df.columns and not pd.isna(df.at[0, feature]):
                    continue

                if feature not in processed_input:
                    # Use default value
                    if feature in self.default_values:
                        value = self.default_values[feature]
                    elif feature in self.encoder and len(self.encoder[feature].classes_) > 0:
                        value = self.encoder[feature].classes_[0]
                    else:
                        value = "Unknown"
                else:
                    value = processed_input[feature]

                # Handle empty/None values
                if isinstance(value, (list, pd.Series, np.ndarray)):
                    # Handle array-like values
                    if isinstance(value, pd.Series) and value.empty:
                        value = self.default_values.get(feature, "Unknown")
                    elif len(value) == 0:
                        value = self.default_values.get(feature, "Unknown")
                    elif isinstance(value, pd.Series):
                        value = value.iloc[0]  # Take first value of series
                    elif isinstance(value, np.ndarray):
                        value = value[0]  # Take first value of array
                    elif isinstance(value, list) and len(value) > 0:
                        value = value[0]  # Take first value of list
                else:
                    # Handle scalar values
                    if pd.isna(value) or value == '':
                        value = self.default_values.get(feature, "Unknown")

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

        # Extract conditions from OtherConditions field
        conditions = []
        if 'OtherConditions' in user_input:
            if isinstance(user_input['OtherConditions'], str):
                conditions = [c.strip().lower() for c in user_input['OtherConditions'].split(',') if c.strip()]
            elif isinstance(user_input['OtherConditions'], list):
                conditions = [c.strip().lower() for c in user_input['OtherConditions'] if c.strip()]

        # Add binary columns for each condition the model expects
        expected_health_conditions = ['High cholesterol', 'Hypertension']
        for condition in expected_health_conditions:
            df[condition] = 1 if condition.lower() in conditions else 0

        # Make sure all required features exist
        for feature_name in self.model.feature_names_:
            if feature_name not in df.columns:
                # Add missing features with default values
                if feature_name in USER_NUM_FEATURES:
                    df[feature_name] = self.default_values.get(feature_name, 0.0)
                else:
                    df[feature_name] = 0  # Default for binary/categorical features

        # Final check - ensure all required features are present
        missing_features = [f for f in self.model.feature_names_ if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0  # Add default value


        # return df[USER_NUM_FEATURES + USER_CAT_FEATURES]
        # return df[expected_order]
        return df[self.model.feature_names_]

    def filter_allergies(self, meal: Dict, user_allergies: List[str]) -> bool:
        """Check if meal contains user's allergens"""
        if not user_allergies or not meal or meal.get('AllergyStatus', 'None') == 'None':
            return True

        meal_allergens = [a.strip().lower() for a in str(meal.get('AllergyStatus', '')).split(',') if a.strip()]
        user_allergies = [a.strip().lower() for a in user_allergies if a.strip()]

        return not any(allergen in meal_allergens for allergen in user_allergies)

    def filter_trigger_foods(self, meal: Dict, trigger_foods: List[str]) -> bool:
        """Check if meal contains user's trigger foods"""
        if not trigger_foods or not meal:
            return True

        # Check meal description, name, ingredients for trigger foods
        meal_text = ' '.join([
            str(meal.get('MealName', '')),
            str(meal.get('Description', '')),
            str(meal.get('Ingredients', ''))
        ]).lower()

        trigger_foods = [t.strip().lower() for t in trigger_foods if t.strip()]

        return not any(trigger in meal_text for trigger in trigger_foods)

    def check_dietary_restrictions(self, meal: Dict, diet: str) -> bool:
        """Check if meal meets dietary restrictions"""
        if not diet or not meal:
            return True

        diet = diet.strip().lower()
        meal_preference = str(meal.get('Preferences', '')).lower()

        # Check for vegetarian restrictions
        if diet == 'vegetarian':
            if 'vegetarian' in meal_preference:
                return True

            # Check for meat keywords in meal
            meat_keywords = ['beef', 'chicken', 'pork', 'lamb', 'meat', 'fish', 'seafood']
            meal_text = ' '.join([
                str(meal.get('MealName', '')),
                str(meal.get('Description', '')),
                str(meal.get('Ingredients', ''))
            ]).lower()

            return not any(meat in meal_text for meat in meat_keywords)

        # Direct match with preferences
        return diet in meal_preference

    def adjust_for_special_conditions(self, meals, conditions):
        """Adjust meal choices based on special health conditions"""
        if not conditions:
            return meals

        # Handle both string and list inputs for conditions
        if isinstance(conditions, str):
            conditions = [c.strip().lower() for c in conditions.split(',') if c.strip()]
        elif isinstance(conditions, list):
            conditions = [c.strip().lower() for c in conditions if c and isinstance(c, str)]
        else:
            # If it's neither string nor list, return meals unchanged
            return meals

        if 'high cholesterol' in conditions:
            # Filter out high fat meals
            filtered_meals = []
            for meal in meals:
                fat = float(meal.get('Fat(g)', 0))
                calories = float(meal.get('Calories', 1000))  # Prevent division by zero
                fat_percentage = (fat * 9 / calories) * 100  # Fat has 9 calories per gram

                if fat_percentage < 30:  # Keep meals with less than 30% calories from fat
                    filtered_meals.append(meal)

            if filtered_meals:  # Only return filtered meals if we have some left
                return filtered_meals

        return meals  # Return original meals if no filtering occurred or if all meals were filtered out

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
                suitable_plans = list(self.meal_plans.keys())[:5]  # Fallback - increase to 5 for more options



            user_diet = user_input.get('DietFollowed', '')

            # Filter by preferences/allergies/conditions
            user_allergies = user_input.get('Allergies', [])
            if isinstance(user_allergies, str):  # Backward compatibility
                user_allergies = user_allergies.split(',')
            user_allergies = [a.strip() for a in user_allergies if a.strip()]

            # Handle OtherConditions (supports both list and string)
            user_conditions = user_input.get('OtherConditions', [])
            if isinstance(user_conditions, str):  # Backward compatibility
                user_conditions = user_conditions.split(',')
            user_conditions = [c.strip() for c in user_conditions if c.strip()]

            # Handle TriggerFoods (supports both list and string)
            trigger_foods = user_input.get('TriggerFoods', [])
            if isinstance(trigger_foods, str):  # Backward compatibility
                trigger_foods = trigger_foods.split(',')
            trigger_foods = [t.strip() for t in trigger_foods if t.strip()]

            valid_plans = []
            for plan_id in suitable_plans:
                if plan_id not in self.meal_plans:
                    continue

                meals = self.meal_plans[plan_id]
                # Check if ALL meals in the plan meet both allergy and diet requirements
                all_meals_valid = True

                # Apply condition-specific filters first (e.g., low fat for high cholesterol)
                adjusted_meals = self.adjust_for_special_conditions(meals, user_conditions)

                for m in adjusted_meals:
                    allergy_ok = self.filter_allergies(m, user_allergies)
                    diet_ok = self.check_dietary_restrictions(m, user_diet)
                    trigger_ok = self.filter_trigger_foods(m, trigger_foods)

                    if not (allergy_ok and diet_ok and trigger_ok):
                        all_meals_valid = False
                        break

                if all_meals_valid:
                    valid_plans.append(plan_id)

            # Special handling for elderly patients
            is_elderly = user_input.get('Age', 0) > 65

            # Select best match
            chosen_plan = None
            if valid_plans:
                # For elderly, prioritize easier-to-digest, nutrient-dense meals
                if is_elderly and len(valid_plans) > 1:
                    # Get a plan with moderate calories and higher protein
                    plan_scores = []
                    for plan_id in valid_plans:
                        plan_meals = self.meal_plans[plan_id]
                        total_calories = sum(float(m.get('Calories', 0)) for m in plan_meals)
                        total_protein = sum(float(m.get('Protein(g)', 0)) for m in plan_meals)

                        # Score based on moderate calories (1500-1800) and higher protein
                        calorie_score = 1 - abs((total_calories - 1650) / 1000)  # Closer to 1650 is better
                        protein_score = total_protein / 100  # Higher protein is better

                        plan_scores.append((plan_id, calorie_score + protein_score))

                    # Get plan with highest score
                    plan_scores.sort(key=lambda x: x[1], reverse=True)
                    chosen_plan = plan_scores[0][0]
                else:
                    chosen_plan = random.choice(valid_plans)
            elif suitable_plans:
                chosen_plan = random.choice(suitable_plans)
            elif self.meal_plans:
                chosen_plan = list(self.meal_plans.keys())[0]  # Last resort
            else:
                return {"error": "No meal plans available"}

            plan_meals = self.sanitize_meal_data(self.meal_plans[chosen_plan])

            print("Available meal data fields:")
            if plan_meals and len(plan_meals) > 0:
                print(f"Sample meal keys: {list(plan_meals[0].keys())}")
                print(f"Sample meal data: {plan_meals[0]}")

            # Calculate nutrition safely
            nutrition = {
                'TotalCalories': sum(
                    float(m.get('CalorieCount',
                                m.get('Calories', nutrition_data.get(m.get('MealID', 0), {}).get('Calories', 0)))) for m
                    in
                    plan_meals),
                'NetCarbs(g)': sum(
                    float(m.get('Carbs(g)', nutrition_data.get(m.get('MealID', 0), {}).get('Carbohydrates', 0)) -
                          m.get('Fiber(g)', nutrition_data.get(m.get('MealID', 0), {}).get('Fiber', 0))) for m in
                    plan_meals),
                'Protein(g)': sum(
                    float(m.get('Protein(g)', nutrition_data.get(m.get('MealID', 0), {}).get('Protein', 0))) for m in
                    plan_meals),
                'Fat(g)': sum(
                    float(m.get('Fat(g)', nutrition_data.get(m.get('MealID', 0), {}).get('Fats', 0))) for m in
                    plan_meals)
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

            # Add special recommendations for elderly
            special_recommendations = []
            if is_elderly:
                special_recommendations = [
                    "Consider smaller, more frequent meals if appetite is reduced",
                    "Ensure adequate hydration throughout the day",
                    "Choose softer foods if chewing is difficult",
                    "Consider supplementing with vitamin D and calcium for bone health"
                ]

                # Add cholesterol-specific recommendations
                if 'high cholesterol' in str(user_conditions).lower():
                    special_recommendations.extend([
                        "Focus on heart-healthy fats like olive oil and avocados",
                        "Include soluble fiber from oats and barley to help lower cholesterol",
                        "Limit saturated fats from full-fat dairy and fatty meats"
                    ])

            return {
                'MealPlanID': chosen_plan,
                'PlanName': plan_name,
                'Nutrition': nutrition,
                'Meals': plan_meals,
                'UserProfileMatch': {
                    'GlycemicLoad': user_input.get('GlycemicPreference', 'Medium'),
                    'AllergiesAvoided': user_allergies,
                    'DietaryPreference': user_diet,
                    'SpecialConditions': user_conditions,
                    'TriggerFoodsAvoided': trigger_foods
                },
                'SpecialRecommendations': special_recommendations if special_recommendations else None
            }

        except Exception as e:
            import traceback
            print(f"Error in recommend: {e}")
            print(traceback.format_exc())
            return {"error": str(e), "details": traceback.format_exc()}


recommender = MealRecommender()


def recommend(data):
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
              example: 95
            Gender:
              type: string
              example: Female
            Height:
              type: number
              example: 160
            Weight:
              type: number
              example: 80
            FastingGlucose:
              type: number
              example: 130
            Allergies:
              type: string
              example: Dairy
            DietFollowed:
              type: string
              example: Vegetarian
            OtherConditions:
              type: string
              example: High cholesterol
            TriggerFoods:
              type: string
              example: Sugary snacks
    responses:
      200:
        description: Successful response
      400:
        description: Bad request
      500:
        description: Internal server error
    """
    try:
        # data = request.get_json()
        if not data:
            return jsonify({'error': 'No input provided'}), 400

        # Print request data for debugging
        print(f"Received request data: {data}")

        # Validate essential input
        required_fields = ['Age', 'Gender', 'Height', 'Weight']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Add default values for optional fields
        data.setdefault('Allergies', '')
        data.setdefault('DietFollowed', '')
        data.setdefault('OtherConditions', '')
        data.setdefault('TriggerFoods', '')

        result = recommender.recommend(data)
        print("Service :")
        print(result)
        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {e}\nDetails: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

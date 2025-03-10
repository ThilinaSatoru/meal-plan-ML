import pickle
import warnings
import os
from typing import Dict, Any
import numpy as np
import json
import random

import pandas as pd
from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from swagger.params import example_params

warnings.filterwarnings('ignore')

# Constants
ENCODER_PATH = 'encoder_meal.pkl'
MODEL_PATH = 'model_meal.pkl'
MEAL_DATASET_PATH = 'sri_lankan_meal_dataset.csv'
PROBABILITY_THRESHOLD = 0.1  # Minimum probability threshold for considering a meal

CAT_COLUMNS = [
    'Gender', 'Location', 'Occupation', 'DiabetesType',
    'FavoriteFoods', 'HealthGoals', 'DietChallenges',
    'TraditionalFoods', 'CookingFrequency', 'CookingMethods'
]
REQUIRED_FIELDS = [
    'Age', 'Gender', 'Height', 'Weight', 'Location', 'Occupation',
    'DiabetesType', 'DiagnosedYearsAgo', 'FastingGlucose',
    'PostprandialGlucose', 'FavoriteFoods', 'HealthGoals',
    'DietChallenges', 'TraditionalFoods', 'CookingFrequency',
    'CookingMethods'
]


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types with NaN handling"""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            if np.isnan(obj):
                return None  # Convert NaN to None for JSON compatibility
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)
CORS(app)
swagger = Swagger(app)
app.json_encoder = NumpyJSONEncoder


def load_resources() -> tuple:
    """Load and validate all required resources"""
    try:
        with open(ENCODER_PATH, 'rb') as f:
            encoder_meal = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model_meal = pickle.load(f)

        if not os.path.exists(MEAL_DATASET_PATH):
            raise FileNotFoundError(f"Meal dataset not found at {MEAL_DATASET_PATH}")

        meal_df = pd.read_csv(MEAL_DATASET_PATH)
        meal_lookup = meal_df.groupby('MealID').apply(
            lambda x: x.drop('MealID', axis=1).to_dict('records')
        ).to_dict()

        return encoder_meal, model_meal, meal_lookup

    except Exception as e:
        raise SystemExit(f"Failed to load resources: {str(e)}")


# Load resources on startup
encoder_meal, model_meal, meal_lookup = load_resources()


def validate_input(input_data: Dict[str, Any]) -> None:
    """Validate input data structure and content"""
    missing_fields = [field for field in REQUIRED_FIELDS if field not in input_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    if not isinstance(input_data.get('Age'), (int, float)) or input_data['Age'] < 0:
        raise ValueError("Age must be a positive number")

    if not isinstance(input_data.get('Height'), (int, float)) or input_data['Height'] <= 0:
        raise ValueError("Height must be a positive number")

    if not isinstance(input_data.get('Weight'), (int, float)) or input_data['Weight'] <= 0:
        raise ValueError("Weight must be a positive number")

    if input_data.get('Gender') not in ['Male', 'Female']:
        raise ValueError("Gender must be either 'Male' or 'Female'")


def preprocess_input(sample_json: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for model prediction"""
    columns_to_drop = [
        'RecordID', 'Name', 'OtherConditions', 'FoodsAvoided',
        'Intolerances', 'TriggerFoods', 'DietFollowed', 'Allergies'
    ]

    df = pd.DataFrame([sample_json])
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Encode categorical features
    for col in CAT_COLUMNS:
        if col not in encoder_meal:
            raise ValueError(f"Missing encoder for column: {col}")
        df[col] = encoder_meal[col].transform(df[col])

    return df


def get_random_recommendation(df: pd.DataFrame) -> int:
    """Get a random meal recommendation from suitable options"""
    if hasattr(model_meal, 'predict_proba'):
        # Get prediction probabilities
        proba = model_meal.predict_proba(df.values)[0]
        # Get indices of meals above threshold
        suitable_indices = np.where(proba >= PROBABILITY_THRESHOLD)[0]
        if len(suitable_indices) == 0:
            # If no meals above threshold, take top 3 meals
            suitable_indices = np.argsort(proba)[-3:]

        # Choose random index from suitable ones
        chosen_idx = random.choice(suitable_indices)
        return int(encoder_meal['MealID'].inverse_transform([chosen_idx])[0])
    else:
        # If predict_proba not available, use basic prediction
        prediction = model_meal.predict(df.values)
        return int(encoder_meal['MealID'].inverse_transform([prediction])[0])


def process_meal_info(meal_info: Any) -> Any:
    """Convert NumPy types to Python native types in meal info and handle NaN values"""
    if isinstance(meal_info, list):
        return [{k: None if isinstance(v, (float, np.float_)) and np.isnan(v) else
                    v.item() if isinstance(v, np.generic) else v
                 for k, v in meal.items()}
                for meal in meal_info]
    else:
        return {k: None if isinstance(v, (float, np.float_)) and np.isnan(v) else
                  v.item() if isinstance(v, np.generic) else v
               for k, v in meal_info.items()}


def inference_meal(sample_json: Dict[str, Any]) -> Dict[str, Any]:
    """Process input and generate a random meal recommendation"""
    df = preprocess_input(sample_json)

    # Get random recommendation
    meal_id = get_random_recommendation(df)

    # Lookup meal details
    meal_info = meal_lookup.get(meal_id)
    if not meal_info:
        raise ValueError(f"MealID {meal_id} not found in dataset")

    # Process meal info to convert NumPy types
    processed_info = process_meal_info(meal_info)

    return {
        'MealID': meal_id,
        'Meals': processed_info if isinstance(processed_info, list) else [processed_info]
    }


@app.route('/predict_meal', methods=['POST'])
@swag_from({
    'summary': 'Predict a random meal recommendation',
    'description': 'Recommends a random suitable meal based on user health profile and preferences',
    'parameters': [example_params],
    'responses': {
        200: {
            'description': 'Successful recommendation',
            'schema': {
                'type': 'object',
                'properties': {
                    'MealID': {'type': 'integer'},
                    'Meals': {
                        'type': 'array',
                        'items': {'type': 'object'}
                    }
                }
            }
        },
        400: {'description': 'Invalid input'},
        500: {'description': 'Server error'}
    }
})
def predict_meal():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        validate_input(input_data)
        result = inference_meal(input_data)

        # Ensure all values are JSON serializable
        response_json = json.dumps(result, cls=NumpyJSONEncoder)

        return Response(response=response_json, status=200, mimetype='application/json')

    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")

        # If JSON parsing fails, return an empty response
        try:
            json.dumps(result, cls=NumpyJSONEncoder)  # Attempt serialization
        except Exception:
            return Response(response="{}", status=200, mimetype='application/json')

        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )
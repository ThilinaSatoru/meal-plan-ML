from flask import request
from flask_restx import Namespace, Resource, fields

from app.services.meal_service import get_all_meals, add_meals_from_csv

meal_bp = Namespace("meal", description="Operations related to plants")

# Define the Meal model with a default example
meal_model = meal_bp.model('Meal', {
    'file': fields.String(required=True, description='CSV file Path.', example='meal.csv'),
})


@meal_bp.route("/")
class MealList(Resource):

    def get(self):
        meals = get_all_meals()
        return meals, 200

    @meal_bp.expect(meal_model)
    def post(self):
        data = request.json
        meal = add_meals_from_csv(data["file"])
        return meal, 201

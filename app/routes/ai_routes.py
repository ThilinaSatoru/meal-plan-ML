from flask_restx import Namespace, Resource, fields, ValidationError

from app.services.AI.ai_service import recommend
from app.services.dto import ProfileAiDTO

ai_bp = Namespace("ai", description="Operations related to plants")

profile_ai_model = ai_bp.model('ProfileAiDTO', {
    'Age': fields.Integer(required=True, description='User age', example=95, min=1, max=120),
    'Allergies': fields.String(required=False, description='Food allergies', example='Dairy', nullable=True),
    'DietFollowed': fields.String(required=False, description='Diet type', example='Vegetarian', nullable=True),
    'FastingGlucose': fields.Integer(required=True, description='Fasting glucose level', example=130),
    'Gender': fields.String(required=True,
                            description='Gender',
                            example='Female',
                            enum=['Male', 'Female', 'Other']),
    'Height': fields.Float(required=True, description='Height in cm', example=160, min=50, max=250),
    'OtherConditions': fields.String(required=False, description='Other health conditions', example='High cholesterol',
                                     nullable=True),
    'TriggerFoods': fields.String(required=False, description='Foods that trigger issues', example='Sugary snacks',
                                  nullable=True),
    'Weight': fields.Float(required=True, description='Weight in kg', example=80, min=20, max=300)
})

# Models
nutrition_summary_model = ai_bp.model('NutritionSummary', {
    'TotalCalories': fields.Float(required=True, example=1150.0),
    'NetCarbs(g)': fields.Float(required=True, example=82.0),
    'Protein(g)': fields.Float(required=True, example=78.0),
    'Fat(g)': fields.Float(required=True, example=30.0)
})

meal_model = ai_bp.model('Meal', {
    'MealID': fields.Integer(required=True, example=2),
    'MealName': fields.String(required=True, example='Mediterranean (1800 cal)'),
    'Type': fields.String(required=True, example='Breakfast'),
    'MealDetails': fields.String(required=True, example='Suwandel Rice (1 cup)...'),
    'Calories': fields.Integer(required=True, example=320),
    'Carbs(g)': fields.Integer(required=True, example=38),
    'Protein(g)': fields.Integer(required=True, example=15),
    'Fiber(g)': fields.Integer(required=True, example=6),
    'GlycemicLoad': fields.String(required=True, example='Medium'),
    'AllergyStatus': fields.String(required=True, example='Contains fish'),
    'Preferences': fields.String(required=True, example='Pescetarian')
})

user_profile_match_model = ai_bp.model('UserProfileMatch', {
    'GlycemicLoad': fields.String(required=True, example='Medium'),
    'AllergiesAvoided': fields.List(fields.String, example=['Dairy']),
    'DietaryPreference': fields.String(example='Vegetarian'),
    'SpecialConditions': fields.String(example='High cholesterol'),
    'TriggerFoodsAvoided': fields.List(fields.String, example=['Sugary snacks'])
})

meal_plan_response = ai_bp.model('MealPlanResponse', {
    'MealPlanID': fields.Integer(example=2),
    'PlanName': fields.String(example='Mediterranean (1800 cal)'),
    'Meals': fields.List(fields.Nested(meal_model)),
    'Nutrition': fields.Nested(nutrition_summary_model),
    'SpecialRecommendations': fields.List(fields.String),
    'UserProfileMatch': fields.Nested(user_profile_match_model)
})


# Controller
@ai_bp.route("/")
class MealRecommendation(Resource):
    @ai_bp.expect(profile_ai_model)
    @ai_bp.marshal_with(meal_plan_response)
    def post(self):
        try:
            data = ProfileAiDTO().load(ai_bp.payload)
            rq_data = recommend(data)  # Direct dict from service
            print(rq_data)  # Verify structure matches models
            return rq_data, 200
        except ValidationError as err:
            return {'errors': err.messages}, 400

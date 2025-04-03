from .ai_routes import ai_bp
from .meal_routes import meal_bp
from .nutrition_routes import nutrition_bp
from .profile_routes import profile_bp


def register_routes(app, api):
    # Add the namespaces directly to the API instance
    api.add_namespace(profile_bp)
    api.add_namespace(meal_bp)
    api.add_namespace(nutrition_bp)
    api.add_namespace(ai_bp)

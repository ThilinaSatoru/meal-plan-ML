from dotenv import load_dotenv
from flask import Flask
from flask_restx import Api

from app.database import db
from app.routes import register_routes, profile_bp, meal_bp, nutrition_bp

load_dotenv()


def create_app():
    app = Flask(__name__)

    # Load configurations
    app.config.from_object("app.config.Config")

    # Initialize database
    db.init_app(app)

    # Initialize Swagger (Flask-RESTX API)
    api = Api(app, doc="/docs")  # Swagger UI will be available at /docs

    # Register routes (passing api to register_routes)
    register_routes(app, api)

    return app

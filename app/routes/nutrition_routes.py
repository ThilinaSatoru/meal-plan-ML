from flask import request
from flask_restx import Namespace, Resource

from app.services.nutrition_service import get_all, insert_from_csv

nutrition_bp = Namespace("nutrition", __name__)


@nutrition_bp.route("/")
class ProfileList(Resource):

    def get(self):
        nutrition = get_all()
        return nutrition, 200

    def post(self):
        data = request.json
        nutrition = insert_from_csv(data["file"])
        return nutrition, 201

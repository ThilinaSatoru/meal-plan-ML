from flask import request
from flask_restx import Namespace, Resource

from app.services.profile_service import get_all, insert_from_csv

profile_bp = Namespace("profile", __name__)


@profile_bp.route("/")
class ProfileList(Resource):

    def get(self):
        profile = get_all()
        return profile, 200

    def post(self):
        data = request.json
        profile = insert_from_csv(data["file"])
        return profile, 201

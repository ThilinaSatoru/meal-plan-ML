from marshmallow import Schema, fields, validate


class ProfileAiDTO(Schema):
    Age = fields.Integer(required=True, validate=validate.Range(min=1, max=120))
    Allergies = fields.List(fields.String(),required=False, allow_none=True)
    DietFollowed = fields.String(required=False, allow_none=True)
    FastingGlucose = fields.Integer(required=True)
    Gender = fields.String(
        required=True,
        validate=validate.OneOf(["Male", "Female", "Other"])
    )
    Height = fields.Float(required=True, validate=validate.Range(min=50, max=250))
    OtherConditions = fields.List(fields.String(),required=False, allow_none=True)
    TriggerFoods = fields.List(fields.String(),required=False, allow_none=True)
    Weight = fields.Float(required=True, validate=validate.Range(min=20, max=300))

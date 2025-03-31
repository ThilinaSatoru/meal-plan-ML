example_params = {
    'name': 'body',
    'in': 'body',
    'required': True,
    'schema': {
        'type': 'object',
        'properties': {
            'RecordID': {'type': 'integer'},
            'Age': {'type': 'integer'},
            'Gender': {'type': 'string', 'enum': ['Male', 'Female']},
            'Height': {'type': 'integer'},
            'Weight': {'type': 'integer'},
            'Location': {'type': 'string'},
            'FBS': {'type': 'string'},
            'BMI': {'type': 'number'},
            'HbA1c': {'type': 'string'},
            'DiagnosedYearsAgo': {'type': 'integer'},
            'FastingGlucose': {'type': 'integer'},
            'PostprandialGlucose': {'type': 'integer'},
            'OtherConditions': {'type': 'string', 'nullable': True},
            'FavoriteFoods': {'type': 'string'},
            'FoodsAvoided': {'type': 'string', 'nullable': True},
            'DietFollowed': {'type': 'string', 'nullable': True},
            'TriggerFoods': {'type': 'string', 'nullable': True},
            'Allergies': {'type': 'string', 'nullable': True},
            'TraditionalFoods': {'type': 'string'},
            'CookingFrequency': {'type': 'string'},
            'CookingMethods': {'type': 'string'},
            'MealID': {'type': 'integer'}
        },
        'example': [
            {
                "RecordID": 1,
                "Age": 64,
                "Gender": "Male",
                "Height": 168,
                "Weight": 80,
                "Location": "Urban area",
                "FBS": "Diabetes",
                "BMI": 0.002834467,
                "HbA1c": "Diabetes",
                "DiagnosedYearsAgo": 2,
                "FastingGlucose": 138,
                "PostprandialGlucose": 235,
                "OtherConditions": "High cholesterol",
                "FavoriteFoods": "Rice and curry",
                "FoodsAvoided": None,
                "DietFollowed": "Vegetarian",
                "TriggerFoods": None,
                "Allergies": "Dairy",
                "TraditionalFoods": "Pickled vegetables",
                "CookingFrequency": "Rarely",
                "CookingMethods": "Steaming",
                "MealID": 4
            }
        ]
    }
}

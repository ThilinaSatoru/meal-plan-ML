from app.database import db


class Meal(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    meal_id = db.Column(db.Integer, nullable=False)
    meal_name = db.Column(db.String(255), nullable=False)
    meal_type = db.Column(db.String(50), nullable=False)
    meal_details = db.Column(db.Text, nullable=False)
    calories = db.Column(db.Integer, nullable=False)
    carbs = db.Column(db.Float, nullable=False)
    protein = db.Column(db.Float, nullable=False)
    fiber = db.Column(db.Float, nullable=False)
    glycemic_load = db.Column(db.String(50), nullable=False)
    allergy_status = db.Column(db.String(255), nullable=True)
    preferences = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'meal_id': self.meal_id,
            'meal_name': self.meal_name,
            'meal_type': self.meal_type,
            'meal_details': self.meal_details,
            'calories': self.calories,
            'carbs': self.carbs,
            'protein': self.protein,
            'fiber': self.fiber,
            'glycemic_load': self.glycemic_load,
            'allergy_status': self.allergy_status,
            'preferences': self.preferences,
            # Include related nutrition if needed
            'nutrition': [nutrition.to_dict() for nutrition in self.nutrition] if self.nutrition else []
        }

    def __repr__(self):
        return f"<Meal {self.meal_name}>"


class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(255), nullable=False)
    fbs = db.Column(db.String(50), nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    hba1c = db.Column(db.String(50), nullable=False)
    diagnosed_years_ago = db.Column(db.Integer, nullable=False)
    fasting_glucose = db.Column(db.Integer, nullable=False)
    postprandial_glucose = db.Column(db.Integer, nullable=False)
    other_conditions = db.Column(db.Text, nullable=True)
    favorite_foods = db.Column(db.Text, nullable=True)
    foods_avoided = db.Column(db.Text, nullable=True)
    diet_followed = db.Column(db.String(255), nullable=True)
    trigger_foods = db.Column(db.Text, nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    traditional_foods = db.Column(db.Text, nullable=True)
    cooking_frequency = db.Column(db.String(50), nullable=True)
    cooking_methods = db.Column(db.String(255), nullable=True)
    meal_id = db.Column(db.Integer, db.ForeignKey('meal.id'), nullable=False)

    meal = db.relationship('Meal', backref=db.backref('profiles', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'age': self.age,
            'gender': self.gender,
            'height': self.height,
            'weight': self.weight,
            'location': self.location,
            'fbs': self.fbs,
            'bmi': self.bmi,
            'hba1c': self.hba1c,
            'diagnosed_years_ago': self.diagnosed_years_ago,
            'fasting_glucose': self.fasting_glucose,
            'postprandial_glucose': self.postprandial_glucose,
            'other_conditions': self.other_conditions,
            'favorite_foods': self.favorite_foods,
            'foods_avoided': self.foods_avoided,
            'diet_followed': self.diet_followed,
            'trigger_foods': self.trigger_foods,
            'allergies': self.allergies,
            'traditional_foods': self.traditional_foods,
            'cooking_frequency': self.cooking_frequency,
            'cooking_methods': self.cooking_methods,
            'meal_id': self.meal_id,
            # Include related meal if needed
            'meal': self.meal.to_dict() if self.meal else None
        }

    def __repr__(self):
        return f"<Profile {self.id} - {self.gender} - {self.age} years>"


class Nutrition(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    meal_id = db.Column(db.Integer, db.ForeignKey('meal.id'), nullable=False)
    glycemic_index = db.Column(db.Integer, nullable=False)
    calories = db.Column(db.Integer, nullable=False)
    carbohydrates = db.Column(db.Float, nullable=False)
    protein = db.Column(db.Float, nullable=False)
    fats = db.Column(db.Float, nullable=False)
    fiber = db.Column(db.Float, nullable=False)
    sodium = db.Column(db.Integer, nullable=False)
    sugar = db.Column(db.Float, nullable=False)

    meal = db.relationship('Meal', backref=db.backref('nutrition', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'meal_id': self.meal_id,
            'glycemic_index': self.glycemic_index,
            'calories': self.calories,
            'carbohydrates': self.carbohydrates,
            'protein': self.protein,
            'fats': self.fats,
            'fiber': self.fiber,
            'sodium': self.sodium,
            'sugar': self.sugar
        }

    def __repr__(self):
        return f"<Nutrition for Meal {self.meal_id}>"

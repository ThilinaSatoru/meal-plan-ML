import pandas as pd

# Load CSV file
df = pd.read_csv("Survey_Dietary_Habit_Galle-cleaned.csv")  # Change 'data.csv' to your actual file path

# Select necessary columns
columns_needed = [
    "Age_Numeric", "Gender", "Height_Numeric", "Weight_Numeric", "Location",
    "Years_With_Diabetes_Numeric", "FBS_Numeric", "PPBS_Numeric", "HbA1c_Numeric",
    "Most_Frequent_Food_Type", "Meal_Preparation_Methods_Boiling_e.g._boiled_rice_boiled_vegeta",
    "Dietary_Approach", "Diet_Challenges_Limited_availability_of_diabetes_friendly_foods",
    "Dietary_Preference", "Familiarity_Traditional_Foods", "Interest_Traditional_Food_Recommendations_Numeric",
    "Has_Food_Allergies"
]
df = df[columns_needed]

# Rename columns to match required format
df.rename(columns={
    "Meal_Preparation_Methods_Boiling_e.g._boiled_rice_boiled_vegeta": "Meal_Preparation_Methods",
    "Diet_Challenges_Limited_availability_of_diabetes_friendly_foods": "Diet_Challenges"
}, inplace=True)

# Add UserID and MealID
df.insert(0, "UserID", range(1, len(df) + 1))
df["MealID"] = ["M" + str(i).zfill(3) for i in range(1, len(df) + 1)]

# Save to new CSV
df.to_csv("transformed_data.csv", index=False)

print("Data transformation completed and saved to transformed_data.csv")

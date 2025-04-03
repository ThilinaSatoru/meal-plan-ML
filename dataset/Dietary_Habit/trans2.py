import pandas as pd

# Define the input and output CSV file paths
input_csv = 'Survey_Dietary_Habit_Galle.csv'
output_csv = 'Survey_Dietary_Habit_Galle-zzz.csv'

# Read the input CSV file
df = pd.read_csv(input_csv)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Print the column names to verify
print("Columns in the CSV file:", df.columns)


# Define a function to transform the data
def transform_data(row):
    def safe_median(value):
        try:
            # Split the range and convert to integers
            range_values = list(map(int, value.split('-')))
            return sum(range_values) // 2
        except ValueError:
            # Handle non-numeric values
            if "More than" in value:
                return int(value.replace("More than ", "")) + 1  # Assuming "More than X" means X+1
            return None  # Return None or a default value for other non-numeric cases

    def extract_glucose_level(value):
        try:
            # Extract the numeric part from the string
            numeric_part = int(''.join(filter(str.isdigit, value)))
            return numeric_part
        except ValueError:
            return None

    try:
        # Extract and clean the age group
        age_group = row["What is your age group?"].replace(" years", "").replace("–", "-")
        age = safe_median(age_group)

        # Extract and clean the height
        height_group = row["What is your height?"].replace(" cm", "").replace("–", "-")
        height = safe_median(height_group)

        # Extract and clean the weight
        weight_group = row["What is your current weight?"].replace(" kg", "").replace("–", "-")
        weight = safe_median(weight_group)

        # Extract and clean the diagnosed years
        diagnosed_years = row["How long have you been diagnosed with diabetes?"].replace(" years", "").replace("–", "-")
        diagnosed_years_ago = safe_median(diagnosed_years)

        # Derive DiabetesType
        diabetes_type = "Type 2" if diagnosed_years_ago is not None else "Not Specified"

        # Derive FastingGlucose
        fasting_glucose = extract_glucose_level(
            row["What was your last recorded fasting blood sugar (FBS) level (before eating in the morning)?"])

        # Derive PostprandialGlucose
        postprandial_glucose = extract_glucose_level(
            row["What was your last recorded postprandial blood sugar (after meals, 2 hours later) level?"])

        return {
            "RecordID": row.name + 1,  # Assuming RecordID starts from 1
            "Name": "Not Provided",
            "Age": age,
            "Gender": row["What is your gender?"],
            "Height": height,
            "Weight": weight,
            "Location": row["Where do you live?"],
            "Occupation": "Not Provided",
            "DiabetesType": diabetes_type,
            "DiagnosedYearsAgo": diagnosed_years_ago,
            "FastingGlucose": fasting_glucose,
            "PostprandialGlucose": postprandial_glucose,
            "OtherConditions": row["If Yes, what are they?"],
            "FavoriteFoods": row["Which type of food do you consume most frequently?"],
            "FoodsAvoided": row["Are there any foods you avoid for health reasons (besides allergies)?"],
            "DietFollowed": row["What is your dietary preference?"],
            "TriggerFoods": "Not Provided",
            "Allergies": row["Do you have any food allergies?"],
            "Intolerances": "Not Provided",
            "HealthGoals": "Not Provided",
            "DietChallenges": row["What are the biggest challenges you face in maintaining a diabetes-friendly diet?"],
            "TraditionalFoods": "; ".join([
                row["1.  Traditional Grains & Cereals"],
                row["2.  Traditional Rice Varieties"],
                row["3.  Herbal Porridges (Kola Kenda)"],
                row["4.  Traditional Vegetable & Leafy Green Dishes"],
                row["5.  Traditional Sri Lankan Curries & Dishes"],
                row["6.  Herbal Drinks"],
                row["7.Traditional  Snacks"]
            ]),
            "CookingFrequency": "Rarely",  # Inferred from meal preparation information
            "CookingMethods": row["What are the methods usually used to prepare your daily meals?"],
            "MealID": row.name + 1  # Assuming MealID starts from 1
        }
    except KeyError as e:
        print(f"Missing column: {e}")
        return None


# Apply the transformation to each row
transformed_data = df.apply(transform_data, axis=1)

# Filter out any rows that resulted in None due to missing columns
transformed_data = transformed_data.dropna()

# Convert the transformed data to a DataFrame
transformed_df = pd.DataFrame(transformed_data.tolist())

# Write the transformed data to the output CSV file
transformed_df.to_csv(output_csv, index=False)

print(f"Transformed data has been written to {output_csv}")

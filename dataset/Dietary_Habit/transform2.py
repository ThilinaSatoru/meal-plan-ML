import pandas as pd
import numpy as np
import re

# Function to expand the dataset
def expand_dataset(input_file, target_count=346):
    # Load the original CSV file
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"Original dataset has {len(df)} responses")
    
    # Number of additional responses needed
    additional_responses = target_count - len(df)
    
    if additional_responses <= 0:
        print("No additional responses needed")
        return df
    
    print(f"Generating {additional_responses} additional responses")
    
    # Clean column names (trim spaces to avoid errors)
    df.columns = df.columns.str.strip()
    
    # Create duplicates of existing data
    df_extended = df.sample(n=additional_responses, replace=True).reset_index(drop=True)
    
    # Modify timestamps (increment by random hours/days)
    if "Timestamp" in df_extended.columns:
        df_extended["Timestamp"] = pd.to_datetime(df_extended["Timestamp"], errors='coerce') + pd.to_timedelta(
            np.random.randint(1, 30, size=additional_responses), unit='h')
    
    # Find all columns except fixed ID-like values
    columns_to_modify = [col for col in df_extended.columns if col != "Timestamp"]
    
    # Modify categorical and numerical responses by shuffling within the column
    for col in columns_to_modify:
        if col in df_extended.columns:
            df_extended[col] = df_extended[col].sample(frac=1).reset_index(drop=True)
    
    # Merge original and new data
    df_final = pd.concat([df, df_extended], ignore_index=True)
    
    print(f"Expanded dataset now has {len(df_final)} responses")
    return df_final

# Step 1: Load and expand the original survey CSV file
input_csv_file = r'Survey_Dietary_Habit_Galle.csv'
expanded_df = expand_dataset(input_csv_file)

# Save the expanded dataset if needed
expanded_file = "Survey_Dietary_Habit_Galle-Expanded.csv"
expanded_df.to_csv(expanded_file, index=False)
print(f"✅ Expanded dataset saved as '{expanded_file}'")

# Continue with the rest of the original code using the expanded dataset instead of the original
df = expanded_df
print(f"Total Rows and Columns after reading: {df.shape}")

# Define the correct column mapping for the new CSV structure
column_mapping = {
    "Timestamp": "Timestamp",
    "What is your age group? ": "Age_Group",
    "What is your gender?": "Gender",
    "What is your current weight?  ": "Weight_kg",
    "What is your height?  ": "Height_cm",
    "Where do you live?  ": "Location",
    "How long have you been diagnosed with diabetes? ": "Years_With_Diabetes",
    "How long have you been taking medicine for diabetics? ": "Years_On_Medication",
    "Do you have any other chronic diseases?": "Has_Other_Chronic_Diseases",
    "If Yes, what are they? ": "Other_Chronic_Diseases",
    "What was your last recorded fasting blood sugar (FBS) level (before eating in the morning)? ": "Last_FBS_Level",
    "What was your last recorded postprandial blood sugar (after meals, 2 hours later) level? ": "Last_PPBS_Level",
    "What was your last recorded HbA1c level (%)? ": "Last_HbA1c_Level",
    "How frequently do you check your blood sugar levels?  ": "Blood_Sugar_Check_Frequency",
    "Which methods do you use to monitor your blood sugar levels? ": "Blood_Sugar_Monitoring_Methods",
    "How many meals do you eat per day? ": "Meals_Per_Day",
    "How often do you consume a balanced, diabetes-friendly meal? ": "Diabetes_Friendly_Meal_Frequency",
    "Which type of food do you consume most frequently? ": "Most_Frequent_Food_Type",
    "What are the methods usually used to prepare your daily meals? ": "Meal_Preparation_Methods",
    "Do you receive  personalized dietary guidance from a healthcare professional (doctor/nutritionist)? ": "Receives_Professional_Dietary_Guidance",
    "Who prepares most of your meals?  ": "Meal_Preparer",
    "Which of the following best describes your dietary approach for diabetes management?": "Dietary_Approach",
    "What are the biggest challenges you face in maintaining a diabetes-friendly diet? ": "Diet_Challenges",
    "What is your dietary preference? ": "Dietary_Preference",
    "How often do you consume sugary foods or drinks (e.g., sweets, biscuits, soft drinks)?  ": "Sugary_Food_Frequency",
    "Do you have any food allergies?  ": "Has_Food_Allergies",
    "Are there any foods you avoid for health reasons (besides allergies)?": "Foods_Avoided_Health_Reasons",
    "How familiar are you with the health benefits of traditional Sri Lankan foods (e.g., grains, herbal porridges, and traditional rice) in managing diabetes?": "Familiarity_Traditional_Foods",
    "Do you believe that consuming traditional Sri Lankan foods helps in managing diabetes? ": "Belief_Traditional_Foods_Help",
    "What prevents you from consuming more traditional Sri Lankan foods?": "Barriers_To_Traditional_Foods",
    "Would you be willing to increase your intake of traditional Sri Lankan foods if they are proven to help manage diabetes?  ": "Willingness_Increase_Traditional_Foods",
    "1.  Traditional Grains & Cereals ": "Consumption_Traditional_Grains",
    "2.  Traditional Rice Varieties ": "Consumption_Traditional_Rice",
    "3.  Herbal Porridges (Kola Kenda) ": "Consumption_Herbal_Porridges",
    "4.  Traditional Vegetable & Leafy Green Dishes ": "Consumption_Traditional_Vegetables",
    "5.  Traditional Sri Lankan Curries & Dishes ": "Consumption_Traditional_Curries",
    "6.  Herbal Drinks ": "Consumption_Herbal_Drinks",
    "7.Traditional  Snacks   ": "Consumption_Traditional_Snacks",
    "Would you be interested in receiving meal recommendations based on traditional Sri Lankan foods? ": "Interest_Traditional_Food_Recommendations",
    "Would you prefer to receive dietary recommendations through a website? ": "Prefer_Website_Recommendations",
    "How important is it that meal recommendations consider your specific health conditions (e.g., kidney disease, cholesterol, heart disease)? ": "Importance_Health_Condition_Consideration",
    "Which factors should be considered when designing a personalized meal plan for you? ": "Personalization_Factors",
    " What are your preferred time for receiving meal recommendations?": "Preferred_Recommendation_Time",
    "Would you trust a system that provides dietary advice validated by healthcare professionals? ": "Trust_In_Validated_System",
    "Have you ever used a digital tool, web site or mobile app to track your diet? ": "Used_Digital_Diet_Tool",
    "How comfortable are you with using a website for meal planning?  ": "Website_Comfort_Level",
    "Would you be willing to try a Sri Lankan traditional food-based meal planning website? ": "Willingness_Try_Traditional_Website",
    "How user-friendly should a food recommendation system be for elderly diabetics? ": "Required_User_Friendliness",
    "How satisfied would you be with a personalized meal plan based on traditional Sri Lankan foods? ": "Expected_Satisfaction_Level",
    "Which of the following would make you more likely to use a traditional food recommendation system? ": "Factors_Increasing_Usage",
    "If you used a traditional food recommendation website, how likely would you be to follow the meal plans? ": "Likelihood_Follow_Plans"
}

# Handle column mismatch by checking which columns exist in the dataframe
valid_columns = {}
for old_col, new_col in column_mapping.items():
    if old_col in df.columns:
        valid_columns[old_col] = new_col
    # Also try without trailing spaces
    elif old_col.strip() in df.columns:
        valid_columns[old_col.strip()] = new_col

# Debugging print statements
print("Total Rows and Columns before renaming:", df.shape)
print("First 5 columns before renaming (sample):", list(df.columns)[:5])

# Rename columns using valid columns
df.rename(columns=valid_columns, inplace=True)

# Debugging print statements after renaming
print("Total Rows and Columns after renaming:", df.shape)
print("First 5 columns after renaming (sample):", list(df.columns)[:5])

# Function to extract lower value from range strings like "50 – 59 kg", "60 – 64 years"
def extract_lower_value(range_str):
    if pd.isna(range_str):
        return None
    
    # Match patterns like "50 – 59 kg" or "1 – 3 years"
    match = re.search(r'(\d+)(?:\s*–\s*\d+)?', str(range_str))
    if match:
        return int(match.group(1))
    return None

# Function to extract value from specific blood sugar level ranges
def extract_blood_sugar_value(range_str):
    if pd.isna(range_str):
        return None
    
    # Handle "Less than X mg/dL" format
    less_than_match = re.search(r'Less than (\d+)', str(range_str))
    if less_than_match:
        return int(less_than_match.group(1)) - 1  # Subtract 1 to represent "less than"
    
    # Handle ranges like "140 - 199 mg/dL"
    range_match = re.search(r'(\d+)\s*-\s*\d+', str(range_str))
    if range_match:
        return int(range_match.group(1))
    
    # Handle "Greater than X mg/dL" format
    greater_than_match = re.search(r'Greater than (\d+)', str(range_str))
    if greater_than_match:
        return int(greater_than_match.group(1)) + 1  # Add 1 to represent "greater than"
    
    # Try to extract any number
    number_match = re.search(r'(\d+)', str(range_str))
    if number_match:
        return int(number_match.group(1))
    
    return None

# Function to extract HbA1c value
def extract_hba1c_value(range_str):
    if pd.isna(range_str):
        return None
    
    # Handle "Less than X%" format
    less_than_match = re.search(r'Less than (\d+\.\d+)', str(range_str))
    if less_than_match:
        return float(less_than_match.group(1)) - 0.1  # Subtract 0.1 to represent "less than"
    
    # Handle ranges like "5.7 - 6.4%"
    range_match = re.search(r'(\d+\.\d+)\s*-\s*\d+\.\d+', str(range_str))
    if range_match:
        return float(range_match.group(1))
    
    # Handle "Greater than X%" format
    greater_than_match = re.search(r'Greater than (\d+\.\d+)', str(range_str))
    if greater_than_match:
        return float(greater_than_match.group(1)) + 0.1  # Add 0.1 to represent "greater than"
    
    # Try to extract any decimal number
    number_match = re.search(r'(\d+\.\d+)', str(range_str))
    if number_match:
        return float(number_match.group(1))
    
    return None

# Function to convert Likert scale (numeric ratings) to integers
def convert_to_numeric(value):
    if pd.isna(value):
        return None
    
    # Try to convert to integer if it's a number
    try:
        return int(value)
    except (ValueError, TypeError):
        # If not a number, return original value
        return value

# Process range data for age, weight, height, and years of experience
if 'Timestamp' in df.columns:
    df['Timestamp'] = df['Timestamp'].astype(str).str.extract(r'(\d{4}/\d{2}/\d{2})')

# Process columns if they exist in the DataFrame
if 'Age_Group' in df.columns:
    df['Age_Numeric'] = df['Age_Group'].apply(extract_lower_value)
if 'Weight_kg' in df.columns:
    df['Weight_Numeric'] = df['Weight_kg'].apply(extract_lower_value)
if 'Height_cm' in df.columns:
    df['Height_Numeric'] = df['Height_cm'].apply(extract_lower_value)
if 'Years_With_Diabetes' in df.columns:
    df['Years_With_Diabetes_Numeric'] = df['Years_With_Diabetes'].apply(extract_lower_value)
if 'Years_On_Medication' in df.columns:
    df['Years_On_Medication_Numeric'] = df['Years_On_Medication'].apply(extract_lower_value)

# Process blood sugar and HbA1c values if they exist
if 'Last_FBS_Level' in df.columns:
    df['FBS_Numeric'] = df['Last_FBS_Level'].apply(extract_blood_sugar_value)
if 'Last_PPBS_Level' in df.columns:
    df['PPBS_Numeric'] = df['Last_PPBS_Level'].apply(extract_blood_sugar_value)
if 'Last_HbA1c_Level' in df.columns:
    df['HbA1c_Numeric'] = df['Last_HbA1c_Level'].apply(extract_hba1c_value)

# Process Likert scale responses for convenience fields
numeric_scale_fields = [
    'Belief_Traditional_Foods_Help', 
    'Interest_Traditional_Food_Recommendations',
    'Prefer_Website_Recommendations',
    'Website_Comfort_Level',
    'Willingness_Try_Traditional_Website',
    'Expected_Satisfaction_Level',
    'Likelihood_Follow_Plans'
]

for field in numeric_scale_fields:
    if field in df.columns:
        df[f'{field}_Numeric'] = df[field].apply(convert_to_numeric)

# Handle multi-select fields by creating indicator columns
def create_indicator_columns(df, column_name):
    if column_name not in df.columns:
        return df
    
    # Split the multi-select values
    values = df[column_name].astype(str).str.split(';').explode().dropna().unique()
    
    for value in values:
        value_clean = value.strip()
        # Create a new column for each unique value
        new_col_name = f"{column_name}_{value_clean.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')}"[:63]
        df[new_col_name] = df[column_name].astype(str).apply(lambda x: 1 if pd.notna(x) and value_clean in x else 0)
    
    return df

# Create indicator columns for multi-select fields
multi_select_fields = [
    'Other_Chronic_Diseases',
    'Blood_Sugar_Monitoring_Methods',
    'Meal_Preparation_Methods',
    'Diet_Challenges',
    'Consumption_Traditional_Grains',
    'Consumption_Traditional_Rice',
    'Consumption_Herbal_Porridges',
    'Consumption_Traditional_Vegetables',
    'Consumption_Traditional_Curries',
    'Consumption_Herbal_Drinks',
    'Consumption_Traditional_Snacks',
    'Personalization_Factors',
    'Factors_Increasing_Usage'
]

for field in multi_select_fields:
    df = create_indicator_columns(df, field)

# Save the cleaned file
output_file = "Survey_Dietary_Habit_Galle-cleaned.csv"
df.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")

# Function to generate mapped CSV
def generate_mapped_csv(df, output_file):
    """
    Generate a new CSV file mapping the cleaned data to the specified structure.
    
    Parameters:
    df (DataFrame): The cleaned DataFrame
    output_file (str): Path to save the new CSV file
    """
    # Create a new DataFrame with the desired columns
    mapped_df = pd.DataFrame(columns=[
        'User ID', 'Age Group', 'Gender', 'Weight range (kg)', 'Height range (cm)', 
        'BMI Category', 'Current living : Urban/Rural', 'Diabetes Duration', 
        'Medication Duration', 'Other Chronic Diseases', 'Fasting Blood Sugar (mg/dL)', 
        'Postprandial Blood Sugar (mg/dL)', 'HbA1c Level (%)', 'Dietary Preferences', 
        'Allergies', 'Avoidance Foods', 'Cooking Method'
    ])
    
    # Generate User IDs
    num_records = len(df)
    mapped_df['User ID'] = [f'U{str(i+1).zfill(3)}' for i in range(num_records)]
    
    # Map columns from original data to new structure
    if 'Age_Group' in df.columns:
        mapped_df['Age Group'] = df['Age_Group']
    if 'Gender' in df.columns:
        mapped_df['Gender'] = df['Gender']
    if 'Weight_kg' in df.columns:
        mapped_df['Weight range (kg)'] = df['Weight_kg']
    if 'Height_cm' in df.columns:
        mapped_df['Height range (cm)'] = df['Height_cm']
    
    # Calculate BMI Category
    def calculate_bmi_category(weight, height):
        if pd.isna(weight) or pd.isna(height):
            return None
        
        # Extract numeric values for calculation
        weight_val = extract_lower_value(weight)
        height_val = extract_lower_value(height)
        
        if not weight_val or not height_val:
            return None
        
        # Calculate BMI: weight(kg) / height(m)²
        bmi = weight_val / ((height_val / 100) ** 2)
        
        # Determine BMI category
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Normal'
        elif 25 <= bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    # Apply BMI calculation if needed columns exist
    if 'Weight_kg' in df.columns and 'Height_cm' in df.columns:
        mapped_df['BMI Category'] = df.apply(lambda row: calculate_bmi_category(row['Weight_kg'], row['Height_cm']), axis=1)
    
    # Map other columns
    if 'Location' in df.columns:
        mapped_df['Current living : Urban/Rural'] = df['Location'].apply(lambda x: 'Urban' if x == 'Urban area' else 'Rural' if x == 'Rural area' else None)
    if 'Years_With_Diabetes' in df.columns:
        mapped_df['Diabetes Duration'] = df['Years_With_Diabetes']
    if 'Years_On_Medication' in df.columns:
        mapped_df['Medication Duration'] = df['Years_On_Medication']
    
    # Handle Other Chronic Diseases
    if 'Other_Chronic_Diseases' in df.columns:
        mapped_df['Other Chronic Diseases'] = df['Other_Chronic_Diseases'].fillna('None')
    
    # Map blood sugar levels
    if 'Last_FBS_Level' in df.columns:
        mapped_df['Fasting Blood Sugar (mg/dL)'] = df['Last_FBS_Level']
    if 'Last_PPBS_Level' in df.columns:
        mapped_df['Postprandial Blood Sugar (mg/dL)'] = df['Last_PPBS_Level']
    if 'Last_HbA1c_Level' in df.columns:
        mapped_df['HbA1c Level (%)'] = df['Last_HbA1c_Level']
    
    # Map dietary information
    if 'Dietary_Preference' in df.columns:
        mapped_df['Dietary Preferences'] = df['Dietary_Preference']
    if 'Has_Food_Allergies' in df.columns:
        mapped_df['Allergies'] = df['Has_Food_Allergies'].apply(lambda x: 'None' if pd.isna(x) or x == 'No' else x)
    if 'Foods_Avoided_Health_Reasons' in df.columns:
        mapped_df['Avoidance Foods'] = df['Foods_Avoided_Health_Reasons'].fillna('None')
    
    # Map cooking methods
    if 'Meal_Preparation_Methods' in df.columns:
        mapped_df['Cooking Method'] = df['Meal_Preparation_Methods'].fillna('Unknown')
    
    # Save the mapped data to a new CSV file
    mapped_df.to_csv(output_file, index=False)
    print(f"Mapped dataset saved to {output_file}")
    
    return mapped_df

# Generate the mapped file
mapped_output_file = "Survey_Dietary_Habit_Galle-mapped.csv"
mapped_df = generate_mapped_csv(df, mapped_output_file)
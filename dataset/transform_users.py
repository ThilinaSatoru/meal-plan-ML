import pandas as pd

# Load the dataset, keeping 'None' as strings (not converting to NaN)
df = pd.read_csv('user.csv', na_values=[], keep_default_na=False)


# Function to determine FBS category
def get_fbs_category(fasting_glucose):
    if pd.isna(fasting_glucose) or fasting_glucose == 'None':
        return 'None'
    fasting_glucose = float(fasting_glucose)
    if fasting_glucose < 100:
        return 'Normal'
    elif 100 <= fasting_glucose <= 125:
        return 'Prediabetes'
    else:
        return 'Diabetes'


# Function to determine HbA1c category
def get_hba1c_category(fasting, postprandial):
    if pd.isna(fasting) or pd.isna(postprandial) or fasting == 'None' or postprandial == 'None':
        return 'None'
    fasting = float(fasting)
    postprandial = float(postprandial)
    avg_glucose = (fasting + postprandial) / 2
    hba1c_value = (avg_glucose + 46.7) / 28.7
    hba1c_value = round(hba1c_value, 1)

    if hba1c_value < 5.7:
        return 'Normal'
    elif 5.7 <= hba1c_value <= 6.4:
        return 'Prediabetes'
    else:
        return 'Diabetes'


# Update FBS and HbA1c columns
df['FBS'] = df['FastingGlucose'].apply(get_fbs_category)
df['HbA1c'] = df.apply(lambda row: get_hba1c_category(row['FastingGlucose'], row['PostprandialGlucose']), axis=1)

# Save the corrected dataset (preserving 'None' strings)
df.to_csv('diabetes_user_profiles_with_mealID.csv', index=False, na_rep='None')

print("✅ FBS and HbA1c values updated correctly. 'None' strings preserved.")
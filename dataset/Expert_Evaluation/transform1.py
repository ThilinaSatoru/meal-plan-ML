import pandas as pd

# Step 1: Load the original survey CSV file into a DataFrame
input_csv_file = r'Survey_Expert_Evaluation_Galle-Expanded.csv'  # Replace with your input CSV file path
# df = pd.read_csv(input_csv_file)
# Load the file with different encoding
df = pd.read_csv(input_csv_file, encoding='utf-8-sig', sep=",", skip_blank_lines=False)
print(f"Total Rows and Columns after reading: {df.shape}")


# Define the correct column mapping
column_mapping = {
    "Timestamp": "Timestamp",
    "  What is your professional specialization?  ": "Professional_Specialization",
    "How many years of experience do you have in treating/managing diabetic patients? ": "Experience_Years",
    "How often do you provide dietary recommendations to elderly diabetic patients?": "Dietary_Recommendation_Frequency",
    " Do elderly diabetic patients require different dietary guidelines compared to younger adults with diabetes?  ": "Different_Guidelines_for_Elderly",
    "What are the most common diabetes-related complications you observe in elderly patients? ": "Common_Complications_Elderly",
    "What challenges do elderly diabetics face in maintaining proper nutrition? ": "Nutrition_Challenges",
    "How important is portion control in meal planning for elderly diabetic patients? ": "Portion_Control_Importance",
    "Have you ever used a digital tool or app to provide dietary recommendations for elderly diabetic patients? ": "Used_Digital_Tool",
    "What is your primary method for diagnosing diabetes in elderly patients? ": "Diagnosis_Method",
    "In your professional opinion, what fasting blood sugar (FBS) level indicates different stages of diabetes? ": "FBS_Stages",
    "What postprandial blood sugar (after meals) level do you consider normal or concerning?": "Postprandial_Blood_Sugar_Levels",
    "What HbA1c percentage do you consider well-managed for elderly diabetics?": "HbA1c_Threshold",
    "How do you assess blood sugar control in elderly diabetic patients? ": "Blood_Sugar_Control_Assessment",
    "What dietary or lifestyle factors do you consider when recommending meal plans for elderly diabetics? ": "Dietary_Lifestyle_Factors",
    "How well does the system align with existing dietary guidelines for elderly diabetics? ": "System_Alignment_With_Guidelines",
    "  Do you believe Sri Lankan traditional foods can be effectively integrated into structured diabetes-friendly meal plans? ": "Belief_in_Traditional_Foods",
    "How scientifically valid do you consider the inclusion of traditional Sri Lankan foods in diabetes management? ": "Scientific_Validity_of_Traditional_Foods",
    "Which food categories should be excluded from the system due to their potential negative effects on diabetic patients? ": "Excluded_Food_Categories",
    " How frequently do your patients inquire about traditional foods for diabetes management?  ": "Patient_Inquiry_Traditional_Foods",
    " Would you recommend a personalized meal plan based on traditional Sri Lankan foods for diabetic patients?  ": "Recommend_Traditional_Food_Plan",
    "In your opinion, how suitable are traditional Sri Lankan foods for elderly people with diabetes?  ": "Suitability_of_Traditional_Foods",
    "How important is it to include traditional Sri Lankan rice varieties in a diabetes-friendly meal plan? (e.g., Red Raw Rice, Heenati Rice, Suwandel Rice, Pachchaperumal Rice)": "Importance_of_Traditional_Rice",
    "How important is it to include traditional Sri Lankan grains & cereals in a diabetes meal plan? (e.g., Kurakkan (Finger Millet), Meneri (Foxtail Millet), Thana Hal (Sorghum), Maize (Corn))": "Importance_of_Traditional_Grains",
    "How important is it to include herbal porridges in diabetes meal plans for elderly individuals? (e.g., Gotu Kola Kenda, Polpala Kenda, Hathawariya Kenda)": "Importance_of_Herbal_Porridges",
    "How important is it to include traditional Sri Lankan vegetables & leafy greens in a diabetes meal plan? (e.g., Gotu Kola Sambol, Murunga Kola Mallum, Kohila Mallum)": "Importance_of_Vegetables_Leafy_Greens",
    "How important is it to include traditional Sri Lankan curries & dishes in diabetes meal plans? (e.g., Kollu (Horse Gram Curry), Dhal Curry, Green Gram Curry, Jackfruit Seeds Curry)": "Importance_of_Traditional_Curries",
    "How important is it to include herbal drinks in a diabetes-friendly meal plan? (e.g., Beli Mal Tea, Ranawara Tea, Coriander Tea (Koththamalli))": "Importance_of_Herbal_Drinks",
    "How important is it to include traditional Sri Lankan snacks in diabetes-friendly diets? (e.g., Roasted Gram (Kadala), Boiled Jackfruit Seeds (Kos Ata), Thala Guli (Sesame Balls))": "Importance_of_Traditional_Snacks",
    "How accurately does the system consider individual health conditions when generating meal recommendations? (e.g., hypertension, kidney disease, cholesterol issues)?  ": "System_Accuracy_for_Health_Conditions",
    "What additional health conditions should be integrated into meal plan adjustments?  ": "Additional_Health_Conditions",
    " How important is it that the system allows personalized adjustments to meal plans?  ": "Importance_of_Personalized_Adjustments",
    "How easy is it for elderly diabetics to navigate and use the system? ": "System_Navigation_Ease",
    "What factors could increase the adoption of this system among elderly diabetics?  ": "Factors_for_System_Adoption",
    "What are the main obstacles that might prevent elderly diabetics from using the system effectively? ": "Obstacles_for_System_Use",
    "Based on your expertise, How would you rate the accuracy of the system’s recommended meals in meeting the nutritional and health needs of elderly diabetics? ": "Accuracy_of_Recommended_Meals",
    "How likely are you to recommend this system to diabetic patients if improvements are made based on expert feedback? ": "Likelihood_to_Recommend_System",
    "Would you be willing to collaborate in future research for refining this system? ": "Willingness_for_Research_Collaboration",
    "How suitable do you think a website would be for delivering meal recommendations to elderly diabetic users?  ": "Suitability_of_Website_for_Recommendations",
    "Would you be willing to use this system as a reference tool when advising diabetic patients?  ": "Use_System_as_Reference_Tool",
    "What would make you more likely to trust in Machine Learning based food recommendation system?": "Trust_Factors_for_ML_Recommendation",
}


# Debugging print statements
print("Total Rows and Columns before renaming:", df.shape)
print("First 5 rows before renaming:", df.head())

# Rename columns
df.rename(columns=column_mapping, inplace=True)

# Debugging print statements after renaming
print("Total Rows and Columns after renaming:", df.shape)
print("First 5 rows after renaming:", df.head())

# Convert numeric columns to proper format (if applicable)
numeric_columns = ["FBS_Stages", "Postprandial_Blood_Sugar_Levels", "HbA1c_Threshold"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts errors to NaN

# Save the cleaned file
output_file = "Survey_Expert_Evaluation_Galle-cleaned.csv"
df.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")


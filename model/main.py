import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from diabetes_meal_recommendation import (
    data_pipeline, train_model, evaluate_model, inference_meal
)


def explore_new_dataset(user_data_path='diabetes_user_profiles_new_format.csv',
                       meal_data_path='sri_lankan_meal_dataset_new_format.csv'):
    """
    Explore the new dataset formats with visualizations
    """
    # Load datasets
    users_df = pd.read_csv(user_data_path)
    meals_df = pd.read_csv(meal_data_path)
    
    print(f"User dataset shape: {users_df.shape}")
    print(f"Meal dataset shape: {meals_df.shape}")
    
    # Display sample of user data
    print("\nUser data sample:")
    print(users_df.head(2))
    
    # Display sample of meal data
    print("\nMeal data sample:")
    print(meals_df.head(2))
    
    # Distribution of categorical variables in user data
    cat_cols = [
        'Age Group', 'Gender', 'BMI Category', 'Current living : Urban/Rural',
        'Diabetes Duration', 'Dietary Preferences'
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, data=users_df, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of meal types
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Meal Type', data=meals_df)
    plt.title('Distribution of Meal Types')
    plt.show()
    
    # Nutritional information comparison
    plt.figure(figsize=(12, 8))
    
    nutrition_cols = ['Calories (kcal)', 'Carbohydrates (g)', 'Protein (g)', 'Fats (g)', 'Fiber (g)']
    melted_df = pd.melt(meals_df, id_vars=['Meal ID', 'Meal Type'], 
                        value_vars=nutrition_cols,
                        var_name='Nutrient', value_name='Value')
    
    sns.barplot(x='Meal ID', y='Value', hue='Nutrient', data=melted_df)
    plt.title('Nutritional Comparison of Meals')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Glycemic Index distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Meal ID', y='Glycemic Index (GI)', data=meals_df, palette='viridis')
    plt.title('Glycemic Index by Meal')
    plt.axhline(y=55, color='r', linestyle='--', label='Low GI Threshold')
    plt.legend()
    plt.show()
    
    # Return datasets for further analysis
    return users_df, meals_df

def count_multi_select_values(df, column):
    """
    Count occurrences of each value in a multi-select column (semicolon-separated)
    """
    all_values = []
    for values in df[column].dropna():
        all_values.extend([v.strip() for v in values.split(';')])
    
    value_counts = pd.Series(all_values).value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return value_counts


def main():
    """
    Run the complete diabetes meal recommendation pipeline
    """
    print("="*50)
    print("Diabetes Meal Recommendation System")
    print("="*50)
    
    # Step 1: Generate sample data
    print("\nStep 1: Checking if sample data exists...")
    try:
        # Try to load the data to see if it exists
        user_df = pd.read_csv('diabetes_user_profiles_new_format.csv')
        meal_df = pd.read_csv('sri_lankan_meal_dataset_new_format.csv')
        print("Sample data found. Using existing data.")
    except FileNotFoundError:
        print("Sample data not found. Creating sample data...")
        # Import and run the sample data creation script
        import sample_user_data
        
    # Step 2: Process data
    print("\nStep 2: Processing data...")
    X, Y, encoder = data_pipeline()
    print(f"Data processed. X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    model, X_train, X_test, Y_train, Y_test = train_model(X, Y)
    print("Model training complete.")
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluating model...")
    evaluate_model(model, X_train, Y_train, X_test, Y_test, encoder)
    
    # Step 5: Test inference
    print("\nStep 5: Testing inference with a sample user...")
    # Sample user data
    sample_user = {
        "User ID": "U999",
        "Age Group": "60 – 64 years",
        "Gender": "Female",
        "Weight range (kg)": "50 – 59 kg",
        "Height range (cm)": "150 – 159 cm",
        "BMI Category": "Normal",
        "Current living : Urban/Rural": "Urban",
        "Diabetes Duration": "1 – 3 years",
        "Medication Duration": "1 – 3 years",
        "Other Chronic Diseases": "High blood pressure (Hypertension);High cholesterol",
        "Fasting Blood Sugar (mg/dL)": "Less than 70 mg/dL",
        "Postprandial Blood Sugar (mg/dL)": "Less than 140 mg/dL",
        "HbA1c Level (%)": "Less than 5.7%",
        "Dietary Preferences": "Non-vegetarian",
        "Allergies": "No known allergies",
        "Avoidance Foods": "No specific food avoidance",
        "Cooking Method": "Boiling (e.g., boiled rice, boiled vegetables);Stir-frying with minimal oil"
    }
    
    result = inference_meal(sample_user)
    print("\nRecommended meal for sample user:")
    print(f"Meal ID: {result['Meal ID']}")
    print("\nMeal Details:")
    for key, value in result['Meal'][0].items():
        print(f"  {key}: {value}")
    
    print("\nPipeline complete! The model is ready to use.")
    print("="*50)

if __name__ == "__main__":
    main()
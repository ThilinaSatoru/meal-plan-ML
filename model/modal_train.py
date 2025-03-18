import pickle
import pprint
import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
from collections import defaultdict
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings('ignore')

# Define paths for the new dataset format
USER_DATA_PATH = 'diabetes_user_profiles_new_format.csv'
MEAL_DATA_PATH = 'sri_lankan_meal_dataset_new_format.csv'

# Define columns for the new dataset format
cat_columns = [
    'Gender', 'Age Group', 'Weight range (kg)', 'Height range (cm)', 
    'BMI Category', 'Current living : Urban/Rural', 'Diabetes Duration', 
    'Medication Duration', 'Other Chronic Diseases', 'Dietary Preferences', 
    'Allergies', 'Avoidance Foods', 'Cooking Method'
]

num_columns = [
    'Fasting Blood Sugar (mg/dL)', 'Postprandial Blood Sugar (mg/dL)', 'HbA1c Level (%)'
]

output_column = ['Meal ID']

def preprocess_numeric_columns(df):
    """
    Process numeric columns that might contain ranges or text
    """
    # Process Fasting Blood Sugar
    def process_fbs(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return value
        # Handle "Less than X" format
        if "Less than" in value:
            return float(value.replace("Less than", "").replace("mg/dL", "").strip())
        # Handle ranges like "70-100 mg/dL"
        if "-" in value:
            low, high = value.split("-")
            low = float(low.strip().replace("mg/dL", ""))
            high = float(high.strip().replace("mg/dL", ""))
            return (low + high) / 2
        return float(value.replace("mg/dL", "").strip())
    
    # Process Postprandial Blood Sugar
    def process_ppbs(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return value
        # Handle "Less than X" format
        if "Less than" in value:
            return float(value.replace("Less than", "").replace("mg/dL", "").strip())
        # Handle ranges
        if "-" in value:
            low, high = value.split("-")
            low = float(low.strip().replace("mg/dL", ""))
            high = float(high.strip().replace("mg/dL", ""))
            return (low + high) / 2
        return float(value.replace("mg/dL", "").strip())
    
    # Process HbA1c
    def process_hba1c(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return value
        # Handle "Less than X" format
        if "Less than" in value:
            return float(value.replace("Less than", "").replace("%", "").strip())
        # Handle ranges
        if "-" in value:
            low, high = value.split("-")
            low = float(low.strip().replace("%", ""))
            high = float(high.strip().replace("%", ""))
            return (low + high) / 2
        return float(value.replace("%", "").strip())
    
    # Apply processing to each column
    df['Fasting Blood Sugar (mg/dL)'] = df['Fasting Blood Sugar (mg/dL)'].apply(process_fbs)
    df['Postprandial Blood Sugar (mg/dL)'] = df['Postprandial Blood Sugar (mg/dL)'].apply(process_ppbs)
    df['HbA1c Level (%)'] = df['HbA1c Level (%)'].apply(process_hba1c)
    
    return df

def data_pipeline(
        datapath=USER_DATA_PATH,
        cat_columns=cat_columns,
        num_columns=num_columns,
        output_column=output_column
    ):
    """
    Process the new dataset format and prepare it for model training
    """
    # Read the data
    df = pd.read_csv(datapath)
    
    # Process numeric columns
    df = preprocess_numeric_columns(df)
    
    # Create encoders for categorical columns
    encoder = defaultdict(LabelEncoder)
    for col in cat_columns + output_column:
        encoder[col].fit(df[col].fillna('Unknown'))
        df[col] = df[col].fillna('Unknown')
        df[col] = encoder[col].transform(df[col])
    
    # Save the encoders
    with open('encoder_meal_new.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    # One-hot encode multi-select fields (separated by semicolons)
    multi_select_cols = ['Other Chronic Diseases', 'Allergies', 'Avoidance Foods', 'Cooking Method']
    
    # Create X (features) and Y (target)
    X = df[num_columns].copy()
    
    # Add encoded categorical columns to X
    for col in cat_columns:
        if col not in multi_select_cols:
            X[col] = df[col]
    
    Y = df[output_column]
    
    X, Y = np.array(X), np.array(Y).ravel()
    return X, Y, encoder

def train_model(X, Y):
    """
    Train a CatBoost model on the prepared data
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    
    # Train the model
    cat = CatBoostClassifier(
        iterations=1000, 
        learning_rate=0.1, 
        loss_function='MultiClass', 
        depth=6
    )
    
    cat.fit(
        X, Y, 
        eval_set=(X_test, Y_test), 
        verbose=100
    )
    
    # Save the model
    with open('model_meal_new.pkl', 'wb') as f:
        pickle.dump(cat, f)
    
    return cat, X_train, X_test, Y_train, Y_test

def evaluate_model(model, X_train, Y_train, X_test, Y_test, encoder):
    """
    Evaluate the trained model and display metrics
    """
    P_train = model.predict(X_train)
    P_test = model.predict(X_test)
    
    # Get class names
    target_names = encoder['Meal ID'].classes_
    target_names = [f'Meal ID: {name}' for name in target_names]
    
    # Print classification report
    print("---------------------- Train CLS REPORT ----------------------")
    clf_report = classification_report(
        Y_train,
        P_train,
        target_names=target_names
    )
    print(clf_report)
    
    print("---------------------- Test CLS REPORT ----------------------")
    clf_report = classification_report(
        Y_test, 
        P_test,
        target_names=target_names
    )
    print(clf_report)
    
    # Plot confusion matrices
    cm = confusion_matrix(Y_train, P_train)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Train Confusion Matrix')
    plt.show()
    
    cm = confusion_matrix(Y_test, P_test)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Test Confusion Matrix')
    plt.show()

def inference_meal(
        sample_json,
        meal_path=MEAL_DATA_PATH,
        cat_columns=cat_columns,
        num_columns=num_columns
    ):
    """
    Make a meal recommendation for a user based on their profile
    """
    # Load encoders and model
    with open('encoder_meal_new.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    with open('model_meal_new.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Create DataFrame from input
    df = pd.DataFrame([sample_json])
    
    # Process numeric columns
    df = preprocess_numeric_columns(df)
    
    # Encode categorical columns
    for col in cat_columns:
        df[col] = df[col].fillna('Unknown')
        df[col] = encoder[col].transform(df[col])
    
    # Extract features in the same order as training
    x = df[num_columns + [col for col in cat_columns if col not in ['Other Chronic Diseases', 'Allergies', 'Avoidance Foods', 'Cooking Method']]].values
    
    # Make prediction
    p = model.predict(x)
    p = int(p.squeeze())
    meal_id = encoder['Meal ID'].inverse_transform([p])[0]
    
    # Get meal details
    df_meal = pd.read_csv(meal_path)
    df_meal = df_meal[df_meal['Meal ID'] == meal_id]
    
    meal_dict = df_meal.to_dict(orient='records')
    
    response = {}
    response['Meal ID'] = meal_id
    response['Meal'] = meal_dict
    return response

# Example usage
def run_pipeline():
    """
    Run the complete pipeline: data processing, model training, and evaluation
    """
    print("Processing data...")
    X, Y, encoder = data_pipeline()
    
    print("Training model...")
    model, X_train, X_test, Y_train, Y_test = train_model(X, Y)
    
    print("Evaluating model...")
    evaluate_model(model, X_train, Y_train, X_test, Y_test, encoder)
    
    print("Pipeline complete!")

def example_inference():
    """
    Example of making a prediction with the trained model
    """
    # Sample user data in new format
    sample_user = {
        "User ID": "U099",
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
    print("Recommended meal:")
    pprint.pprint(result)

# Uncomment to run the pipeline
# run_pipeline()

# Uncomment to test inference with a sample user
# example_inference()
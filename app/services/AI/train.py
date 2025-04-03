import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
cat_columns = []


def data_pipeline(datapath):
    df = pd.read_csv(datapath)
    df.drop(columns=['RecordID', 'DiagnosedYearsAgo', 'FoodsAvoided', 'TriggerFoods', 'DietFollowed'], inplace=True)

    # Handle multi-label "OtherConditions"
    conditions = df['OtherConditions'].str.get_dummies(', ')
    df = pd.concat([df.drop(columns=['OtherConditions']), conditions], axis=1)

    # Define columns
    num_columns = ['Age', 'Height', 'Weight', 'BMI', 'FastingGlucose', 'PostprandialGlucose']
    cat_columns = ['Gender', 'Location', 'FBS', 'HbA1c', 'FavoriteFoods', 'Allergies', 'TraditionalFoods',
                   'CookingMethods']
    output_column = ['MealID']

    # Encode ordinal features separately
    frequency_map = {'Rarely': 0, 'Few times a week': 1, 'Daily': 2}
    df['CookingFrequency'] = df['CookingFrequency'].map(frequency_map)
    num_columns.append('CookingFrequency')  # Treat as numerical

    # Encode remaining categoricals
    encoder = defaultdict(LabelEncoder)
    for col in cat_columns + ['MealID']:
        df[col] = encoder[col].fit_transform(df[col])

    # Save the encoder
    with open('encoder_meal.pkl', 'wb') as f:
        pickle.dump(dict(encoder), f)

    X = df.drop(columns=output_column)
    Y = df[output_column].values.ravel()

    return X, Y, encoder, df


def train_model(X, Y):
    print("\n" + "=" * 50)
    print("Starting model training...")

    # Get categorical feature indices
    cat_features_indices = [X.columns.get_loc(col) for col in cat_columns]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Display training/testing split information
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Model parameters
    params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'loss_function': 'MultiClass',
        'depth': 6,
        'cat_features': cat_features_indices,  # Critical!
        'auto_class_weights': 'Balanced',  # Handle imbalance
        'verbose': 100,
        'random_seed': 42
    }

    print("\nModel parameters:")
    for param, value in params.items():
        print(f"- {param}: {value}")

    # Train the model
    cat = CatBoostClassifier(**params)
    cat.fit(X_train, Y_train,
            eval_set=(X_test, Y_test),
            early_stopping_rounds=50,
            plot=False)

    # Save the model
    with open('model_meal.pkl', 'wb') as f:
        pickle.dump(cat, f)

    print(f"\nBest iteration: {cat.get_best_iteration()}")

    return cat, X_train, X_test, Y_train, Y_test


def evaluate_model(model, X_train, X_test, Y_train, Y_test, encoder, X, Y, df):
    print("\n" + "=" * 50)
    print("Model Evaluation:")

    # Make predictions
    P_train = model.predict(X_train)
    P_test = model.predict(X_test)

    # Generate class names for reports
    target_names = [f'Meal ID: {name}' for name in encoder['MealID'].classes_]

    # Basic accuracy scores
    train_accuracy = accuracy_score(Y_train, P_train)
    test_accuracy = accuracy_score(Y_test, P_test)

    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Detailed classification reports
    print("\nTrain Classification Report")
    train_report = classification_report(Y_train, P_train, target_names=target_names, output_dict=True)
    print(classification_report(Y_train, P_train, target_names=target_names))

    print("\nTest Classification Report")
    test_report = classification_report(Y_test, P_test, target_names=target_names, output_dict=True)
    print(classification_report(Y_test, P_test, target_names=target_names))

    # Cross-validation for robustness check
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 most important features:")
    print(importance_df.head(10))

    # Show sample predictions with ground truth
    print("\n" + "=" * 50)
    print("Sample Predictions from Test Set:")

    # Randomly select 5 samples from test set
    sample_indices = np.random.choice(len(X_test), size=5, replace=False)
    sample_X = X_test.iloc[sample_indices]
    sample_Y = Y_test[sample_indices]

    # Get predictions and probabilities
    sample_preds = model.predict(sample_X)
    sample_probs = model.predict_proba(sample_X)

    # Create a DataFrame for display
    results = []

    for i, idx in enumerate(sample_indices):
        actual_meal_id = encoder['MealID'].inverse_transform([sample_Y[i]])[0]
        predicted_meal_id = encoder['MealID'].inverse_transform([sample_preds[i]])[0]
        confidence = np.max(sample_probs[i]) * 100

        # Extract key features for this sample
        sample_features = {}
        for feature in ['Age', 'FBS', 'FastingGlucose', 'PostprandialGlucose']:
            if feature in X_test.columns:
                value = sample_X.iloc[i][feature]

                # Decode categorical features
                if feature == 'FBS':
                    value = encoder[feature].inverse_transform([int(value)])[0]

                sample_features[feature] = value

        result = {
            'Sample': i + 1,
            'Actual Meal ID': actual_meal_id,
            'Predicted Meal ID': predicted_meal_id,
            'Confidence': f"{confidence:.2f}%",
            'Status': 'Correct' if actual_meal_id == predicted_meal_id else 'Incorrect',
            **sample_features
        }
        results.append(result)

    # Display the sample predictions
    samples_df = pd.DataFrame(results)
    print(samples_df)

    print("\n" + "=" * 50)

    return train_report, test_report, importance_df


def main():
    datapath = 'E:\\PartProjects\\MealPlan\\meal plan recommendation\\dataset\\diabetes_user_profiles_with_mealID.csv'
    print("=" * 50)
    print("Diabetes Meal Recommendation Model")
    print("=" * 50)

    X, Y, encoder, df = data_pipeline(datapath)
    model, X_train, X_test, Y_train, Y_test = train_model(X, Y)
    train_report, test_report, importance_df = evaluate_model(model, X_train, X_test, Y_train, Y_test, encoder, X, Y,
                                                              df)

    # Additional analysis: Meal recommendation patterns based on diabetes type
    try:
        print("\nAnalyzing meal recommendations by diabetes type...")

        # Since we've encoded the data, we need to decode for analysis
        diabetes_types = encoder['FBS'].classes_
        meal_ids = encoder['MealID'].classes_

        print(f"\nNumber of diabetes types: {len(diabetes_types)}")
        print(f"Number of unique meal IDs: {len(meal_ids)}")

        print("\nModel training complete with detailed performance metrics!")
        print("=" * 50)

    except Exception as e:
        print(f"Error in additional analysis: {e}")


if __name__ == "__main__":
    main()

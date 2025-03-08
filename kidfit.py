import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
try:
    data = pd.read_csv(r"C:\Users\sharm\Desktop\children_clothing_data.csv", encoding='ISO-8859-1')  # Adjust the file path
except FileNotFoundError:
    print("Dataset file not found. Please provide a valid dataset.")
    exit()

# Data Overview
print(data.head())

# Check for missing values and handle them
if data.isnull().sum().any():
    print("Dataset contains missing values. Handling them...")
    data = data.fillna(data.median(numeric_only=True))  # Filling missing numerical values
    data = data.fillna(data.mode().iloc[0])  # Filling missing categorical values

# Features and Target
X = data[['height', 'weight', 'age', 'gender']]  # Features  
y = data['size']  # Target variable

# Preprocessing steps
# 1. StandardScaler for numeric columns: 'height', 'weight', 'age'
# 2. OneHotEncoder for categorical columns: 'gender'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['height', 'weight', 'age']),  # Scale numerical columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender'])  # One-hot encode 'gender'
    ])
    

# Create an SVM classifier pipeline with preprocessing and SVM model
svm_model = make_pipeline(preprocessor, SVC(kernel='rbf', random_state=42))

# Perform cross-validation (using 5-fold cross-validation for example)
cv_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')  # cv=5 for 5-fold CV
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average cross-validation accuracy: {np.mean(cv_scores):.2f}")

# Train the model on the full dataset after cross-validation
svm_model.fit(X, y)

# Evaluate the model on a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = svm_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.2f}")

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("Confusion Matrix:\n", cm)

# Optionally, visualize the confusion matrix using Seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Function to ask the user for input and predict clothing size
def get_user_input():
    while True:
        try:
            # Collecting user inputs
            height = float(input("Enter height (in cm): "))
            weight = float(input("Enter weight (in kg): "))
            age = int(input("Enter age (in years): "))
            
            # Ensure gender input is valid
            gender = input("Enter gender (male/female): ").strip().lower()
            if gender not in ['male', 'female']:
                raise ValueError("Gender must be 'male' or 'female'. Please try again.")
            
            break  # If all inputs are valid, exit the loop
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
    return pd.DataFrame({'height': [height], 'weight': [weight], 'age': [age], 'gender': [gender]})

# Function to make predictions based on user input
def predict_clothing_size(svm_model):
    print("Enter the following details to get the recommended clothing size:")

    # Get user input
    user_input = get_user_input()

    # Make prediction
    predicted_size = svm_model.predict(user_input)

    # Display the result
    print(f"Recommended clothing size: {predicted_size[0]}")

# Call the function to make a prediction
predict_clothing_size(svm_model)

# Optionally, save the model for future use
joblib.dump(svm_model, 'kids_clothing_svm_model.pkl')
print("Model saved as 'kids_clothing_svm_model.pkl'")


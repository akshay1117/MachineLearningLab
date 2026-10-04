# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 3: LOGISTIC REGRESSION
# DATASET: PIMA INDIANS DIABETES
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP03_LOGISTIC_REGRESSION_PIMA_DIABETES/logistic_regression_pima_diabetes.py
# 5. Review the console output for evaluation metrics.
# 
# ==============================================================================

# Logistic Regression on Pima Indians Diabetes Dataset
# Predict diabetes Outcome (0 = No, 1 = Yes) using patient health metrics,
# and compare Logistic Regression performance with and without feature scaling.

# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 2: Load the Dataset
# The Pima dataset has no header row, so we define column names manually
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Loading directly from the remote URL as per the academic reference
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
data = pd.read_csv(url, names=columns)

# Step 3: Select Input Features and Target Variable
# X contains all the health metrics (features)
X = data.drop("Outcome", axis=1)
# y is the binary outcome we want to predict (0 = negative, 1 = positive)
y = data["Outcome"]

# Step 4: Split the Dataset into Training and Testing sets
# 80% used for training, 20% for testing. random_state ensures reproducible splits.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# ==============================================================================
# PART A - Logistic Regression WITHOUT Feature Scaling
# ==============================================================================

# Create the model. max_iter is increased to 1000 to allow the solver to converge.
model = LogisticRegression(max_iter=1000)

# Train the model on raw (unscaled) feature values
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using standard classification metrics
print("Without Feature Scaling")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))


# ==============================================================================
# PART B - Logistic Regression WITH Feature Scaling
# ==============================================================================
# Scaling standardizes features by removing the mean and scaling to unit variance.
scaler = StandardScaler()

# Fit the scaler on training data only, then transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a new model on the scaled features
model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)

# Predict on the scaled test data
y_pred_scaled = model_scaled.predict(X_test_scaled)

# Evaluate the scaled model using the same metrics for fair comparison
print("\nWith Feature Scaling")
print("Accuracy :", accuracy_score(y_test, y_pred_scaled))
print("Precision:", precision_score(y_test, y_pred_scaled))
print("Recall :", recall_score(y_test, y_pred_scaled))
print("F1-Score :", f1_score(y_test, y_pred_scaled))


# ==============================================================================
# PART C - Comparison of Results (With vs Without Feature Scaling)
# ==============================================================================
# Create a DataFrame to display the metrics side-by-side cleanly
comparison = pd.DataFrame({
    "Model": ["Without Scaling", "With Scaling"],
    "Accuracy": [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_scaled)],
    "Precision": [precision_score(y_test, y_pred), precision_score(y_test, y_pred_scaled)],
    "Recall": [recall_score(y_test, y_pred), recall_score(y_test, y_pred_scaled)],
    "F1-Score": [f1_score(y_test, y_pred), f1_score(y_test, y_pred_scaled)]
})

print("\nModel Comparison")
print(comparison)

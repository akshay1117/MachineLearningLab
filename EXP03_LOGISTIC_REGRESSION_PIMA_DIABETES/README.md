# Experiment 3: Logistic Regression

## Aim

To implement a Logistic Regression classifier on the Pima Indians Diabetes dataset to predict whether a patient has diabetes based on diagnostic measurements.

## Task

Load the dataset, handle invalid zero values, split the data, scale the features, train a logistic regression model, make predictions, and evaluate the model using accuracy, confusion matrix, and ROC curve.

## Theory

**Logistic Regression** is a supervised machine learning algorithm used primarily for **binary classification** (where the output is either 0 or 1, Yes or No, True or False).

Despite its name containing "regression," it is a classification algorithm.

**How it works:**

1. It calculates a linear combination of the input features (`z = w₁x₁ + w₂x₂ + ... + b`).
2. It passes this value `z` through the **Sigmoid function** `σ(z) = 1 / (1 + e^(-z))`.
3. The Sigmoid function maps any real number into a probability value strictly between 0 and 1.
4. A **classification threshold** (usually 0.5) is applied. If the probability is ≥ 0.5, the class is predicted as 1 (Diabetes). Otherwise, it is 0 (No Diabetes). This creates a linear **decision boundary**.

## Implementation Plan

1. **Import Libraries:** Import Pandas, Numpy, and Scikit-learn modules.
2. **Load Dataset:** Load Pima Indians Diabetes dataset into a DataFrame using `pd.read_csv()`.
3. **Assign Columns:** Manually assign column names as the dataset lacks a header.
4. **Preprocess Data:** Replace invalid zero values (Glucose, Blood Pressure, etc.) with NaN and impute using column medians.
5. **Split Dataset:** Split into 80% training and 20% testing sets.
6. **Train Unscaled Model:** Create and train a logistic regression model using raw, unscaled training data.
7. **Evaluate Unscaled Model:** Evaluate using accuracy, precision, recall, and F1 Score on test data.
8. **Scale Features:** Apply `StandardScaler` to the input features.
9. **Train Scaled Model:** Train another logistic regression model using the scaled training data.
10. **Evaluate Scaled Model:** Predict test outcomes and calculate accuracy, precision, recall, and F1 score.
11. **Compare & Display:** Compare metrics of both models and display the results.

## Dataset

- **Name:** Pima Indians Diabetes Database
- **Source:** Originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
- **Target:** `Outcome` (Class variable: 0 = No diabetes, 1 = Diabetes).
- **Features:** 8 numerical variables (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
- **Obtaining it:** The Python script automatically downloads the dataset and saves it to `data/pima-indians-diabetes.csv` if it doesn't already exist locally.

## Algorithm

- `sklearn.linear_model.LogisticRegression` is used for the model.
- `sklearn.preprocessing.StandardScaler` is used to scale features so they have a mean of 0 and standard deviation of 1.

## Expected Output

When you run the program, the console will output:

- Dataset shape and head.
- A log indicating invalid zeros were handled.
- Evaluation metrics (Accuracy, Precision, Recall, F1).
- An example prediction showing the raw patient data, the calculated probability, the model's prediction, and the actual outcome.
- Two visualizations will appear side-by-side: a Confusion Matrix heatmap and an ROC Curve.

## Viva Questions & Answers
1. **Why is Logistic Regression used for classification instead of Linear Regression?**
   **Answer:** Linear regression outputs continuous values that can be < 0 or > 1, and is highly sensitive to outliers. Logistic regression uses a sigmoid function to bound predictions between 0 and 1, making it ideal for probabilities and classification.
2. **What is the Sigmoid function and what is its output range?**
   **Answer:** The sigmoid function is an S-shaped curve defined as `1 / (1 + e^-x)`. It maps any real-valued number into a range between 0 and 1.
3. **What is the difference between a False Positive and a False Negative?**
   **Answer:** A False Positive is when the model predicts the positive class but the actual class is negative (Type I error). A False Negative is predicting the negative class when the actual class is positive (Type II error).
4. **Explain Precision, Recall, and F1-Score.**
   **Answer:** Precision is the ratio of true positives to total predicted positives (accuracy of positive predictions). Recall is the ratio of true positives to actual positives (ability to find all positive instances). F1-Score is the harmonic mean of Precision and Recall.
5. **What is the role of the decision threshold in Logistic Regression?**
   **Answer:** The decision threshold (usually 0.5) converts the predicted probability into a discrete class label. Probabilities above the threshold are assigned to the positive class (1), and below to the negative class (0).

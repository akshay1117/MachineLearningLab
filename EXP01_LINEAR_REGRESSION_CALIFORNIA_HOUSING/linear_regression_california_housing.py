# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 1: LINEAR REGRESSION WITH ONE VARIABLE
# DATASET: CALIFORNIA HOUSING
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP01_LINEAR_REGRESSION_CALIFORNIA_HOUSING/linear_regression_california_housing.py
# 5. Review the console output for evaluation metrics and close the plot window to exit.
# 
# ==============================================================================

# Linear Regression using Direct Least Squares and Gradient Descent 
# Dataset : California Housing Dataset 
# Objective: Predict the median house value using the average # number of rooms per household (AveRooms). 

# Step 1: Import Required Libraries
# pandas is used for data manipulation and analysis
import pandas as pd
# matplotlib is used for creating static, interactive, and animated visualizations
import matplotlib.pyplot as plt
# train_test_split splits arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# LinearRegression solves the regression using ordinary least squares (direct method)
# SGDRegressor solves the regression using Stochastic Gradient Descent (iterative optimization)
from sklearn.linear_model import LinearRegression, SGDRegressor
# mean_squared_error and r2_score are metrics to evaluate the performance of our models
from sklearn.metrics import mean_squared_error, r2_score
# fetch_california_housing is a utility to download and load the California housing dataset
from sklearn.datasets import fetch_california_housing

# Step 2: Load the Dataset
# Load the dataset as a pandas DataFrame for easier manipulation
data = fetch_california_housing(as_frame=True).frame

print(data.head())

# Step 4: Select Input Feature and Target Variable
# We use 'AveRooms' (Average Number of Rooms) as our single input feature (X)
# We predict 'MedHouseVal' (Median House Value) as our target (y)
X = data[["AveRooms"]]
y = data["MedHouseVal"]

# Step 5: Split the Dataset
# We split the data into 80% training data and 20% testing data. 
# random_state ensures reproducibility of the splits.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )


# ==============================================================================
# PART A - Linear Regression using Direct Least Squares Solution 
# ==============================================================================
print("Normal Equation:")
# Create the model using the Ordinary Least Squares method
lr_model = LinearRegression()

# Train the model using the training data
lr_model.fit(X_train, y_train)

# Predict on the test data
y_pred_lr = lr_model.predict(X_test)

# Display model parameters (Slope/Weight and Intercept/Bias)
print("Slope (Coefficient) :", lr_model.coef_[0])
print("Intercept :", lr_model.intercept_)

# Evaluate the model
print("\nModel Performance")
print("Mean Squared Error :", mean_squared_error(y_test, y_pred_lr))
print("R² Score :", r2_score(y_test, y_pred_lr))


# ==============================================================================
# PART B - Linear Regression using Gradient Descent 
# ==============================================================================
print("\nGradient Descent:")
# Create the SGD model. We specify max iterations, initial learning rate (eta0), 
# and a constant learning rate schedule for this experiment.
sgd_model = SGDRegressor( max_iter=5000, eta0=0.001, learning_rate="constant", random_state=42)

# Train the model iteratively using Gradient Descent
sgd_model.fit(X_train, y_train)

# Predict on the test data using the trained SGD model
y_pred_sgd = sgd_model.predict(X_test)

# Display model parameters (SGD attempts to approximate the least squares parameters)
print("Slope (Coefficient) :", sgd_model.coef_[0])
# Note: In older sklearn versions, intercept_ was an array, hence [0]. 
# We'll use [0] as provided in the PDF to match the code exactly.
print("Intercept :", sgd_model.intercept_[0])

# Evaluate the model 
print("\nModel Performance")
print("Mean Squared Error :", mean_squared_error(y_test, y_pred_sgd))
print("R² Score :", r2_score(y_test, y_pred_sgd))


# ==============================================================================
# PART C - Comparison of Both Models
# ==============================================================================
# We create a DataFrame to cleanly compare MSE and R2 scores side-by-side.
comparison = pd.DataFrame( { 
    "Metric": ["MSE", "R² Score"], 
    "LinearRegression": [ mean_squared_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr) ], 
    "SGDRegressor": [ mean_squared_error(y_test, y_pred_sgd), r2_score(y_test, y_pred_sgd ) ] 
} )

# ==============================================================================
# PART D - Visualization 
# ==============================================================================
# Sort test data for smooth regression lines. Without sorting, plotting lines 
# across scatter data points creates chaotic scribbles back and forth.
plot_data = X_test.copy()
plot_data["Actual"] = y_test.values
plot_data["LinearRegression"] = y_pred_lr
plot_data["SGDRegressor"] = y_pred_sgd
plot_data = plot_data.sort_values(by="AveRooms")

plt.figure(figsize=(8,6))

# Actual data
plt.scatter( plot_data["AveRooms"], plot_data["Actual"], alpha=0.3, label="Actual Data" )

# Direct Least Squares Solution (Red Line)
plt.plot( plot_data["AveRooms"], plot_data["LinearRegression"], color="red", linewidth=3, label="LinearRegression" )

# Gradient Descent Solution (Green Dashed Line)
plt.plot( plot_data["AveRooms"], plot_data["SGDRegressor"], color="green", linewidth=3, linestyle="--", label="SGDRegressor" )

# Setup graph titles and labels
plt.title("Linear Regression using Two Approaches")
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median House Value")

# Limit axes as shown in the PDF for better visualization of the dense area
plt.xlim(1,8)
plt.ylim(1,6)

plt.legend()
plt.grid(True)
plt.savefig("exp1_plot.png")

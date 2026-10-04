# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 2: POLYNOMIAL REGRESSION
# DATASET: AUTO MPG
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP02_POLYNOMIAL_REGRESSION_AUTO_MPG/polynomial_regression_auto_mpg.py
# 5. Review the console output for evaluation metrics and close the plot window to exit.
# 
# ==============================================================================

# Polynomial Regression on Auto MPG Dataset (Seaborn)
# Dataset : Auto MPG Dataset (Seaborn)
# Objective: Predict miles per gallon (MPG) based on engine displacement.

# Step 1: Import required libraries
import seaborn as sns # Seaborn provides a very simple way to load the 'mpg' dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline # Allows chaining steps (PolynomialFeatures -> LinearRegression)
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
# Seaborn has the auto-mpg dataset built-in. It will download it automatically if not cached.
df = sns.load_dataset("mpg")
print(df.head())

# Remove missing values
# Drops rows that contain any missing (NaN) values to ensure clean data for training
df.dropna(inplace=True)

# Select Feature and Target
# We want to see how engine displacement (X) affects fuel efficiency (y)
X = df[["displacement"]]
y = df["mpg"]

# Split dataset
# 80% used for training the model, 20% used for testing predictions
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# ==============================================================================
# Part A: Linear Regression
# ==============================================================================
# We first try a standard straight-line (degree 1) model as a baseline.
linear = LinearRegression()
linear.fit(X_train, y_train)

# Predict using test data
y_pred_linear = linear.predict(X_test)

# Evaluate model performance
print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R² :", r2_score(y_test, y_pred_linear))

# ==============================================================================
# Part B: Polynomial Regression
# ==============================================================================
# We will loop through degrees 2, 3, and 4 to see how higher order polynomials perform.
degrees = [2, 3, 4]
for d in degrees:
    # Create the model using a pipeline. 
    # This automatically transforms the input X into polynomial features of degree d,
    # and then passes those features into a Linear Regression model.
    model = make_pipeline(
        PolynomialFeatures(d),
        LinearRegression()
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model performance
    print(f"\nPolynomial Degree {d}")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² :", r2_score(y_test, y_pred))

# ==============================================================================
# Part C: Visualization
# ==============================================================================
plt.figure(figsize=(10, 6))

# Scatter plot of all the actual data
plt.scatter(X, y, color="blue", alpha=0.6, label="Actual Data")

# Smooth x values: Create 300 evenly spaced points between min and max X for a smooth curve
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Linear Regression Line (Degree 1)
# Note: In older pandas/sklearn, X_plot passing might raise a warning about missing feature names.
# To match the PDF exactly, we pass it directly.
plt.plot( X_plot, linear.predict(X_plot), color="red", label="Linear Regression")

# Polynomial Curves
colors = ["green", "orange", "purple"]
for d, color in zip(degrees, colors):
    # Recreate the pipeline to train on the complete dataset for drawing the final curve
    model = make_pipeline(
        PolynomialFeatures(d),
        LinearRegression()
    )
    
    # Train the model using the complete dataset (as per the PDF's approach for the graph)
    model.fit(X, y)
    
    # Draw the polynomial regression curve
    plt.plot( X_plot, model.predict(X_plot), color=color, label=f"Degree {d}" )

# Add Labels and Display the Graph
plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.savefig("exp2_plot.png")

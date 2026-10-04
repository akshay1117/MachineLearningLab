# Experiment 2: Polynomial Regression

## Aim

To implement Polynomial Regression on the Auto MPG dataset to predict fuel efficiency and compare it with simple linear regression.

## Task

Load the Auto MPG dataset, select a feature that has a non-linear relationship with the target (like Weight vs MPG), handle missing values, apply polynomial feature transformation, and train a regression model. Compare the results with a standard linear regression model.

## Theory

**Polynomial Regression** is a form of regression analysis in which the relationship between the independent variable `x` and the dependent variable `y` is modeled as an nth degree polynomial.

While simple linear regression fits a straight line (`y = b₀ + b₁x`), polynomial regression fits a curve.
For a polynomial of degree 2, the equation is:
`y = b₀ + b₁x + b₂x²`

**Why use it?**
Ordinary linear regression assumes a straight-line relationship. If the data is inherently curved (like car weight vs. MPG—heavier cars drastically reduce MPG up to a point, then the effect levels off), a straight line will **underfit** the data. By transforming the input features to include their squares, cubes, etc., a standard linear model can learn curved relationships.

## Implementation Plan

1. **Import Libraries:** Import Numpy, Seaborn, Matplotlib, and Scikit-learn.
2. **Load Dataset:** Load Auto MPG dataset and remove missing values using `dropna()`.
3. **Select Features:** Set `displacement` as independent variable and `mpg` as target.
4. **Split Dataset:** Split into training and testing sets.
5. **Train Linear Model:** Create and train a linear regression model.
6. **Predict (Linear):** Predict MPG values for the test data.
7. **Evaluate (Linear):** Evaluate the linear model using MSE and R² score.
8. **Create Polynomial Models:** Create polynomial regression models for degrees 2, 3, and 4.
9. **Train Polynomial Models:** Train each polynomial model using training data.
10. **Predict (Polynomial):** Predict MPG values for testing data using each model.
11. **Evaluate (Polynomial):** Calculate MSE and R² Score for each polynomial model.
12. **Compare Models:** Compare performance between linear and polynomial regression.
13. **Generate Smooth Values:** Generate smooth displacement values for curve plotting.
14. **Visualize Results:** Plot actual MPG values alongside linear and polynomial curves.
15. **Interpret Results:** Identify which model balances prediction error and explained variance best.

## Dataset

- **Name:** Auto MPG Dataset
- **Source:** UCI Machine Learning Repository
- **Target:** `mpg` (Miles per gallon)
- **Feature Selected:** `weight` (Car weight in lbs).
- **Obtaining it:** The Python script is written to automatically download the dataset from the UCI URL and save it as `data/auto-mpg.csv` if it doesn't already exist locally.

## Algorithm

- `sklearn.preprocessing.PolynomialFeatures` is used to create polynomial terms (e.g., `x²`).
- `sklearn.linear_model.LinearRegression` is used to fit the model. *Note that polynomial regression is just linear regression applied to polynomial features!*

## Expected Output

When you run the program, the console will print:

- The dimensions of the dataset before and after cleaning.
- The MSE and R² score for standard Linear Regression.
- The MAE, MSE, RMSE, and R² score for Polynomial Regression.
- You will see that Polynomial Regression generally yields a higher R² score.
- A visualization window showing the actual data points, the straight linear regression line (blue), and the curved polynomial regression line (red).

## Viva Questions

1. **Is Polynomial Regression a linear or non-linear model?**
   *It is a linear model. Although it fits a non-linear curve to the data, the model itself is linear with respect to its coefficients (b₀, b₁, b₂).*
2. **What does `degree` mean in polynomial regression?**
   *The degree is the maximum power of the independent variable. Degree 2 creates `x²`, degree 3 creates `x²` and `x³`, etc.*
3. **What happens if the degree is set too high (e.g., degree=15)?**
   *The model will suffer from **overfitting**. It will pass exactly through the training points but perform terribly on new test data, resulting in a wildly fluctuating curve.*
4. **Why did we sort the test data before plotting?**
   *If we plot a line between unsorted points, matplotlib will draw zigzag lines all over the plot connecting points in their random order. Sorting `X` ensures the line draws smoothly from left to right.*
5. **How does Polynomial Regression improve the Auto MPG prediction?**
   *The relationship between car weight and MPG is non-linear (an inverse curve). Polynomial regression captures this curve, reducing the error (MSE) compared to a straight line.*

## Key Learning Points

- Standard linear models can learn non-linear patterns if we apply feature transformations.
- Comparing models using R² and MSE.
- The concept of underfitting (when a straight line fails to capture a curved trend).

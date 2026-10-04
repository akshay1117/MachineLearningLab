# Experiment 1: Linear Regression with One Variable

## Aim

To implement Simple (Univariate) Linear Regression on the California Housing dataset to predict house values based on a single feature.

## Task

Select a single independent variable from the California Housing dataset and use it to build a linear regression model that predicts the median house value. Evaluate the model and visualize the regression line.

## Theory

**Linear Regression** is a supervised machine learning algorithm used to predict a continuous numerical output. It assumes a linear relationship between the independent variable (X) and the dependent variable (y).

**Simple Linear Regression** uses only *one* independent variable.
The equation of the line is:
`y = b₀ + b₁x`

Where:

- `y`: The predicted target variable.
- `x`: The input feature.
- `b₀`: The y-intercept (value of `y` when `x` is 0).
- `b₁`: The slope or coefficient (how much `y` changes for a 1-unit change in `x`).

The model learns by minimizing the differences between the predicted values and the actual values. This is typically done using the Ordinary Least Squares (OLS) method, which minimizes the Mean Squared Error (MSE).

## Implementation Plan

1. **Import Libraries:** Import Pandas, Matplotlib, and Scikit-learn.
2. **Load Dataset:** Use `fetch_california_housing()` and convert to a Pandas DataFrame.
3. **Explore Data:** Display records, dimensions, column names, and summary statistics.
4. **Select Features:** Set `AveRooms` as input (X) and `MedHouseVal` as target (y).
5. **Split Dataset:** Divide into 80% training and 20% testing sets using `train_test_split()`.
6. **Train Normal Model:** Create and train a `LinearRegression` model using the training set.
7. **Predict (Normal):** Predict house prices for the testing dataset.
8. **Display Parameters:** Output the slope and intercept of the regression line.
9. **Evaluate (Normal):** Compute Mean Squared Error (MSE) and R² score.
10. **Train Gradient Descent Model:** Create and train an `SGDRegressor` using the training data.
11. **Predict (Gradient Descent):** Predict house prices using the SGD model.
12. **Evaluate (Gradient Descent):** Calculate MSE and R² Score for the SGD model.
13. **Compare Models:** Compare the evaluation metrics of both models.
14. **Visualize Results:** Plot actual data points and regression lines for both approaches.
15. **Interpret Results:** Compare predicted values, parameters, and metrics to understand the effectiveness of the two approaches.

## Dataset

- **Name:** California Housing Dataset
- **Source:** Built-in `sklearn` datasets (`fetch_california_housing`).
- **Target:** `MedHouseVal` (Median house value for California districts in hundreds of thousands of dollars).
- **Feature Selected:** `MedInc` (Median income in block group). We selected this because income generally has a strong linear correlation with house prices.
- **Obtaining it:** It downloads automatically when you run the code.

## Algorithm

The code uses `sklearn.linear_model.LinearRegression`. It calculates the optimal slope and intercept using Ordinary Least Squares underneath.

## Expected Output

When you run the program, the console will print:

- The shape of the dataset.
- The calculated model equation (Intercept and Slope).
- Evaluation metrics (MAE, MSE, RMSE, R²).
- A matplotlib window will open showing a scatter plot of the data points in blue and the predicted straight regression line in red.

## Viva Questions

1. **What is the difference between simple and multiple linear regression?**
   *Simple linear regression uses one independent variable. Multiple linear regression uses two or more.*
2. **What does the R² (R-squared) score indicate?**
   *It indicates the proportion of the variance in the dependent variable that is predictable from the independent variable. A score of 1.0 means perfect prediction.*
3. **Why do we split the dataset into train and test sets?**
   *To evaluate the model's performance on unseen data, which helps verify that the model hasn't just memorized the training data (overfitting).*
4. **What is Mean Squared Error (MSE)?**
   *It is the average of the squared differences between the predicted and actual values.*
5. **What is the significance of the slope (`b₁`) in our equation?**
   *It shows how much the house value increases for every one-unit increase in median income.*

## Key Learning Points

- How to extract a single column to perform univariate regression.
- How to interpret the mathematical equation of the trained model.
- How to visualize a best-fit regression line over a scatter plot.

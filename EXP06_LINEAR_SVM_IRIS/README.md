# Experiment 6: Linear Support Vector Machine

## Aim

To implement a Linear Support Vector Machine (SVM) classifier on the Iris dataset and visualize the linear decision boundaries.

## Task

Load the Iris dataset, select two features (for 2D visualization), scale the features, train a linear SVM, evaluate its accuracy, and plot the 2D decision boundary showing the different classes.

## Theory

**Support Vector Machine (SVM)** is a powerful supervised algorithm used for both classification and regression, but mostly for classification.

**How it works:**

1. **Hyperplane:** SVM aims to find a line (or in higher dimensions, a "hyperplane") that best separates the classes in the dataset.
2. **Margin:** It's not just about finding *any* line that separates the data. SVM tries to find the line that maximizes the **margin**—the distance between the line and the nearest data points of any class.
3. **Support Vectors:** The data points that lie closest to the decision boundary (the ones dictating the margin) are called support vectors. If you remove all other data points, the position of the dividing line would not change.
4. **Linear Kernel:** A linear kernel implies we are drawing a strictly straight line (or flat plane) to separate the data. If the data is not linearly separable, other kernels (like RBF or Poly) can be used to warp the space.

## Implementation Plan

1. **Import Libraries:** Import Numpy, Pandas, Matplotlib, and Scikit-learn.
2. **Load Dataset:** Use `load_iris()` to load the dataset.
3. **Convert Problem:** Convert the 3-class problem into a binary classification (Setosa vs Non-Setosa).
4. **Select Features:** Select Sepal Length and Sepal Width as the two feature dimensions.
5. **Split Dataset:** Split the data into training and testing sets.
6. **Standardize Data:** Standardize the features using `StandardScaler()`.
7. **Train SVM:** Create and train an SVM classifier using `SVC(kernel='linear', C=1.0)`.
8. **Predict:** Predict the classes for the testing dataset.
9. **Evaluate:** Calculate accuracy, confusion matrix, and classification report.
10. **Visualize Boundary:** Visualize decision boundary (solid lines), SVM margins (dashed lines), and support vectors (circled points).
11. **Analyze Margin:** Analyze how the support vectors determine the margin.

## Dataset

- **Name:** Iris Dataset
- **Source:** Built-in `sklearn` datasets (`load_iris`).
- **Target:** 3 species of Iris flowers (Setosa, Versicolour, Virginica).
- **Features Selected:** Sepal Length, Sepal Width (we ignored petal dimensions for visualization purposes).
- **Obtaining it:** It downloads automatically when you run the code.

## Algorithm

- `sklearn.svm.SVC(kernel='linear')`

## Expected Output

When you run the program, the console will output:

- Dataset and subset shapes.
- Accuracy of the model.
- Classification Report.
- A matplotlib window showing:
  - A Confusion Matrix.
  - A beautiful 2D Decision Boundary plot where the background color represents the model's classification zones, and the dots represent the actual flower samples.

## Viva Questions

1. **What is a Support Vector?**
   *Support vectors are the specific data points that lie closest to the decision boundary. They are the most difficult to classify and essentially define the margin.*
2. **Why is feature scaling crucial for SVM?**
   *SVM tries to maximize the geometric distance between classes. If one feature is measured in thousands and another in decimals, the larger feature will completely dominate the distance calculation, making the SVM ineffective.*
3. **What is a Hyperplane?**
   *A hyperplane is a decision boundary. In 2D, it is a line. In 3D, it is a flat plane. In n-dimensions, it is an (n-1) dimensional flat subspace.*
4. **What does the 'linear' kernel do?**
   *It attempts to separate the data using a perfectly straight line/plane without mathematically transforming the input space into higher dimensions.*
5. **What is a "Hard Margin" vs "Soft Margin"?**
   *A hard margin allows absolutely no misclassifications in the training data (which can lead to overfitting). A soft margin allows some misclassifications or points inside the margin to create a more generalized boundary. This tradeoff is controlled by the `C` parameter in sklearn.*

## Key Learning Points

- How SVM works by maximizing the margin.
- The absolute necessity of standardizing data for distance/margin-based algorithms.
- How to visualize a classification boundary using `numpy.meshgrid`.

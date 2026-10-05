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

## Viva Questions & Answers
1. **What is the main objective of a Support Vector Machine (SVM)?**
   **Answer:** To find the optimal hyperplane in an N-dimensional space that maximizes the margin (distance) between different classes, ensuring the best possible separation.
2. **What are Support Vectors?**
   **Answer:** Support vectors are the data points that lie closest to the decision surface (hyperplane). They are the most critical points because they dictate the position and orientation of the hyperplane.
3. **What is a hyperplane and what is a margin in SVM?**
   **Answer:** A hyperplane is the decision boundary separating classes. The margin is the distance between the hyperplane and the closest data points (support vectors) from either class. SVM aims to maximize this margin.
4. **How does the 'C' regularization parameter affect the SVM model?**
   **Answer:** 'C' dictates the trade-off between maximizing the margin and minimizing classification errors. A high 'C' strictly penalizes misclassifications (smaller margin, potential overfitting). A low 'C' allows a wider margin with some misclassifications (potential underfitting).
5. **What is the Kernel Trick and when is it used?**
   **Answer:** The Kernel Trick is a mathematical technique used to implicitly map non-linearly separable data into a higher-dimensional space where it becomes linearly separable, without explicitly computing the coordinates in that space.

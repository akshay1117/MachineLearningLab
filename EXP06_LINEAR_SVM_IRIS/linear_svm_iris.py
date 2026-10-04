# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 6: SUPPORT VECTOR MACHINE (LINEAR SVM)
# DATASET: IRIS
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP06_LINEAR_SVM_IRIS/linear_svm_iris.py
# 5. Review the console output for evaluation metrics and close plot windows to exit.
# 
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ==============================================================================
# 1. Load and preprocess the dataset
# ==============================================================================
# The Iris dataset is built into sklearn
iris = datasets.load_iris()

# For a 2D visualization, we only select the first two features:
# Sepal Length and Sepal Width
X = iris.data[:, :2] 
y = iris.target
feature_names = iris.feature_names[:2]

# Binary classification: Setosa (0) vs. Non-Setosa (1)
# The original dataset has 3 classes. We convert it to a binary problem.
# If class is 0 (Setosa), keep it 0. Otherwise (Versicolor/Virginica), make it 1.
y_binary = np.where(y == 0, 0, 1)
target_names = ['Setosa', 'Non-Setosa']

# Standardize the features
# SVMs are sensitive to feature scales, so we normalize them to have mean=0 and variance=1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# ==============================================================================
# 2. Train the Linear SVM
# ==============================================================================
# kernel="linear": We want a straight line (hyperplane) to separate the classes.
# C=1.0: Regularization parameter. Controls the trade-off between maximizing the margin 
# and minimizing classification errors.
svm_clf = SVC(kernel="linear", C=1.0)
svm_clf.fit(X_train, y_train)


# ==============================================================================
# 3. Generate Predictions for Metrics
# ==============================================================================
y_pred = svm_clf.predict(X_test)


# ==============================================================================
# 4. PRINT FINDINGS & ANALYTICS IN CONSOLE
# ==============================================================================
print("--- SVM CLASSIFICATION FINDINGS ---")

# A. Performance Metrics
print("\n1. Classification Report (on Test Data):")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# B. Support Vector Analysis
# Support Vectors are the data points closest to the hyperplane that dictate its position.
num_sv = len(svm_clf.support_vectors_)
total_samples = len(X_train)

print(f"\n2. Support Vector Sparsity:")
print(f"Total Support Vectors: {num_sv} out of {total_samples} samples")
print(f"Class 0 (Setosa) SVs: {svm_clf.n_support_[0]}")
print(f"Class 1 (Non-Setosa) SVs: {svm_clf.n_support_[1]}")
print(f"Percentage of dataset used as SVs: {(num_sv/total_samples)*100:.2f}%")

# C. Feature Importance (Weights)
# The weights represent the orthogonal vector to the decision boundary.
weights = svm_clf.coef_[0]
print(f"\n3. Feature Weights (w):")
print(f"{feature_names[0]} Weight: {weights[0]:.4f}")
print(f"{feature_names[1]} Weight: {weights[1]:.4f}")

# Providing a mathematical insight based on the weights calculated
if weights[0] > 0 and weights[1] < 0:
    print("Insight: Non-Setosa leans towards higher Sepal Lengths and lower Sepal Widths.")


# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    """Plots the decision boundary, the margins, and highlights the support vectors."""
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    
    # Generate x coordinates
    x0 = np.linspace(xmin, xmax, 200)
    
    # Equation of the decision boundary: w0*x0 + w1*x1 + b = 0 => x1 = -w0/w1 * x0 - b/w1
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    
    # Equation of the margins: w0*x0 + w1*x1 + b = +/- 1 => x1 = -w0/w1 * x0 - b/w1 +/- 1/w1
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    
    # Highlight the Support Vectors (data points resting exactly on the margins)
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA', label="Support Vectors")
    
    # Plot the lines
    plt.plot(x0, decision_boundary, "k-", linewidth=2, label="Decision Boundary")
    plt.plot(x0, gutter_up, "k--", linewidth=2, label="Margin")
    plt.plot(x0, gutter_down, "k--", linewidth=2)

plt.figure(figsize=(10, 6))

# Plot the data points
# "bo" = blue circles (Non-Setosa)
plt.plot(X_scaled[:, 0][y_binary==1], X_scaled[:, 1][y_binary==1], "bo", label="Non-Setosa")
# "yo" = yellow circles (Setosa)
plt.plot(X_scaled[:, 0][y_binary==0], X_scaled[:, 1][y_binary==0], "yo", label="Setosa")

# Plot the decision boundaries
plot_svc_decision_boundary(svm_clf, X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5)

# Labeling
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.title("Linear SVM on Iris Dataset: Setosa vs. Non-Setosa")
plt.legend(loc="upper left")

# Set axis limits
plt.axis([
    X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 
    X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
])

plt.savefig("exp6_svm_plot.png")

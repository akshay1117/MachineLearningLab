# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 7: MULTILAYER FEED-FORWARD NEURAL NETWORK
# DATASET: WINE QUALITY
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP07_MLP_WINE_QUALITY/mlp_wine_quality.py
# 5. Review the console output and close the plot windows to proceed.
# 
# ==============================================================================

# Wine Quality Classification using Multilayer Feed-Forward NN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# ==============================================================================
# 1. Load the dataset
# ==============================================================================
# We load the dataset directly from the UCI ML Repository URL.
# The wine quality dataset uses semicolons ';' instead of commas to separate values.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

print("Dataset shape:", data.shape)
print("\nFirst five rows:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())
print("\nTarget distribution:\n", data["quality"].value_counts().sort_index())


# ==============================================================================
# 2. Separate features and target
# ==============================================================================
# X contains all the chemical properties of the wine.
X = data.drop("quality", axis=1)
# y is the quality score (from 3 to 8).
y = data["quality"]


# ==============================================================================
# 3. Train-test split
# ==============================================================================
# 80% for training, 20% for testing. stratify=y preserves the class distribution.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("\nTraining samples:", X_train.shape[0], "| Testing samples:", X_test.shape[0])


# ==============================================================================
# 4. Feature scaling
# ==============================================================================
# Neural networks require features to be on a similar scale to learn efficiently.
# StandardScaler centers data around a mean of 0 with a standard deviation of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================================================================
# 5. Design MLP architectures
# ==============================================================================
# We define a dictionary of different neural network architectures to test.
# The tuples define (neurons_in_layer_1, neurons_in_layer_2, ...)
architectures = {
    "1 Layer - 16": (16,), 
    "1 Layer - 32": (32,), 
    "1 Layer - 64": (64,),
    "2 Layers - 32,16": (32, 16), 
    "2 Layers - 64,32": (64, 32),
    "3 Layers - 64,32,16": (64, 32, 16)
}


# ==============================================================================
# 6. Train and evaluate each architecture
# ==============================================================================
results = []
models = {}

for name, hidden_layers in architectures.items():
    print("\n" + "=" * 60, "\nArchitecture:", name, "\n" + "=" * 60)
    
    # Initialize the Multi-Layer Perceptron (MLP) Classifier
    # activation="relu": Rectified Linear Unit, standard for hidden layers
    # solver="adam": Efficient stochastic gradient descent optimizer
    # early_stopping: Stops training when validation score stops improving to prevent overfitting
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers, 
        activation="relu", 
        solver="adam", 
        max_iter=1000,
        early_stopping=True, 
        validation_fraction=0.15, 
        n_iter_no_change=30, 
        random_state=42
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate performance
    # zero_division=0 prevents warnings when a class is never predicted
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Store the results for comparison
    results.append({
        "Architecture": name, 
        "Hidden Layers": len(hidden_layers), 
        "Neurons": str(hidden_layers),
        "Accuracy": accuracy, 
        "Precision": precision, 
        "Recall": recall, 
        "F1 Score": f1, 
        "Iterations": model.n_iter_
    })
    models[name] = model
    
    print(f"Accuracy : {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | Iterations: {model.n_iter_}")


# ==============================================================================
# 7. Compare all architectures
# ==============================================================================
results_df = pd.DataFrame(results)
print("\n\nArchitecture Comparison\n" + "=" * 100)
# Sort by F1 Score to easily see the best performing architecture
print(results_df[["Architecture", "Hidden Layers", "Accuracy", "Precision", "Recall", "F1 Score", "Iterations"]].sort_values("F1 Score", ascending=False).to_string(index=False))


# ==============================================================================
# 8. Combined comparison grouped by number of hidden layers
# ==============================================================================
# This isolates the effect of depth (number of layers) from neuron count
grouped_df = results_df.groupby("Hidden Layers")[["Accuracy", "Precision", "Recall", "F1 Score"]].mean().round(4).reset_index()
print("\nCombined Comparison by Number of Hidden Layers (averaged)\n" + "=" * 100)
print(grouped_df.to_string(index=False))


# ==============================================================================
# 9. Select the best model
# ==============================================================================
best_architecture = results_df.loc[results_df["F1 Score"].idxmax(), "Architecture"]
best_model = models[best_architecture]
print("\nBest architecture:", best_architecture)


# ==============================================================================
# 10. Detailed classification report
# ==============================================================================
y_best_pred = best_model.predict(X_test_scaled)
print("\nClassification Report\n" + "=" * 60)
print(classification_report(y_test, y_best_pred, zero_division=0))


# ==============================================================================
# 11. Confusion matrix
# ==============================================================================
cm = confusion_matrix(y_test, y_best_pred)
classes = sorted(y.unique())

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix - Best MLP")
plt.colorbar()

# Set axis ticks and labels
plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel("Predicted Quality")
plt.ylabel("Actual Quality")

# Add numerical values inside the matrix squares
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("exp7_confusion_matrix.png")


# ==============================================================================
# 12. Performance vs. network depth (combined view)
# ==============================================================================
plt.figure(figsize=(8, 6))
plt.plot(grouped_df["Hidden Layers"], grouped_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(grouped_df["Hidden Layers"], grouped_df["F1 Score"], marker="s", label="F1 Score")
plt.xticks(grouped_df["Hidden Layers"])
plt.ylabel("Score")
plt.xlabel("Number of Hidden Layers")
plt.title("MLP Performance vs Network Depth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("exp7_performance_depth.png")


# ==============================================================================
# Discussion & Conclusion
# ==============================================================================
print("""
--- Discussion: Effect of Architecture on Performance ---
Width: more neurons per layer (16->32->64) improves fit up to a point, then gains plateau and only
training time increases.
Depth: extra hidden layers can model hierarchical patterns but add parameters faster than this small
dataset (~1,600 rows) can constrain, raising overfitting risk and iterations needed.
Class imbalance: quality scores cluster around 5-6 with few samples at 3/8, capping achievable
accuracy/F1 for every architecture equally.

CONCLUSION: A single moderate-width hidden layer (32-64 neurons) gives the best accuracy/stability
trade-off; deeper networks don't consistently outperform it here.
""")

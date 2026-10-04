# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 4: K-NEAREST NEIGHBORS (KNN)
# DATASET: FASHION MNIST
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP04_KNN_FASHION_MNIST/knn_fashion_mnist.py
# 5. Review the console output for evaluation metrics and close the plot windows to exit.
# 
# ==============================================================================

# K-Nearest Neighbors (KNN) on Fashion MNIST Dataset
# Classify clothing images into 10 categories using KNN,
# and analyze the impact of different K values on accuracy and computational efficiency.

# Step 1: Import Required Libraries
import numpy as np, pandas as pd, gzip, urllib.request, struct, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 2: Load the Fashion MNIST Dataset
# Both images and labels are stored in the same gzip-compressed binary format,
# so one helper function handles both by taking a different header size/shape
base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
print("Loading Fashion-MNIST dataset...")

def load_gz(url, header_len, shape=None):
    """Downloads and unzips a binary file directly into memory, skipping disk saves."""
    with urllib.request.urlopen(url, timeout=120) as response:
        data = gzip.decompress(response.read())
        # Convert byte data to NumPy array, skipping the specified header bytes
        arr = np.frombuffer(data[header_len:], dtype=np.uint8)
        return arr.reshape(shape) if shape else arr

# Load training images and reshape them into 3D array (count, height, width)
X_images = load_gz(base_url + "train-images-idx3-ubyte.gz", header_len=16, shape=(-1, 28, 28))
# Load labels (1D array)
y = load_gz(base_url + "train-labels-idx1-ubyte.gz", header_len=8)

# The 10 categories for clothing items
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("Dataset loaded successfully!\n")
print("Dataset Information")
print("--------------------")
print(f"Number of images: {len(X_images)}")
print("Image size: 28 x 28")
print(f"Number of classes: {len(class_names)}\n")
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i} : {name}")
print("\nDisplaying sample images...\n")

# Step 3: Flatten each 28x28 image into a 784-dimensional feature vector
# Standard Machine Learning models (unlike CNNs) require 1D input features.
X = X_images.reshape(len(X_images), -1)

# Step 4: Normalize pixel values to [0, 1] by dividing by 255
# Pixel intensity is between 0 and 255. Scaling improves distance calculations in KNN.
X = X / 255.0

# Step 5: Reduce dataset size to 10,000 samples for faster computation 
# (KNN is computationally expensive because it calculates distance to all training samples)
X = X[:10000]
y = y[:10000]
print("Samples used for KNN: 10000")
print("Features per image: 784")

# Step 6: Split the dataset into training and testing sets
# stratify=y ensures the same proportion of classes in both train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)

# ==============================================================================
# PART A - Train and Evaluate KNN for Different Values of K
# ==============================================================================

# Step 7: Implement the KNN classifier using sklearn's KNeighborsClassifier
# Step 8 & 9: Train/evaluate for each K, recording accuracy, precision, recall, F1, and prediction time
def evaluate_k(k):
    # Train the model with 'k' neighbors
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    
    # Track how long it takes to make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    # We use average="weighted" since this is a multiclass classification problem
    return {
        "K": k,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        "Prediction Time (s)": pred_time
    }

# We will test odd numbers for K to prevent ties in voting
k_values = [1, 3, 5, 7, 9, 11]
comparison_df = pd.DataFrame([evaluate_k(k) for k in k_values])

print("\n--- KNN Performance Comparison ---")
print(comparison_df.round(4).to_string(index=True))

# Find the best K based on accuracy
best_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
best_k = int(best_row["K"])
print("\n--- Best Result ---")
print(f"Best K: {best_k}")
print(f"Best Accuracy: {round(best_row['Accuracy'], 3)}")


# ==============================================================================
# PART B - Visualization
# ==============================================================================

# Step 10: Plot Accuracy vs. K to visualize the impact of K on performance
# Step 11: Plot Prediction Time vs. K to visualize the impact of K on efficiency
def plot_metric(y_col, color, ylabel):
    plt.figure(figsize=(7, 5))
    plt.plot(comparison_df["K"], comparison_df[y_col], marker="o", color=color)
    plt.title(f"{ylabel} vs. K")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.savefig(f"exp4_{ylabel.split()[0].lower()}_vs_k.png")

# You must close each graph window to proceed to the next block of code!
plot_metric("Accuracy", "steelblue", "Accuracy")
plot_metric("Prediction Time (s)", "darkorange", "Prediction Time (seconds)")


# ==============================================================================
# PART C - Test the Best Model on a Sample Image
# ==============================================================================

# Step 12: Use the best K to make a single prediction and visually confirm the result
best_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)

# Select a random image from the test set
sample_index = 10
sample_image = X_test[sample_index]
actual_label = y_test[sample_index]

# Make prediction. reshape(1, -1) is used because predict expects a 2D array (batch of inputs)
predicted_label = best_model.predict(sample_image.reshape(1, -1))[0]

print("\n--- Sample Prediction ---")
print(f"Sample Index: {sample_index}")
print(f"Actual Label: {actual_label}")
print(f"Actual Class: {class_names[actual_label]}")
print(f"Predicted Label: {predicted_label}")
print(f"Predicted Class: {class_names[predicted_label]}")
print("Result:", "CORRECT" if actual_label == predicted_label else "INCORRECT")

# Display the image
plt.figure(figsize=(4, 4))
plt.imshow(sample_image.reshape(28, 28), cmap="gray")
plt.title(f"Actual: {class_names[actual_label]}\nPredicted: {class_names[predicted_label]}")
plt.axis("off")
plt.savefig("exp4_sample_prediction.png")

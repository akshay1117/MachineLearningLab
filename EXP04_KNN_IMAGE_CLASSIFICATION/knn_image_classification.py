# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 4: KNN FOR IMAGE CLASSIFICATION
# DATASET: FASHION-MNIST
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP04_KNN_IMAGE_CLASSIFICATION/knn_image_classification.py
# 5. Review the console output for evaluation metrics and close the plot window to exit.
# 
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def main():
    print("==================================================")
    print("EXPERIMENT 4: KNN FOR IMAGE CLASSIFICATION")
    print("==================================================\n")

    # ---------------------------------------------------------
    # STEP 1: Load the Dataset
    # ---------------------------------------------------------
    print("Loading Fashion-MNIST dataset (this may take a moment on first run)...")
    # We use Keras to download the dataset easily. 
    # It provides official pre-split train and test sets.
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Define the class names corresponding to the numerical labels (0-9)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"Original Training images shape: {X_train_full.shape}")
    print(f"Original Testing images shape: {X_test_full.shape}")

    # ---------------------------------------------------------
    # STEP 2: Subset the Data (For Educational Speed)
    # ---------------------------------------------------------
    # KNN is a "lazy learner". It computes the Euclidean distance between 
    # the test image and EVERY single training image.
    # Running it on 60,000 training images is computationally expensive and slow.
    # For educational purposes, we take a smaller random subset.
    
    train_subset_size = 5000
    test_subset_size = 1000

    X_train = X_train_full[:train_subset_size]
    y_train = y_train_full[:train_subset_size]
    X_test = X_test_full[:test_subset_size]
    y_test = y_test_full[:test_subset_size]
    
    print(f"\nUsing a subset for speed:")
    print(f"Training subset: {X_train.shape[0]} images")
    print(f"Testing subset: {X_test.shape[0]} images")

    # ---------------------------------------------------------
    # STEP 3: Data Preprocessing (Flattening & Scaling)
    # ---------------------------------------------------------
    # Currently, each image is a 28x28 2D matrix of pixels.
    # Traditional ML algorithms (like KNN) require 1D feature vectors.
    # We flatten each 28x28 image into a 1D array of 784 pixels.
    X_train_flat = X_train.reshape(X_train.shape[0], 28 * 28)
    X_test_flat = X_test.reshape(X_test.shape[0], 28 * 28)
    
    # Pixel values range from 0 to 255.
    # We scale them to a range of 0.0 to 1.0. 
    # Normalization prevents large pixel values from dominating the distance metric.
    X_train_scaled = X_train_flat / 255.0
    X_test_scaled = X_test_flat / 255.0

    print(f"\nFlattened Training shape: {X_train_scaled.shape}")

    # ---------------------------------------------------------
    # STEP 4: Train the KNN Model
    # ---------------------------------------------------------
    k = 5  # Number of nearest neighbors to consider
    print(f"\nTraining K-Nearest Neighbors Classifier (K={k})...")
    
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # "Training" in KNN just means storing the data in memory.
    knn.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    # STEP 5: Make Predictions
    # ---------------------------------------------------------
    print("Making predictions on the test subset (Computing distances)...")
    y_pred = knn.predict(X_test_scaled)

    # ---------------------------------------------------------
    # STEP 6: Evaluate the Model
    # ---------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test subset: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    # We use class_names to make the report human-readable
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ---------------------------------------------------------
    # STEP 7: Visualizations
    # ---------------------------------------------------------
    print("\nGenerating visualizations. Please close the plot window to exit.")
    
    fig = plt.figure(figsize=(12, 10))
    
    # ---- Plot 1: Confusion Matrix ----
    ax1 = fig.add_subplot(2, 1, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('Actual Category')
    ax1.set_xlabel('Predicted Category')
    
    # ---- Plot 2: Sample Predictions ----
    # Let's display 5 random test images along with their predictions
    num_samples = 5
    random_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(random_indices):
        # We create subplots for the images in the second row
        ax = fig.add_subplot(2, num_samples, num_samples + i + 1)
        
        # Display the original 28x28 image (unflattened)
        ax.imshow(X_test[idx], cmap='gray')
        
        actual_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        
        # Color title green if correct, red if wrong
        color = 'green' if actual_label == pred_label else 'red'
        ax.set_title(f"True: {actual_label}\nPred: {pred_label}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

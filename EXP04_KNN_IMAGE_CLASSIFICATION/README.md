# Experiment 4: KNN for Image Classification

## Aim

To implement the K-Nearest Neighbors (KNN) algorithm to classify images of clothing from the Fashion-MNIST dataset.

## Task

Load the Fashion-MNIST image dataset, flatten the 2D images into 1D feature vectors, normalize the pixel values, and train a KNN classifier. Evaluate the classifier using accuracy and visualize sample predictions. Because KNN is computationally heavy, use a subset of the data to keep execution times reasonable.

## Theory

**K-Nearest Neighbors (KNN)** is a supervised, distance-based classification algorithm.

**How it works:**

1. **Lazy Learning:** KNN doesn't actually "learn" a model during training. It simply stores all the training data in memory.
2. **Distance Calculation:** When asked to predict a new test image, the algorithm calculates the distance (usually **Euclidean Distance**) between the new image and *every single image* in the training set.
3. **Voting:** It finds the `K` training images that are closest to the test image (the "nearest neighbors").
4. **Prediction:** It assigns the test image to the class that is most common among those `K` neighbors.

**Why Images Must Be Flattened:**
A traditional ML algorithm like KNN expects tabular data (rows and columns). An image is a 2D grid of pixels. To feed an image into KNN, we "flatten" the 28x28 grid into a single row (vector) of 784 numbers. Each pixel becomes a "feature".

## Implementation Plan

1. **Import Libraries:** Import Numpy, Pandas, Matplotlib, TensorFlow, Scikit-learn, and Time.
2. **Load Dataset:** Load Fashion MNIST using `load_data()`.
3. **Reshape Data:** Convert 28x28 images into 784-dimensional vectors.
4. **Normalize Data:** Normalize pixel values to 0-1 by dividing by 255.
5. **Reduce Dataset:** Reduce to 10,000 training and 2,000 test samples.
6. **Define K Values:** Set K=1, 3, 5, 7, 9 for experimentation.
7. **Train Models:** Create and train a KNN classifier for each K value.
8. **Predict & Evaluate:** Predict classes for test data and calculate accuracy.
9. **Measure Time:** Record execution time for training and prediction for each K.
10. **Compare Accuracy:** Compare the accuracy achieved by each K value.
11. **Train Final Model:** Train the final KNN model using the best K value.
12. **Evaluate Final Model:** Evaluate using accuracy, classification report, and confusion matrix.
13. **Visualize Results:** Plot test results.
14. **Analyze Results:** Analyze how K value affects accuracy and computational efficiency.

## Dataset

- **Name:** Fashion-MNIST
- **Source:** Zalando Research (Accessed via `tensorflow.keras.datasets`).
- **Target:** 10 categories of clothing (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Features:** 28x28 grayscale images (784 pixels).
- **Obtaining it:** Downloads automatically on first run via the Keras API.

## Algorithm

- `sklearn.neighbors.KNeighborsClassifier` using the Euclidean distance metric.

## Expected Output

When you run the program, the console will output:

- Original dataset shapes.
- Subset dataset shapes.
- The flattened data shape.
- Overall accuracy (usually around 80-85% for `K=5` on this subset).
- A detailed classification report showing precision, recall, and f1-score for each clothing category.
- A visualization window showing a confusion matrix on top, and 5 actual clothing images with their predicted text underneath.

## Viva Questions & Answers
1. **How does the K-Nearest Neighbors (KNN) algorithm work?**
   **Answer:** KNN classifies a new data point based on the majority class among its 'K' closest neighbors in the feature space, calculated using a distance metric.
2. **What is the significance of the value 'K' in KNN?**
   **Answer:** 'K' determines how many neighbors influence the classification. It controls the bias-variance tradeoff: a small K is noisy and prone to overfitting, while a large K smooths decision boundaries but might underfit.
3. **What happens if K is too small or too large?**
   **Answer:** If K is too small (e.g., K=1), the model overfits and is sensitive to noise. If K is too large, the model underfits and the majority class in the entire dataset may dominate predictions.
4. **Why is feature scaling (normalization/standardization) extremely important for KNN?**
   **Answer:** Because KNN relies on distance metrics (like Euclidean distance). Features with larger numerical ranges will dominate the distance calculation if scaling is not applied.
5. **What distance metrics can be used in KNN?**
   **Answer:** Common distance metrics include Euclidean distance (L2 norm), Manhattan distance (L1 norm), and Minkowski distance.

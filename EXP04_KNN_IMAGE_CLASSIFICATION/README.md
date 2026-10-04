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

## Viva Questions

1. **Why is KNN called a "lazy" learner?**
   *Because it does not build a mathematical model during the training phase. It simply stores the data and defers all the heavy computation until prediction time.*
2. **Why do we normalize the image pixel values by dividing by 255?**
   *To scale features between 0 and 1. Distance metrics are highly sensitive to the scale of features. Though all pixels share the same scale (0-255), standardizing to [0,1] is a standard best practice in ML that prevents numeric overflow and speeds up calculation.*
3. **How is the distance between two images calculated in KNN?**
   *The images are flattened into vectors. The algorithm then calculates the Euclidean distance by summing the squared differences between corresponding pixels of the two images.*
4. **What happens if we choose K = 1?**
   *The model becomes highly sensitive to noise. It will look only at the single closest image. It might overfit.*
5. **Why did we use a subset of the dataset?**
   *Because calculating the distance from one test image to 60,000 training images is computationally expensive. Doing this for 10,000 test images takes a very long time on a standard CPU.*

## Key Learning Points

- How to preprocess 2D image data for traditional 1D Machine Learning models.
- The difference between eager learning (like Logistic Regression) and lazy learning (KNN).
- The tradeoff between data size and computational speed in distance-based algorithms.

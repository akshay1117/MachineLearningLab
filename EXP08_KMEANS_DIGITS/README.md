# Experiment 8: K-Means Clustering

## Aim

To implement K-Means Clustering on the Digits dataset to group similar handwritten digits together without using their labels (Unsupervised Learning).

## Task

Load the Digits dataset, scale the features, apply the K-Means clustering algorithm to find 10 clusters, evaluate the clustering using Inertia and Silhouette Score, and visualize the clusters by reducing the 64-dimensional data to 2 dimensions using PCA.

## Theory

**K-Means** is an **Unsupervised Machine Learning** algorithm. Unlike classification or regression, unsupervised algorithms do not use target labels (`y`) during training. They try to find hidden structures or patterns within the input data (`X`) alone.

**How it works:**

1. **Initialization:** The algorithm randomly places `K` centroids (center points) in the data space.
2. **Assignment:** It calculates the distance (usually Euclidean) from every data point to all `K` centroids. Each data point is assigned to the cluster of its closest centroid.
3. **Update:** The algorithm recalculates the position of each centroid by taking the mean (average) position of all the data points currently assigned to that cluster.
4. **Repeat:** Steps 2 and 3 are repeated until the centroids stop moving (convergence) or a maximum number of iterations is reached.

## Implementation Plan

1. **Import Libraries:** Import Numpy, Matplotlib, and Scikit-learn libraries.
2. **Load Dataset:** Load the Digits dataset using `load_digits()`.
3. **Separate Data:** Separate the 64-pixel features from original digit labels.
4. **Standardize Data:** Standardize features using `StandardScaler`.
5. **Apply PCA:** Apply PCA to obtain two components for 2D cluster visualization.
6. **Implement K-Means:** Implement the K-Means algorithm from scratch using Lloyd's method.
7. **Initialize Centroids:** Initialize K random centroids from the dataset.
8. **Assign Samples:** Assign each sample to its nearest centroid using squared Euclidean distance.
9. **Update Centroids:** Update each centroid using the mean of its cluster samples.
10. **Repeat Steps:** Repeat assignment and update steps until convergence.
11. **Calculate Scores (K=2 to 20):** Calculate Inertia and Silhouette score for different K values (2 to 20).
12. **Calculate ARI:** Calculate the Adjusted Rand Index (ARI) to compare clusters with actual digit labels.
13. **Plot Metrics:** Plot Inertia, Silhouette score, and ARI against different K values.
14. **Visualize Clusters:** Visualize selected cluster configurations (e.g., k=3, 6, 10, 15) using PCA components.
15. **Display Centroids:** Display the k=10 cluster centroids as 8x8 digit images and analyze the effects of K.

## Dataset

- **Name:** Sklearn Digits Dataset
- **Source:** Built-in `sklearn` datasets.
- **Features:** 64 numerical values (representing pixel intensities of 8x8 images).
- **Target:** None used during training.
- **Obtaining it:** It loads automatically from the library.

## Algorithm

- `sklearn.cluster.KMeans`
- `init='k-means++'` ensures the initial random centroids are spread out to speed up convergence.

## Expected Output

When you run the program, the console will output:

- Dataset dimensions (samples and 64 features).
- The applied K value (10).
- Evaluation Metrics: Inertia and Silhouette Score.
- A visualization window showing two PCA-reduced scatter plots. You should notice that the visual groupings the K-Means algorithm discovered (left) closely match the actual true categories of the digits (right).

## Viva Questions

1. **What is the difference between K-Means and KNN?**
   *K-Means is an unsupervised clustering algorithm that finds hidden groups in data. KNN is a supervised classification algorithm that assigns a label to a new point based on its closest known neighbors.*
2. **What does 'K' stand for in K-Means?**
   *K represents the number of clusters (and therefore the number of centroids) the algorithm should find.*
3. **What is Inertia in K-Means?**
   *Inertia is the sum of squared distances of all data points to their respective cluster centroids. The algorithm tries to minimize this value.*
4. **Why did we use PCA in this experiment?**
   *The data has 64 dimensions. We used Principal Component Analysis (PCA) strictly to reduce those 64 dimensions down to 2 dimensions so we could plot it on a standard 2D graph.*
5. **How would you determine the best 'K' if you didn't know it was 10?**
   *By using the Elbow Method: running K-Means for a range of K values, plotting their Inertia, and looking for the "elbow" point where the decrease in inertia sharply levels off.*

## Key Learning Points

- The fundamental difference between supervised and unsupervised learning.
- How K-Means iteratively updates centroids.
- Using PCA as a tool for visualizing high-dimensional data.

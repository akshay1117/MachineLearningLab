# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 8: K-MEANS CLUSTERING
# DATASET: HANDWRITTEN DIGITS
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP08_KMEANS_DIGITS/kmeans_digits.py
# 5. Review the console output for evaluation metrics and close plot windows to exit.
# 
# ==============================================================================

# K-Means Clustering on the Digits Dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ==============================================================================
# 1. Load the dataset
# ==============================================================================
digits = load_digits()
X, y = digits.data, digits.target

print("Dataset shape:", X.shape)
print("Number of true classes:", len(np.unique(y)))
print("Pixel value range:", X.min(), "to", X.max())


# ==============================================================================
# 2. Feature scaling 
# ==============================================================================
# (zero mean, unit variance so no pixel dominates the distance calculation)
X_scaled = StandardScaler().fit_transform(X)


# ==============================================================================
# 3. PCA to 2 components 
# ==============================================================================
# (used only for visualising the 64-dimensional clusters in 2D space)
X_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)


# ==============================================================================
# 4. K-Means implemented from scratch (Lloyd's algorithm)
# ==============================================================================
def kmeans_scratch(X, k, max_iters=100, tol=1e-4, seed=42):
    """Custom implementation of the K-Means clustering algorithm."""
    rng = np.random.default_rng(seed)
    # initialise k random centroids by picking random data points
    centroids = X[rng.choice(len(X), k, replace=False)] 
    
    for _ in range(max_iters):
        # calculate squared euclidean distances using numpy broadcasting
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        
        # assign each point to nearest centroid
        labels = dists.argmin(axis=1) 
        
        # move centroid to cluster mean. If a cluster is empty, keep old centroid.
        new_c = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)])
        
        shift = np.linalg.norm(new_c - centroids)
        centroids = new_c
        
        # stop when centroids stop moving (convergence)
        if shift < tol: 
            break
            
    # sum of squared distances to centroids
    inertia = ((X - centroids[labels]) ** 2).sum() 
    return labels, centroids, inertia

lab_s, _, inertia_s = kmeans_scratch(X_scaled, 10)
print(f"\n[From scratch] K=10 | Inertia: {inertia_s:.1f} | Silhouette: {silhouette_score(X_scaled, lab_s):.4f}")


# ==============================================================================
# 5. Run K-Means (scikit-learn) for different numbers of clusters and evaluate each
# ==============================================================================
k_values = list(range(2, 21))
inertias, silhouettes, aris = [], [], []

for k in k_values:
    # init="k-means++" ensures smart initialization of centroids to speed up convergence
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    
    # compactness (lower = tighter clusters)
    inertias.append(model.inertia_) 
    # separation vs cohesion (-1 to 1)
    silhouettes.append(silhouette_score(X_scaled, labels)) 
    # agreement with true digit labels
    aris.append(adjusted_rand_score(y, labels)) 


# ==============================================================================
# 6. Compare all values of K
# ==============================================================================
print("\nK-Means Comparison\n" + "=" * 50)
print(f"{'K':>3} {'Inertia':>12} {'Silhouette':>11} {'ARI':>7}")
for k, i, s, a in zip(k_values, inertias, silhouettes, aris):
    print(f"{k:>3} {i:>12.1f} {s:>11.4f} {a:>7.3f}")


# ==============================================================================
# 7. Select the best K using the silhouette score
# ==============================================================================
best_k = k_values[int(np.argmax(silhouettes))]
print("\nBest K by silhouette score:", best_k)


# ==============================================================================
# 8. Elbow estimate 
# ==============================================================================
# (point farthest from the line joining first and last inertia)
pts = np.column_stack([k_values, inertias]).astype(float)
line = (pts[-1] - pts[0]) / np.linalg.norm(pts[-1] - pts[0])
vec = pts - pts[0]
elbow_k = k_values[int(np.abs(vec[:, 0] * line[1] - vec[:, 1] * line[0]).argmax())]
print("Elbow estimate of K:", elbow_k)


# ==============================================================================
# 9. Plot inertia, silhouette score and ARI against K 
# ==============================================================================
# (black lines + different markers/linestyles for B&W printing)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

ax[0].plot(k_values, inertias, color="black", marker="o", ls="-")
ax[0].axvline(10, color="gray", ls="--", label="K=10")
ax[0].set_title("Elbow Method - Inertia vs K")
ax[0].set_xlabel("K")
ax[0].set_ylabel("Inertia")
ax[0].legend()

ax[1].plot(k_values, silhouettes, color="black", marker="s", ls="--")
ax[1].axvline(best_k, color="gray", ls=":", lw=2, label=f"Best K={best_k}")
ax[1].set_title("Silhouette Score vs K")
ax[1].set_xlabel("K")
ax[1].set_ylabel("Silhouette")
ax[1].legend()

ax[2].plot(k_values, aris, color="black", marker="^", ls="-.")
ax[2].axvline(10, color="gray", ls="--")
ax[2].set_title("Adjusted Rand Index vs K")
ax[2].set_xlabel("K")
ax[2].set_ylabel("ARI")

for a in ax: 
    a.grid(True, ls=":", color="gray")

plt.savefig("exp8_elbow_silhouette.png")


# ==============================================================================
# 10. Visualise clusters in PCA space 
# ==============================================================================
# (B&W-friendly: unique marker + gray shade per cluster + cluster number at the centre)
markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p", "d", "H", "8"] # 15 distinct marker shapes

def plot_clusters(a, lab, k):
    shades = plt.cm.gray(np.linspace(0.05, 0.85, k)) # dark -> light gray shades
    for c in range(k):
        pts = X_2d[lab == c]
        a.scatter(
            pts[:, 0], pts[:, 1], 
            marker=markers[c % len(markers)], 
            color=shades[c], 
            edgecolors="black", 
            linewidths=0.4, 
            s=22
        )
        # label each cluster with its number
        cx, cy = pts.mean(axis=0) 
        a.text(
            cx, cy, str(c), 
            ha="center", va="center", 
            fontsize=10, fontweight="bold", 
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black")
        )
    a.set_title(f"K={k} (Silhouette={silhouette_score(X_scaled, lab):.3f})")

fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
for a, k in zip(axes, [3, 6, 10, 15]):
    plot_clusters(a, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled), k)
    
plt.savefig("exp8_pca_clusters.png")


# ==============================================================================
# 11. Display the cluster centroids (K=10) as 8x8 digit images
# ==============================================================================
km10 = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X)
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for a, c in zip(axes.ravel(), km10.cluster_centers_):
    a.imshow(c.reshape(8, 8), cmap="gray_r")
    a.axis("off")
plt.savefig("exp8_centroids.png")


# ==============================================================================
# Discussion & Conclusion
# ==============================================================================
print("""
--- Discussion: Effect of K on Clustering Quality ---
Inertia: always decreases as K increases (points get closer to a centroid), so it cannot pick K alone; look
for the elbow (~K=9-10).
Silhouette: stays low (~0.10-0.16) because digit classes overlap in 64-D; it rises after K=7 and peaks near
K=15.
Small K (2-5): merges different digits, giving very low ARI (0.13-0.29). Large K (15-20): splits digits
into writing styles (over-segmentation).
K=10: matches the true number of digits, ARI ~0.53 and centroids resemble real digits.

CONCLUSION: No single metric selects K reliably; combine elbow, silhouette and domain knowledge.
K=10 is a sensible choice, though K-means only partially recovers digit classes because clusters are
non-spherical and overlapping.
""")

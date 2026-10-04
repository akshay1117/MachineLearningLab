# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 5: DECISION TREE
# DATASET: ONLINE RETAIL
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python EXP05_DECISION_TREE_ONLINE_RETAIL/decision_tree_online_retail.py
# 5. Review the console output for evaluation metrics and close the plot windows to exit.
# 
# ==============================================================================

# ID3 Decision Tree for Customer Segmentation
# Online Retail Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ------------------------------------------------------------------------------
# EXTRA: Download helper (not strictly in PDF, but required to run automatically)
# ------------------------------------------------------------------------------
file_path = "data/Online_Retail.xlsx"
if not os.path.exists(file_path):
    print("Downloading Online Retail dataset (this may take a minute due to file size)...")
    os.makedirs("data", exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    urllib.request.urlretrieve(url, file_path)
    print("Download complete.")
# ------------------------------------------------------------------------------

# 1. Load dataset
# The 'openpyxl' engine is explicitly required to read modern .xlsx files in pandas.
df = pd.read_excel(file_path, engine="openpyxl")

# 2. Preprocessing
# Drop rows that do not have a CustomerID (anonymous checkouts)
df = df.dropna(subset=["CustomerID"])
# Remove cancelled orders (Invoices starting with 'C')
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
# Keep only valid transactions (Quantity and Price must be positive)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
# Calculate TotalPrice = Quantity * UnitPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
# Ensure InvoiceDate is a datetime object
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# 3. Create customer features
# Group all transactions by CustomerID to create a summary profile per customer.
customer = df.groupby("CustomerID").agg(
    TotalSpend=("TotalPrice", "sum"),
    TotalQuantity=("Quantity", "sum"), 
    NumInvoices=("InvoiceNo", "nunique"),
    NumProducts=("StockCode", "nunique"),
    LastPurchase=("InvoiceDate", "max")
).reset_index()

# Calculate 'Recency': how many days since their last purchase.
# We set a reference date 1 day after the maximum date in the dataset.
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
customer["Recency"] = (reference_date - customer["LastPurchase"]).dt.days

# Calculate derived averages per customer
customer["AvgOrderValue"] = (customer["TotalSpend"] / customer["NumInvoices"])
customer["AvgQuantity"] = (customer["TotalQuantity"] / customer["NumInvoices"])

# 4. Create customer segments
# We segment customers into 3 buckets (Low, Medium, High) based on their TotalSpend.
# qcut creates bins such that each bin has roughly the same number of customers.
customer["Segment"] = pd.qcut(customer["TotalSpend"], 3, labels=["Low", "Medium", "High"])

# 5. Select features
# These are the independent variables we will use to build the Decision Tree.
features = ["TotalQuantity", "NumInvoices", "NumProducts", "Recency", "AvgOrderValue", "AvgQuantity"]
X = customer[features].values
# The target variable is the Segment. We cast it to string to avoid categorical type issues.
y = customer["Segment"].astype(str).values

# Split data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ==============================================================================
# 6. ID3 Algorithm (Custom Implementation)
# ==============================================================================
class Node:
    """A Node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       # Index of feature to split on
        self.threshold = threshold   # Value of the feature to split on
        self.left = left             # Left subtree (<= threshold)
        self.right = right           # Right subtree (> threshold)
        self.value = value           # Majority class label (if it's a leaf node)

class ID3:
    """ID3 Decision Tree algorithm using Information Gain (Entropy)."""
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.importance = None
    
    def entropy(self, y):
        """Calculates the entropy of a label array."""
        counts = Counter(y)
        p = np.array(list(counts.values())) / len(y)
        return -np.sum(p * np.log2(p))
    
    def best_split(self, X, y):
        """Finds the best feature and threshold to split on to maximize Information Gain."""
        best_gain = 0
        best_feature = None
        best_threshold = None
        parent_entropy = self.entropy(y)
        
        # Iterate over all features
        for f in range(X.shape[1]):
            values = np.unique(X[:, f])
            # Check midpoints between adjacent unique values as potential thresholds
            for t in (values[:-1] + values[1:]) / 2:
                left = X[:, f] <= t
                right = ~left
                
                # Skip if split doesn't divide the data
                if left.sum() == 0 or right.sum() == 0:
                    continue
                    
                # Calculate Information Gain
                gain = parent_entropy - (left.sum() / len(y) * self.entropy(y[left]) + right.sum() / len(y) * self.entropy(y[right]))
                
                # Keep track of the best split found so far
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t
                    
        return best_feature, best_threshold, best_gain
    
    def build(self, X, y, depth=0):
        """Recursively builds the decision tree."""
        # Find the most common label in the current subset
        majority = Counter(y).most_common(1)[0][0]
        
        # Stopping criteria: pure node or max depth reached
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return Node(value=majority)
            
        # Find the best split
        feature, threshold, gain = self.best_split(X, y)
        
        # Stopping criteria: no split improves information gain
        if feature is None:
            return Node(value=majority)
            
        left = X[:, feature] <= threshold
        right = ~left
        
        # Accumulate feature importance based on information gain
        self.importance[feature] += gain * len(y)
        
        # Recurse left and right
        return Node(
            feature=feature,
            threshold=threshold,
            left=self.build(X[left], y[left], depth + 1), 
            right=self.build(X[right], y[right], depth + 1)
        )
    
    def fit(self, X, y):
        """Trains the decision tree on the dataset."""
        self.importance = np.zeros(X.shape[1])
        self.root = self.build(X, y)
        total = self.importance.sum()
        if total > 0:
            self.importance /= total
            
    def predict_one(self, x, node):
        """Predicts the class label for a single sample."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)
        
    def predict(self, X):
        """Predicts class labels for an array of samples."""
        return np.array([ self.predict_one(x, self.root) for x in X ])


# ==============================================================================
# Execution & Evaluation
# ==============================================================================

# 7. Train ID3
model = ID3(max_depth=4)
model.fit(X_train, y_train)

# 8. Prediction and accuracy
pred = model.predict(X_test)
print("ID3 Accuracy:", round(accuracy_score(y_test, pred), 3))

# 9. Feature importance
importance = pd.DataFrame({
    "Feature": features, 
    "Importance": model.importance
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# 10. Feature importance graph
plt.figure(figsize=(8, 5))
plt.barh(importance["Feature"], importance["Importance"])
plt.gca().invert_yaxis() # Highest importance on top
plt.xlabel("Importance")
plt.title("ID3 Feature Importance")
plt.savefig("exp5_feature_importance.png")

# 11. Decision Tree visualization
# We use sklearn's DecisionTreeClassifier to plot the tree visually because 
# plotting a custom tree class requires complex custom graphics code.
tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(18, 9))
plot_tree(
    tree,
    feature_names=features, 
    class_names=["High", "Low", "Medium"], 
    filled=True, 
    rounded=True
)
plt.title("ID3 Decision Tree")
plt.savefig("exp5_decision_tree.png")

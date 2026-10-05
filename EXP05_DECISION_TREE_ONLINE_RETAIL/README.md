# Experiment 5: Decision Tree Classifier

## Aim

To use a Decision Tree algorithm for customer segmentation on the Online Retail dataset.

## Task

Load the transactional Online Retail dataset, group the data by Customer ID to create customer-level features (like total items bought and number of transactions), construct a target variable (High Value vs Standard Customer), and train a Decision Tree classifier using the Entropy criterion (similar to ID3 algorithm). Evaluate and visualize the tree.

## Theory

**Decision Tree** is a supervised machine learning algorithm that can be used for both classification and regression. It operates by breaking down a dataset into smaller and smaller subsets while simultaneously an associated decision tree is incrementally developed.

**Structure:**

- **Root Node:** Represents the entire dataset. It is the starting point of the decision tree.
- **Internal Nodes:** Represent a decision or a test on a specific feature.
- **Leaf Nodes:** Represent the final class labels (the predicted output).

**How it makes decisions:**
The algorithm decides which feature to split on at each node using metrics like **Information Gain** and **Entropy**.

- **Entropy** is a measure of impurity or randomness in the data.
- **Information Gain** measures the reduction in entropy after a dataset is split on an attribute. The attribute that provides the highest information gain is chosen as the splitting node. (This is the core of the ID3 algorithm).

## Implementation Plan

1. **Import Libraries:** Import Pandas, Numpy, Matplotlib, and Scikit-learn.
2. **Load Dataset:** Load the Online Retail Dataset from the Excel file.
3. **Clean Data:** Drop rows with missing customer IDs, remove cancelled orders, and retain only positive quantities/prices.
4. **Aggregate Features:** Group by customer ID to aggregate total spend, total quantity, unique invoices, unique products, and last purchase date.
5. **Derive Features:** Derive Recency, Avg Order Value, and Avg Quantity per invoice.
6. **Create Target Labels:** Divide Total Spend into 3 equal-sized quantile bins (low, medium, high) to create customer segments.
7. **Split Dataset:** Split into 80% training and 20% testing sets, stratified by segment.
8. **Implement ID3:** Implement the ID3 algorithm from scratch (Node class, entropy calculation, information gain, recursive building).
9. **Train Custom Model:** Train the custom ID3 model on the training set and generate predictions.
10. **Evaluate Model:** Evaluate accuracy, compute feature importance (based on information gain), and visualize importance as a horizontal bar chart.

## Dataset

- **Name:** Online Retail Dataset
- **Source:** UCI Machine Learning Repository.
- **Features (Engineered):** `NumTransactions`, `TotalItems`.
- **Target (Constructed):** `IsHighValue` (Binary classification).
- **Obtaining it:** The script attempts to download the 20MB `.xlsx` file automatically via pandas. Alternatively, you can manually download it from UCI and place it at `data/Online_Retail.xlsx`.

## Algorithm

- `sklearn.tree.DecisionTreeClassifier`.
- Using `criterion='entropy'` aligns closely with the ID3 algorithm mentioned in the KTU syllabus.
- We set `max_depth=3` to prevent **overfitting** (where the tree becomes too complex and just memorizes the training data) and to make the plotted tree easily readable on a screen.

## Expected Output

When you run the program, the console will output:

- The download progress (if applicable).
- The shape of the newly engineered customer-level dataset.
- The evaluation metrics (Accuracy, Classification Report).
- A matplotlib window showing a Confusion Matrix and a visual representation of the Decision Tree itself, where you can see the exact thresholds (e.g., `NumTransactions <= 2.5`) used to split the customers.

## Viva Questions & Answers
1. **What is a Decision Tree and how does it make predictions?**
   **Answer:** A Decision Tree is a flowchart-like model that splits data into branches based on feature conditions, eventually reaching a leaf node that represents the predicted class or value.
2. **Explain Entropy and Information Gain.**
   **Answer:** Entropy is a measure of randomness or impurity in a dataset. Information Gain is the reduction in entropy achieved by splitting the data on a specific feature. The tree uses it to choose the best splitting feature.
3. **What is Gini Impurity?**
   **Answer:** Gini impurity measures the frequency at which a randomly chosen element would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the node. It is an alternative to entropy for evaluating splits.
4. **How does a decision tree handle overfitting? What is pruning?**
   **Answer:** Decision trees easily overfit by growing too deep. Pruning is a technique to reduce overfitting by removing branches (nodes) that have little importance or provide little predictive power, simplifying the tree.
5. **What are the main advantages and disadvantages of Decision Trees?**
   **Answer:** **Advantages:** Easy to interpret, visualizable, and requires little data preprocessing (no scaling needed). **Disadvantages:** Highly prone to overfitting, sensitive to small variations in data, and can create biased trees if classes are imbalanced.

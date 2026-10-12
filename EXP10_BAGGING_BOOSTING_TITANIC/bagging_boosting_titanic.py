# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 10: ENSEMBLE LEARNING (BAGGING AND BOOSTING)
# DATASET: TITANIC
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python3 EXP10_BAGGING_BOOSTING_TITANIC/bagging_boosting_titanic.py
#    (or `python` on Windows / within an active virtualenv)
# 5. Review the console output for evaluation metrics and close plot windows to exit.
# 
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)


# ==============================================================================
# 1. LOAD AND PREPROCESS THE TITANIC DATASET
# ==============================================================================
print("==============================================================================")
print("EXPERIMENT 10: BAGGING AND BOOSTING ENSEMBLES ON TITANIC DATASET")
print("==============================================================================")

dataset_path = os.path.join("data", "titanic.csv")

# Load from local data directory, or download via seaborn if missing
if os.path.exists(dataset_path):
    print(f"\n1. Loading Titanic dataset from local file: {dataset_path}")
    raw_data = pd.read_csv(dataset_path)
else:
    print("\n1. Local dataset not found. Downloading Titanic dataset via seaborn...")
    raw_data = sns.load_dataset("titanic")
    os.makedirs("data", exist_ok=True)
    raw_data.to_csv(dataset_path, index=False)
    print(f"   Saved local copy to {dataset_path}")

print(f"   - Raw Dataset Shape: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")

# Select relevant predictor features and target
selected_features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target_col = "survived"

df = raw_data[selected_features + [target_col]].copy()

print("\n   Missing Values Summary (Before Imputation):")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Preprocessing & Imputation:
# 1. Impute missing Age with median
df["age"] = df["age"].fillna(df["age"].median())
# 2. Impute missing Fare with median (if any)
df["fare"] = df["fare"].fillna(df["fare"].median())
# 3. Impute missing Embarked with mode (most frequent port)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Encoding categorical features:
# 'sex': male -> 0, female -> 1
df["sex"] = df["sex"].map({"male": 0, "female": 1})
# 'embarked': S -> 0, C -> 1, Q -> 2
df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df[selected_features]
y = df[target_col]

print(f"\n   Processed Feature Matrix Shape: {X.shape}")
print(f"   Class Distribution: Survived (1) = {sum(y == 1)}, Perished (0) = {sum(y == 0)}")

# Train-Test Split (80% Training, 20% Testing, Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")


# ==============================================================================
# Helper Function for Model Instantiation Across Scikit-Learn Versions
# ==============================================================================
def make_bagging(base_estimator, n_estimators=50, random_state=42):
    try:
        return BaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state)
    except TypeError:
        return BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=random_state)

def make_adaboost(base_estimator, n_estimators=50, learning_rate=1.0, random_state=42):
    try:
        return AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    except TypeError:
        return AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)


# ==============================================================================
# 2. BASELINE CLASSIFIER: SINGLE DECISION TREE
# ==============================================================================
print("\n2. Training Baseline Single Decision Tree Classifier...")
base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
base_tree.fit(X_train, y_train)

y_pred_base = base_tree.predict(X_test)
y_prob_base = base_tree.predict_proba(X_test)[:, 1]

acc_base = accuracy_score(y_test, y_pred_base)
prec_base = precision_score(y_test, y_pred_base)
rec_base = recall_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)

print(f"   - Single Tree: Accuracy={acc_base:.4f}, Precision={prec_base:.4f}, Recall={rec_base:.4f}, F1-Score={f1_base:.4f}")


# ==============================================================================
# 3. ENSEMBLE METHOD 1: BAGGING (BOOTSTRAP AGGREGATING)
# ==============================================================================
print("\n3. Training Bagging Classifier (Parallel Ensemble)...")
# Bagging trains multiple base decision trees in parallel on random bootstrap
# samples of the training set. Aggregation via majority voting reduces variance.
bagging_tree_base = DecisionTreeClassifier(max_depth=5, random_state=42)
bagging_model = make_bagging(bagging_tree_base, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

y_pred_bag = bagging_model.predict(X_test)
y_prob_bag = bagging_model.predict_proba(X_test)[:, 1]

acc_bag = accuracy_score(y_test, y_pred_bag)
prec_bag = precision_score(y_test, y_pred_bag)
rec_bag = recall_score(y_test, y_pred_bag)
f1_bag = f1_score(y_test, y_pred_bag)

print(f"   - Bagging Ensemble: Accuracy={acc_bag:.4f}, Precision={prec_bag:.4f}, Recall={rec_bag:.4f}, F1-Score={f1_bag:.4f}")


# ==============================================================================
# 4. ENSEMBLE METHOD 2: BOOSTING (ADABOOST)
# ==============================================================================
print("\n4. Training Boosting Classifier (Sequential Ensemble via AdaBoost)...")
# AdaBoost trains weak learners (decision stumps of depth 1) sequentially.
# In each round, weights of misclassified instances are increased so subsequent
# stumps focus on the hardest cases. Aggregation is by weighted majority voting.
boosting_base_stump = DecisionTreeClassifier(max_depth=1, random_state=42)
boosting_model = make_adaboost(boosting_base_stump, n_estimators=50, learning_rate=1.0, random_state=42)
boosting_model.fit(X_train, y_train)

y_pred_boost = boosting_model.predict(X_test)
y_prob_boost = boosting_model.predict_proba(X_test)[:, 1]

acc_boost = accuracy_score(y_test, y_pred_boost)
prec_boost = precision_score(y_test, y_pred_boost)
rec_boost = recall_score(y_test, y_pred_boost)
f1_boost = f1_score(y_test, y_pred_boost)

print(f"   - AdaBoost Ensemble: Accuracy={acc_boost:.4f}, Precision={prec_boost:.4f}, Recall={rec_boost:.4f}, F1-Score={f1_boost:.4f}")


# ==============================================================================
# 5. PRINT FINDINGS & COMPARATIVE ANALYTICS
# ==============================================================================
print("\n" + "="*78)
print("5. PERFORMANCE COMPARISON: BASELINE vs BAGGING vs BOOSTING")
print("="*78)

metrics_df = pd.DataFrame({
    "Model": ["Single Decision Tree", "Bagging (Decision Trees)", "AdaBoost (Decision Stumps)"],
    "Accuracy": [acc_base, acc_bag, acc_boost],
    "Precision": [prec_base, prec_bag, prec_boost],
    "Recall": [rec_base, rec_bag, rec_boost],
    "F1-Score": [f1_base, f1_bag, f1_boost]
})

print(metrics_df.to_string(index=False))

print("\nDetailed Classification Reports:")
for name, preds in [("Single Decision Tree", y_pred_base),
                    ("Bagging Classifier", y_pred_bag),
                    ("AdaBoost Classifier", y_pred_boost)]:
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, target_names=["Perished (0)", "Survived (1)"]))

print("-"*78)
print("DISCUSSION: HOW EACH METHOD IMPROVES PERFORMANCE & PROS/CONS")
print("-"*78)
print("""
1. HOW THEY IMPROVE MODEL PERFORMANCE:
   - Bagging (Variance Reduction):
     * Takes independent bootstrap samples of the data and fits a tree to each.
     * Combines their predictions by majority voting.
     * By averaging over diverse trees, it cancels out individual variance and overfitting,
       improving test accuracy and precision significantly over a single tree.
   - Boosting (Bias Reduction):
     * Sequentially trains simple weak learners (e.g. depth-1 decision stumps).
     * Iteratively increases the weights of misclassified instances.
     * Converts a collection of weak learners into a high-capacity strong classifier,
       effectively reducing model bias and improving recall on harder samples.

2. STRENGTHS AND WEAKNESSES:
   A. Bagging:
      [+] Strengths: Highly resistant to overfitting; trees are independent and can be
          trained in parallel; performs reliably even with noisy data.
      [-] Weaknesses: Can be computationally and memory intensive; does not significantly
          reduce bias if base learners are already biased (underfitting).
   B. Boosting (AdaBoost):
      [+] Strengths: Often achieves higher overall accuracy; base learners are fast and simple;
          effectively handles complex decision boundaries.
      [-] Weaknesses: Sensitive to outliers and noise (may overfit to mislabeled points);
          inherently sequential and cannot be fully parallelized across iterations.
""")


# ==============================================================================
# 6. ESTIMATOR PROGRESSION STUDY & VISUALIZATIONS
# ==============================================================================
print("\n6. Generating Visualization Plots...")

# Evaluate performance as number of estimators increases
estimator_counts = [5, 10, 20, 30, 50, 75, 100]
bag_acc_progression = []
boost_acc_progression = []

for n in estimator_counts:
    bag_test = make_bagging(DecisionTreeClassifier(max_depth=5, random_state=42), n_estimators=n, random_state=42)
    bag_test.fit(X_train, y_train)
    bag_acc_progression.append(accuracy_score(y_test, bag_test.predict(X_test)))

    boost_test = make_adaboost(DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=n, random_state=42)
    boost_test.fit(X_train, y_train)
    boost_acc_progression.append(accuracy_score(y_test, boost_test.predict(X_test)))

# Prepare 6-Panel Visualization
fig = plt.figure(figsize=(16, 12))
grid = plt.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)

# Panel 1: Metric Comparison Bar Chart
ax_metrics = fig.add_subplot(grid[0, 0])
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
bar_width = 0.25
indices = np.arange(len(metric_names))

ax_metrics.bar(indices - bar_width, [acc_base, prec_base, rec_base, f1_base],
               width=bar_width, label='Single Tree', color='#6B7280', alpha=0.9)
ax_metrics.bar(indices, [acc_bag, prec_bag, rec_bag, f1_bag],
               width=bar_width, label='Bagging', color='#3B82F6', alpha=0.9)
ax_metrics.bar(indices + bar_width, [acc_boost, prec_boost, rec_boost, f1_boost],
               width=bar_width, label='AdaBoost', color='#F59E0B', alpha=0.9)

ax_metrics.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax_metrics.set_xticks(indices)
ax_metrics.set_xticklabels(metric_names, fontsize=10)
ax_metrics.set_ylim(0.5, 0.95)
ax_metrics.set_ylabel('Score', fontsize=10)
ax_metrics.legend(loc='lower right', fontsize=9)
ax_metrics.grid(axis='y', linestyle='--', alpha=0.7)

# Panel 2: Confusion Matrix - Single Tree
ax_cm_base = fig.add_subplot(grid[0, 1])
cm_base = confusion_matrix(y_test, y_pred_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax_cm_base, cbar=False,
            xticklabels=['Perished', 'Survived'], yticklabels=['Perished', 'Survived'])
ax_cm_base.set_title(f'Single Tree\n(Accuracy: {acc_base:.3f})', fontsize=11, fontweight='bold')
ax_cm_base.set_xlabel('Predicted')
ax_cm_base.set_ylabel('Actual')

# Panel 3: Confusion Matrix - Bagging
ax_cm_bag = fig.add_subplot(grid[0, 2])
cm_bag = confusion_matrix(y_test, y_pred_bag)
sns.heatmap(cm_bag, annot=True, fmt='d', cmap='Blues', ax=ax_cm_bag, cbar=False,
            xticklabels=['Perished', 'Survived'], yticklabels=['Perished', 'Survived'])
ax_cm_bag.set_title(f'Bagging (50 Trees)\n(Accuracy: {acc_bag:.3f})', fontsize=11, fontweight='bold')
ax_cm_bag.set_xlabel('Predicted')
ax_cm_bag.set_ylabel('Actual')

# Panel 4: Confusion Matrix - AdaBoost
ax_cm_boost = fig.add_subplot(grid[1, 0])
cm_boost = confusion_matrix(y_test, y_pred_boost)
sns.heatmap(cm_boost, annot=True, fmt='d', cmap='Oranges', ax=ax_cm_boost, cbar=False,
            xticklabels=['Perished', 'Survived'], yticklabels=['Perished', 'Survived'])
ax_cm_boost.set_title(f'AdaBoost (50 Stumps)\n(Accuracy: {acc_boost:.3f})', fontsize=11, fontweight='bold')
ax_cm_boost.set_xlabel('Predicted')
ax_cm_boost.set_ylabel('Actual')

# Panel 5: ROC Curves
ax_roc = fig.add_subplot(grid[1, 1])
fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
fpr_bag, tpr_bag, _ = roc_curve(y_test, y_prob_bag)
fpr_boost, tpr_boost, _ = roc_curve(y_test, y_prob_boost)

auc_base = auc(fpr_base, tpr_base)
auc_bag = auc(fpr_bag, tpr_bag)
auc_boost = auc(fpr_boost, tpr_boost)

ax_roc.plot(fpr_base, tpr_base, label=f'Single Tree (AUC = {auc_base:.3f})', color='#6B7280', linestyle=':')
ax_roc.plot(fpr_bag, tpr_bag, label=f'Bagging (AUC = {auc_bag:.3f})', color='#3B82F6', linewidth=2)
ax_roc.plot(fpr_boost, tpr_boost, label=f'AdaBoost (AUC = {auc_boost:.3f})', color='#F59E0B', linewidth=2)
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(loc='lower right', fontsize=9)
ax_roc.grid(True, linestyle='--', alpha=0.7)

# Panel 6: Accuracy vs Number of Estimators Progression
ax_prog = fig.add_subplot(grid[1, 2])
ax_prog.plot(estimator_counts, bag_acc_progression, marker='o', color='#3B82F6', linewidth=2, label='Bagging')
ax_prog.plot(estimator_counts, boost_acc_progression, marker='s', color='#F59E0B', linewidth=2, label='AdaBoost')
ax_prog.axhline(acc_base, color='#6B7280', linestyle='--', label=f'Single Tree ({acc_base:.3f})')
ax_prog.set_title('Ensemble Size vs Accuracy', fontsize=12, fontweight='bold')
ax_prog.set_xlabel('Number of Estimators')
ax_prog.set_ylabel('Test Accuracy')
ax_prog.legend(loc='lower right', fontsize=9)
ax_prog.grid(True, linestyle='--', alpha=0.7)

output_plot_path = "exp10_bagging_vs_boosting.png"
plt.savefig(output_plot_path, dpi=300)
print(f"Visualization saved successfully to: {output_plot_path}")
plt.close()

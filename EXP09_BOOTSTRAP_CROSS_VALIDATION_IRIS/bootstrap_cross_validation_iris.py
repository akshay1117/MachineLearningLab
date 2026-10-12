# ==============================================================================
# KTU MACHINE LEARNING LAB
# EXPERIMENT 9: RESAMPLING METHODS (BOOTSTRAPPING & CROSS-VALIDATION)
# DATASET: IRIS
# ==============================================================================
# 
# USAGE INSTRUCTIONS:
# 1. Open a terminal or command prompt.
# 2. Navigate to the root directory of this repository (MachineLearningLab).
# 3. Ensure you have installed the required packages: `pip install -r requirements.txt`
# 4. Run this script using the following command:
#    python3 EXP09_BOOTSTRAP_CROSS_VALIDATION_IRIS/bootstrap_cross_validation_iris.py
#    (or `python` on Windows / within an active virtualenv)
# 5. Review the console output for evaluation metrics and close plot windows to exit.
# 
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# ==============================================================================
# 1. LOAD AND PREPROCESS THE IRIS DATASET
# ==============================================================================
print("==============================================================================")
print("EXPERIMENT 9: BOOTSTRAPPING AND CROSS-VALIDATION ON IRIS DATASET")
print("==============================================================================")

# Load the classic Iris flower dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
n_samples, n_features = X.shape

print(f"\n1. Dataset Information:")
print(f"   - Number of Samples: {n_samples}")
print(f"   - Number of Features: {n_features} ({', '.join(feature_names)})")
print(f"   - Classes: {len(target_names)} ({', '.join(target_names)})")
print(f"   - Samples per class: {np.bincount(y)}")

# Define base classification pipeline (Standardization + Logistic Regression)
# Standardization is applied inside each resample/fold to prevent data leakage.
def create_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])


# ==============================================================================
# 2. IMPLEMENT BOOTSTRAPPING RESAMPLING
# ==============================================================================
print("\n2. Running Bootstrapping Resampling...")
# Bootstrapping draws N samples with replacement from the original dataset of size N.
# On average, ~63.2% of instances are included in the bootstrap training sample,
# and the remaining ~36.8% form the Out-Of-Bag (OOB) validation sample.

np.random.seed(42)
B = 200  # Number of bootstrap resamples
bootstrap_acc_scores = []
bootstrap_f1_scores = []

for b in range(B):
    # Sample indices with replacement
    boot_train_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    # Identify unselected samples as Out-Of-Bag (OOB)
    oob_idx = np.setdiff1d(np.arange(n_samples), boot_train_idx)
    
    # Ensure OOB contains samples and train contains all target classes
    if len(oob_idx) > 0 and len(np.unique(y[boot_train_idx])) == len(target_names):
        model = create_model()
        model.fit(X[boot_train_idx], y[boot_train_idx])
        
        # Evaluate on the Out-Of-Bag (OOB) samples
        y_oob_pred = model.predict(X[oob_idx])
        acc = accuracy_score(y[oob_idx], y_oob_pred)
        f1 = f1_score(y[oob_idx], y_oob_pred, average='macro')
        
        bootstrap_acc_scores.append(acc)
        bootstrap_f1_scores.append(f1)

bootstrap_acc_scores = np.array(bootstrap_acc_scores)
bootstrap_f1_scores = np.array(bootstrap_f1_scores)

# Calculate summary statistics for bootstrapping
boot_acc_mean = np.mean(bootstrap_acc_scores)
boot_acc_std = np.std(bootstrap_acc_scores)
boot_acc_ci = np.percentile(bootstrap_acc_scores, [2.5, 97.5])

boot_f1_mean = np.mean(bootstrap_f1_scores)
boot_f1_std = np.std(bootstrap_f1_scores)
boot_f1_ci = np.percentile(bootstrap_f1_scores, [2.5, 97.5])

print(f"   - Bootstrap Iterations Completed: {len(bootstrap_acc_scores)}")
print(f"   - OOB Accuracy : Mean = {boot_acc_mean:.4f}, Std = {boot_acc_std:.4f}, 95% CI = [{boot_acc_ci[0]:.4f}, {boot_acc_ci[1]:.4f}]")
print(f"   - OOB F1-Score : Mean = {boot_f1_mean:.4f}, Std = {boot_f1_std:.4f}, 95% CI = [{boot_f1_ci[0]:.4f}, {boot_f1_ci[1]:.4f}]")


# ==============================================================================
# 3. IMPLEMENT K-FOLD CROSS-VALIDATION
# ==============================================================================
print("\n3. Running 10-Fold Stratified Cross-Validation...")
# K-Fold CV partitions the dataset into K disjoint folds.
# Each fold is held out once as the test set while the model is trained on the remaining K-1 folds.

K = 10
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
cv_acc_scores = []
cv_f1_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    model = create_model()
    model.fit(X[train_idx], y[train_idx])
    
    y_test_pred = model.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], y_test_pred)
    f1 = f1_score(y[test_idx], y_test_pred, average='macro')
    
    cv_acc_scores.append(acc)
    cv_f1_scores.append(f1)
    print(f"   - Fold {fold:2d}/{K:2d}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")

cv_acc_scores = np.array(cv_acc_scores)
cv_f1_scores = np.array(cv_f1_scores)

# Calculate summary statistics for K-Fold CV
cv_acc_mean = np.mean(cv_acc_scores)
cv_acc_std = np.std(cv_acc_scores)
cv_acc_range = (np.min(cv_acc_scores), np.max(cv_acc_scores))

cv_f1_mean = np.mean(cv_f1_scores)
cv_f1_std = np.std(cv_f1_scores)
cv_f1_range = (np.min(cv_f1_scores), np.max(cv_f1_scores))

print(f"\n   - K-Fold CV Accuracy : Mean = {cv_acc_mean:.4f}, Std = {cv_acc_std:.4f}, Range = [{cv_acc_range[0]:.4f}, {cv_acc_range[1]:.4f}]")
print(f"   - K-Fold CV F1-Score : Mean = {cv_f1_mean:.4f}, Std = {cv_f1_std:.4f}, Range = [{cv_f1_range[0]:.4f}, {cv_f1_range[1]:.4f}]")


# ==============================================================================
# 4. PRINT FINDINGS & COMPARISON TABLE
# ==============================================================================
print("\n" + "="*78)
print("4. COMPARATIVE PERFORMANCE ANALYSIS: BOOTSTRAPPING VS CROSS-VALIDATION")
print("="*78)

comparison_df = pd.DataFrame({
    "Resampling Method": ["Bootstrapping (OOB, B=200)", "Bootstrapping (OOB, B=200)", 
                          f"K-Fold CV (K={K})", f"K-Fold CV (K={K})"],
    "Metric": ["Accuracy", "F1-Score (Macro)", "Accuracy", "F1-Score (Macro)"],
    "Mean Score": [boot_acc_mean, boot_f1_mean, cv_acc_mean, cv_f1_mean],
    "Std Dev": [boot_acc_std, boot_f1_std, cv_acc_std, cv_f1_std],
    "95% CI / Range": [
        f"[{boot_acc_ci[0]:.4f}, {boot_acc_ci[1]:.4f}] (95% CI)",
        f"[{boot_f1_ci[0]:.4f}, {boot_f1_ci[1]:.4f}] (95% CI)",
        f"[{cv_acc_range[0]:.4f}, {cv_acc_range[1]:.4f}] (Min-Max)",
        f"[{cv_f1_range[0]:.4f}, {cv_f1_range[1]:.4f}] (Min-Max)"
    ]
})

print(comparison_df.to_string(index=False))

print("\n" + "-"*78)
print("DISCUSSION: ADVANTAGES AND DISADVANTAGES OF EACH METHOD")
print("-"*78)
print("""
A. BOOTSTRAPPING:
   [+] Advantages:
       1. Yields a complete empirical distribution, allowing direct calculation of 
          standard errors and non-parametric confidence intervals (e.g. 95% CI).
       2. Out-of-Bag (OOB) samples provide built-in validation without needing an 
          independent test split or reducing training size.
       3. Highly effective on smaller datasets where data cannot be spared for large test sets.
   [-] Disadvantages:
       1. Higher computational complexity due to training hundreds of models (B=200).
       2. Sampling with replacement introduces repeated samples, which may cause 
          slight pessimistic bias in OOB evaluation (~36.8% omitted per round).

B. K-FOLD CROSS-VALIDATION:
   [+] Advantages:
       1. Every single sample is guaranteed to be tested exactly once across the K folds.
       2. Computationally faster (trains only K models, e.g. K=10).
       3. Stratification preserves class balance across all folds, reducing variance.
   [-] Disadvantages:
       1. Provides fewer metric estimates (only K values), making empirical confidence 
          intervals less granular unless repeated cross-validation is used.
       2. Can have higher variance in estimated scores if K is chosen too small or too large.
""")


# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
print("\n5. Generating Visualization Plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Accuracy Box Plot Comparison
axes[0, 0].boxplot([bootstrap_acc_scores, cv_acc_scores], tick_labels=['Bootstrapping (B=200)', f'{K}-Fold CV'], patch_artist=True,
                   boxprops=dict(facecolor='#D0E1FD', color='#1D4ED8'),
                   medianprops=dict(color='#B91C1C', linewidth=2))
axes[0, 0].set_title('Accuracy Distribution: Bootstrap vs K-Fold CV', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: F1-Score Box Plot Comparison
axes[0, 1].boxplot([bootstrap_f1_scores, cv_f1_scores], tick_labels=['Bootstrapping (B=200)', f'{K}-Fold CV'], patch_artist=True,
                   boxprops=dict(facecolor='#D1FAE5', color='#047857'),
                   medianprops=dict(color='#B91C1C', linewidth=2))
axes[0, 1].set_title('F1-Score Distribution: Bootstrap vs K-Fold CV', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Macro F1-Score', fontsize=11)
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 3: Bar Chart of Mean Scores with Error Bars (Std Dev)
labels = ['Accuracy', 'F1-Score']
boot_means = [boot_acc_mean, boot_f1_mean]
boot_stds = [boot_acc_std, boot_f1_std]
cv_means = [cv_acc_mean, cv_f1_mean]
cv_stds = [cv_acc_std, cv_f1_std]

x = np.arange(len(labels))
width = 0.35

rects1 = axes[1, 0].bar(x - width/2, boot_means, width, yerr=boot_stds, capsize=5, label='Bootstrapping (OOB)', color='#3B82F6', alpha=0.85)
rects2 = axes[1, 0].bar(x + width/2, cv_means, width, yerr=cv_stds, capsize=5, label=f'{K}-Fold CV', color='#10B981', alpha=0.85)

axes[1, 0].set_ylabel('Score', fontsize=11)
axes[1, 0].set_title('Comparison of Mean Metrics ± Std Dev', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(labels, fontsize=11)
axes[1, 0].set_ylim(0.80, 1.02)
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on the bars
for rect in rects1 + rects2:
    height = rect.get_height()
    axes[1, 0].annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 4: Cumulative Convergence of Bootstrap Mean vs K-Fold Benchmark
cumulative_boot_acc = np.cumsum(bootstrap_acc_scores) / np.arange(1, len(bootstrap_acc_scores) + 1)
axes[1, 1].plot(range(1, B + 1), cumulative_boot_acc, color='#2563EB', linewidth=2, label='Bootstrap Cumulative Mean')
axes[1, 1].axhline(cv_acc_mean, color='#DC2626', linestyle='--', linewidth=2, label=f'{K}-Fold CV Mean ({cv_acc_mean:.3f})')
axes[1, 1].set_title('Bootstrap Mean Accuracy Convergence Over Iterations', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Bootstrap Iteration Number', fontsize=11)
axes[1, 1].set_ylabel('Cumulative Mean Accuracy', fontsize=11)
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
output_plot_path = "exp9_resampling_comparison.png"
plt.savefig(output_plot_path, dpi=300)
print(f"Visualization saved successfully to: {output_plot_path}")
plt.close()

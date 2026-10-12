# Experiment 10: Ensemble Learning (Bagging and Boosting)

## Aim

To implement Bagging (Bootstrap Aggregation) and Boosting (AdaBoost) ensemble methods on the Titanic dataset, evaluate and compare model performance metrics (Accuracy, Precision, Recall, and F1-score), and analyze how each method improves performance alongside their respective strengths and weaknesses.

## Task

Load and preprocess the Titanic passenger dataset, implement Bagging using Decision Trees as the base estimator, implement Boosting using AdaBoost with Decision Stumps as the base estimator, compare performance with a baseline single Decision Tree, evaluate classification metrics (Accuracy, Precision, Recall, F1-Score), examine the effect of ensemble size on accuracy, and discuss how variance reduction and bias reduction improve model predictions.

## Theory

**Ensemble Learning** is a machine learning paradigm where multiple models (often referred to as "weak learners" or "base estimators") are trained and combined to solve a problem, producing a final predictive model that typically outperforms any individual constituent model.

### 1. Bagging (Bootstrap Aggregation)

- **Concept:** Bagging combines predictions from multiple models trained independently and in parallel on different bootstrap subsets of the training data.
- **Mechanism:**
  1. Generate $B$ bootstrap samples $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_B$ by sampling $N$ instances with replacement from the training dataset.
  2. Train a separate base estimator $h_b(x)$ (e.g., deep decision tree) on each bootstrap sample $\mathcal{D}_b$.
  3. Aggregate predictions: For classification, the final output is determined by **majority voting**:
     $$\hat{y}_{\text{bag}} = \operatorname{mode}\left(\{h_1(x), h_2(x), \dots, h_B(x)\}\right)$$
- **Primary Effect:** **Variance Reduction**. Deep individual decision trees have low bias but high variance (they easily overfit). Averaging independent trees drastically reduces variance without increasing bias.

### 2. Boosting (Adaptive Boosting / AdaBoost)

- **Concept:** Boosting is an iterative, sequential ensemble technique where base learners are trained sequentially, each attempting to correct the errors made by its predecessors.
- **Mechanism:**
  1. Initialize equal observation weights $w_i = \frac{1}{N}$ for all samples.
  2. For each iteration $m = 1, \dots, M$:
     - Fit a weak learner $h_m(x)$ (typically a Decision Stump of depth 1) to the data using current sample weights.
     - Compute the weighted classification error:
       $$\epsilon_m = \frac{\sum_{i: y_i \neq h_m(x_i)} w_i}{\sum_{i=1}^N w_i}$$
     - Compute the estimator weight $\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$.
     - Update sample weights: Increase weights of misclassified instances and decrease weights of correctly classified instances:
       $$w_i \leftarrow w_i \exp\left(-\alpha_m y_i h_m(x_i)\right)$$
     - Renormalize sample weights so that $\sum_{i=1}^N w_i = 1$.
  3. Combine estimators via **weighted majority vote**:
     $$\hat{y}_{\text{boost}} = \operatorname{sign}\left(\sum_{m=1}^M \alpha_m h_m(x)\right)$$
- **Primary Effect:** **Bias Reduction**. Boosting converts a collection of weak learners that individually perform only slightly better than random guessing into a high-capacity, low-bias strong learner.

## Implementation Plan

1. **Import Libraries:** Import NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.
2. **Load Dataset:** Load the Titanic dataset from `data/titanic.csv` or automatically download via `seaborn`.
3. **Clean & Impute Missing Data:**
   - Impute missing `age` with median.
   - Impute missing `fare` with median.
   - Impute missing `embarked` with mode (`'S'`).
4. **Feature Encoding:**
   - Map `sex` to binary integers (`male`: 0, `female`: 1).
   - Map `embarked` port codes to numeric values (`S`: 0, `C`: 1, `Q`: 2).
5. **Train-Test Split:** Split into 80% training and 20% testing sets, stratified by survival status (`stratify=y`).
6. **Train Baseline Decision Tree:** Train a single Decision Tree (`max_depth=5`) to provide a baseline reference.
7. **Train Bagging Ensemble:**
   - Configure `BaggingClassifier` with 50 decision tree base estimators (`max_depth=5`).
   - Fit and predict on test set.
8. **Train AdaBoost Ensemble:**
   - Configure `AdaBoostClassifier` with 50 decision stumps (`max_depth=1`).
   - Fit and predict on test set.
9. **Evaluate Metrics:** Calculate Accuracy, Precision, Recall, and F1-score for all three models and display side-by-side.
10. **Analyze Ensemble Size:** Evaluate how model accuracy progresses as the number of estimators increases from 5 to 100.
11. **Visualize Results:** Generate a 6-panel composite figure (`exp10_bagging_vs_boosting.png`) containing metric bars, confusion matrices, ROC curves, and the ensemble size progression trajectory.

## Dataset

- **Name:** Titanic Passenger Survival Dataset
- **Source:** Automatically fetched via Seaborn / cached in `data/titanic.csv`.
- **Target:** `survived` (0 = Perished, 1 = Survived).
- **Features Used:**
  - `pclass`: Ticket class (1st, 2nd, 3rd)
  - `sex`: Gender (male/female)
  - `age`: Age in years
  - `sibsp`: Number of siblings / spouses aboard
  - `parch`: Number of parents / children aboard
  - `fare`: Passenger fare
  - `embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Algorithm

- `sklearn.tree.DecisionTreeClassifier` (Baseline learner)
- `sklearn.ensemble.BaggingClassifier` (Bootstrap Aggregating)
- `sklearn.ensemble.AdaBoostClassifier` (Adaptive Boosting)

## Expected Output

When you run the script, the console will output:

- Raw dataset shape and missing values breakdown.
- Preprocessed feature matrix dimensions and survival distribution.
- Performance metrics for Single Tree, Bagging, and AdaBoost.
- Side-by-side performance comparison table (Accuracy, Precision, Recall, F1-Score).
- Detailed classification reports and confusion matrices.
- Theoretical discussion on how variance reduction (Bagging) and bias reduction (Boosting) affect accuracy, along with strengths and weaknesses.
- A 6-panel visualization (`exp10_bagging_vs_boosting.png`) depicting:
  - Metric comparison bar chart.
  - Three confusion matrices (Single Tree, Bagging, AdaBoost).
  - ROC Curves and AUC metrics.
  - Accuracy progression as ensemble size varies from 5 to 100.

## Viva Questions & Answers

1. **What is Ensemble Learning, and why does it often outperform single models?**
   **Answer:** Ensemble learning combines predictions from multiple individual models (base estimators) to make a final decision. By combining diverse models through averaging or voting, individual errors cancel out, reducing overall variance or bias and improving generalization performance.

2. **What is the fundamental difference between Bagging and Boosting?**
   **Answer:** In Bagging, base learners are trained independently and in parallel on random bootstrap samples, and predictions are aggregated by uniform majority voting (reducing variance). In Boosting, base learners are trained sequentially, where each learner focuses on misclassified samples from earlier learners, and predictions are combined by weighted voting (reducing bias).

3. **Why do we use deep decision trees for Bagging but shallow decision stumps for AdaBoost?**
   **Answer:** Bagging requires base learners with low bias and high variance (deep trees), so that averaging them reduces variance without hurting accuracy. Boosting requires weak learners with high bias and low variance (shallow decision stumps), and relies on sequential weighted iterations to drive down the bias.

4. **What are the Out-of-Bag (OOB) samples in Bagging?**
   **Answer:** In Bagging, each bootstrap sample uses roughly 63.2% of the original training instances. The remaining ~36.8% of instances not included in a particular bootstrap sample are called Out-of-Bag (OOB) samples, and they can be used as an internal validation set to evaluate the model without cross-validation.

5. **What are the primary limitations of AdaBoost compared to Bagging?**
   **Answer:** AdaBoost is highly sensitive to noisy data and outliers because it exponentially increases the weights of misclassified instances, causing it to overfit on noise. Furthermore, because it is sequential, AdaBoost cannot be easily parallelized across estimators like Bagging can.

# Experiment 9: Resampling Methods (Bootstrapping & Cross-Validation)

## Aim

To implement and compare Bootstrapping and K-Fold Cross-Validation resampling methods on the Iris dataset, evaluate model performance metrics (Accuracy and F1-score), and discuss the advantages and disadvantages of each technique.

## Task

Load and preprocess the Iris flower dataset, implement bootstrapping with Out-of-Bag (OOB) evaluation across multiple resamples, implement K-Fold stratified cross-validation, evaluate classification performance metrics (Accuracy, F1-Score), compare the resulting distributions and confidence intervals, and analyze the trade-offs of both resampling techniques.

## Theory

**Resampling methods** are statistical and machine learning techniques that repeatedly draw samples from a dataset and refit a model on each sample to assess model variability, estimate test error, and quantify parameter certainty.

### 1. Bootstrapping

- **Concept:** Bootstrapping is a non-parametric approach based on **sampling with replacement** from an empirical dataset of size $N$.
- **Sample Distribution:** When sampling $N$ observations with replacement, the probability that a specific observation is *not* chosen in a single draw is $\left(1 - \frac{1}{N}\right)$. For large $N$, the probability of an observation being omitted across all $N$ draws converges to:
  $$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368 \quad (36.8\%)$$
- **Out-of-Bag (OOB) Samples:** The remaining unselected $\approx 36.8\%$ of instances form the Out-of-Bag (OOB) set, which serves as a natural test set to evaluate model performance without requiring an external held-out set.
- **Empirical Confidence Intervals:** Running $B$ iterations (e.g., $B = 200$) yields an empirical distribution of test statistics, enabling direct calculation of $95\%$ non-parametric confidence intervals between the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles.

### 2. K-Fold Cross-Validation

- **Concept:** The original dataset is randomly partitioned into $K$ equal-sized, mutually exclusive subsets (folds). In stratified cross-validation, each fold retains the overall class distribution.
- **Evaluation Loop:** For each of the $K$ iterations, one fold is retained as the validation data for testing, and the remaining $K - 1$ folds are combined to form the training set.
- **Final Estimate:** The overall performance estimate is the average score across all $K$ folds:
  $$\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^K A_k$$
  where $A_k$ is the performance metric on fold $k$.

## Implementation Plan

1. **Import Libraries:** Import NumPy, Pandas, Matplotlib, and Scikit-learn modules.
2. **Load Dataset:** Load the 4-feature, 3-class Iris dataset using `sklearn.datasets.load_iris()`.
3. **Build Pipeline:** Encapsulate feature standardization (`StandardScaler`) and a linear classifier (`LogisticRegression`) in a pipeline to avoid data leakage during resampling.
4. **Implement Bootstrapping:**
   - Draw $B = 200$ bootstrap samples with replacement.
   - Extract the Out-of-Bag (OOB) sample for each iteration.
   - Train the model on the bootstrap set and evaluate Accuracy and Macro F1-score on the OOB set.
   - Compute mean, standard deviation, and $95\%$ empirical confidence intervals.
5. **Implement K-Fold Cross-Validation:**
   - Configure a 10-Fold Stratified split (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`).
   - Train and evaluate the model across all 10 folds.
   - Compute mean, standard deviation, and min-max range for Accuracy and Macro F1-score.
6. **Compare Metrics:** Tabulate comparative statistics side-by-side in a clean summary table.
7. **Discuss Trade-Offs:** Analyze advantages, limitations, and suitable practical use cases for both methods.
8. **Visualize Results:** Plot box plots, bar charts with error bars, and convergence curves, saving the figure as `exp9_resampling_comparison.png`.

## Dataset

- **Name:** Iris Dataset
- **Source:** Built-in `sklearn` dataset (`load_iris`).
- **Samples:** 150 instances (50 per class).
- **Features:** 4 numeric attributes (Sepal Length, Sepal Width, Petal Length, Petal Width).
- **Target:** 3 flower classes (Setosa, Versicolor, Virginica).
- **Obtaining it:** Loads automatically through `sklearn.datasets`.

## Algorithm

- `sklearn.linear_model.LogisticRegression(max_iter=1000)` inside `sklearn.pipeline.Pipeline`
- Custom vectorized Bootstrapping with OOB evaluation ($B=200$)
- `sklearn.model_selection.StratifiedKFold(n_splits=10)`

## Expected Output

When you run the script, the console will output:

- Dataset characteristics and class balance.
- Step-by-step progress and mean $\pm$ standard deviation for both Bootstrapping and K-Fold CV.
- A side-by-side comparative analytics table containing Mean Scores, Standard Deviations, and Confidence Intervals / Ranges.
- In-depth theoretical discussion highlighting advantages and disadvantages.
- A four-panel visualization (`exp9_resampling_comparison.png`) showing:
  - Accuracy and F1-score box plots comparing variance.
  - Bar charts displaying mean metric scores $\pm$ standard deviations.
  - Bootstrapping cumulative mean convergence trajectory against the K-Fold CV mean line.

## Viva Questions & Answers

1. **What is the difference between Bootstrapping and Cross-Validation?**
   **Answer:** Bootstrapping resamples data *with replacement* to produce samples of the same size as the original dataset, using the remaining Out-of-Bag (~36.8%) points for validation. Cross-Validation partitions data *without replacement* into $K$ disjoint subsets, systematically using each subset once for testing and the rest for training.

2. **Why is the Out-of-Bag (OOB) sample proportion approximately 36.8%?**
   **Answer:** For a dataset of size $N$, the probability of an individual instance *not* being chosen in $N$ independent draws with replacement is $(1 - 1/N)^N$. As $N \to \infty$, this expression converges to $1/e \approx 0.368$, meaning roughly 36.8% of unique observations are left out in each bootstrap draw.

3. **What is the primary advantage of Bootstrapping over K-Fold Cross-Validation?**
   **Answer:** Bootstrapping generates an entire empirical sampling distribution rather than just a few scalar estimates, enabling the calculation of non-parametric standard errors, variance estimates, and robust $95\%$ confidence intervals.

4. **What is Stratified K-Fold Cross-Validation, and why is it preferred for classification?**
   **Answer:** Stratified K-Fold ensures that each fold contains roughly the same proportion of each target class as the complete dataset. This prevents class imbalance or missing classes in individual folds, resulting in lower variance and more reliable error estimates.

5. **When should you choose K-Fold Cross-Validation over Bootstrapping?**
   **Answer:** K-Fold CV is preferred when computational budget is limited, as it requires training only $K$ models (e.g., 5 or 10) instead of hundreds in bootstrapping. It is also preferred when you must ensure that every single observation is tested exactly once without sample duplication.

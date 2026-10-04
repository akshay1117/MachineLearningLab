# KTU Machine Learning Lab (PCCSL508)

Welcome to the comprehensive Machine Learning Lab repository designed specifically for the KTU (Kerala Technological University) B.Tech CSE Semester 5 syllabus (PCCSL508).

This repository contains clean, student-friendly, and executable Python implementations of the core Machine Learning experiments. It is designed not just to be run, but to be **studied** for lab exams, vivas, and practical understanding.

## 🎯 Purpose of this Repository

- **Learn:** Understand how ML algorithms work from the ground up.
- **Practice:** Run and modify the code to see how different parameters affect output.
- **Prepare:** Use the included README files and viva questions to prepare for exams.
- **Reproduce:** Ensure you can easily reproduce academic results locally.

## 🛠 Software Requirements

- Python 3.8+ (Python 3.10+ recommended)
- `pip` (Python package manager)

## 📦 Installation Instructions

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd MachineLearningLab
   ```

2. **Create a virtual environment (Optional but highly recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 📋 Experiment Index

| No. | Experiment | Algorithm | Dataset |
| --- | --- | --- | --- |
| 1 | [Linear Regression with One Variable](EXP01_LINEAR_REGRESSION_CALIFORNIA_HOUSING/) | Simple Linear Regression | California Housing |
| 2 | [Polynomial Regression](EXP02_POLYNOMIAL_REGRESSION_AUTO_MPG/) | Polynomial Regression | Auto MPG |
| 3 | [Logistic Regression](EXP03_LOGISTIC_REGRESSION_PIMA_DIABETES/) | Logistic Regression | Pima Indians Diabetes |
| 4 | [KNN for Image Classification](EXP04_KNN_IMAGE_CLASSIFICATION/) | K-Nearest Neighbors | Fashion-MNIST |
| 5 | [Decision Tree Classifier](EXP05_DECISION_TREE_ONLINE_RETAIL/) | Decision Tree (ID3/CART) | Online Retail |
| 6 | [Linear Support Vector Machine](EXP06_LINEAR_SVM_IRIS/) | Linear SVM | Iris |
| 7 | [Multilayer Feed Forward Network](EXP07_MLP_WINE_QUALITY/) | Artificial Neural Network (MLP) | Wine Quality |
| 8 | [K-Means Clustering](EXP08_KMEANS_DIGITS/) | K-Means | Digits |

## 📊 Dataset Instructions

Most experiments in this repository use datasets built into `scikit-learn` or `keras` which will download automatically when you run the code.

However, some experiments require you to download external datasets (e.g., Auto MPG, Pima Indians Diabetes, Online Retail, Wine Quality).

- Check the `README.md` inside each experiment's folder for exact dataset download links.
- Place the downloaded `.csv` or `.xlsx` files inside a `data/` folder in the root directory, or next to the scripts as specified in each experiment.
- The Python scripts use a `DATASET_PATH` variable that you can easily modify if you place your dataset elsewhere.

## 🚀 How to Run Experiments

Navigate to the root directory of the repository and run the scripts using Python:

```bash
python EXP01_LINEAR_REGRESSION_CALIFORNIA_HOUSING/linear_regression_california_housing.py
```

## 💡 Study / Viva Tips

- **Read the comments:** Every logical block of code is commented to explain *why* it is there.
- **Check the READMEs:** Each experiment has a `README.md` with Theory, Steps, Expected Output, and **Viva Questions**.
- **Change parameters:** Try changing the number of clusters in K-Means, the degree of polynomial regression, or the learning rate of the neural network to see what happens!

## 🐛 Troubleshooting

- **ModuleNotFoundError:** Make sure you have installed all packages from `requirements.txt`. If you used a virtual environment, make sure it is activated.
- **FileNotFoundError:** You forgot to download the required dataset for that specific experiment. Check the experiment's `README.md`.
- **macOS TensorFlow Issues:** If you are on an Apple Silicon Mac (M1/M2/M3), you might need to install `tensorflow-macos` and `tensorflow-metal` for optimal performance, though standard `tensorflow` should still work via Rosetta or standard pip.

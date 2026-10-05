# Experiment 7: Multilayer Feed Forward Network (MLP)

## Aim

To implement an Artificial Neural Network, specifically a Multilayer Perceptron (MLP), to classify the quality of wine using the Wine Quality dataset.

## Task

Load the dataset, engineer a binary classification target (Good Wine vs Bad Wine), standardize the features, build a sequential neural network architecture using TensorFlow/Keras, compile it, train it using Backpropagation (via Adam optimizer), and plot the training and validation accuracy/loss over epochs.

## Theory

**Multilayer Feed Forward Network (MLP)** is a class of Artificial Neural Network (ANN) consisting of at least three layers of nodes: an input layer, a hidden layer, and an output layer.

- **Feed Forward:** The data strictly moves in one direction—forward—from the input nodes, through the hidden nodes, and to the output nodes. There are no loops or cycles in the network.
- **Neuron/Node:** The basic unit of a neural network. It receives inputs, multiplies them by weights, adds a bias, and passes the sum through an activation function.
- **Activation Function:** Introduces non-linearity to the network, allowing it to learn complex patterns. We use **ReLU** (Rectified Linear Unit) for hidden layers and **Sigmoid** for the final output layer (since we want a probability between 0 and 1).
- **Backpropagation:** During training, the network makes a prediction, calculates the error (loss), and then works backwards through the network to adjust the weights to minimize that error.

## Implementation Plan

1. **Import Libraries:** Import Pandas, Numpy, Matplotlib, and Scikit-learn modules.
2. **Load Dataset:** Load the Wine Quality dataset.
3. **Explore Data:** Display dataset shape, sample records, and check for missing values.
4. **Separate Data:** Separate input features and the quality target variable.
5. **Split Dataset:** Split into 80% training and 20% testing data.
6. **Standardize Data:** Standardize input features using `StandardScaler`.
7. **Define Architectures:** Define multiple MLP hidden layer configurations (e.g., (16), (32), (64), (32, 16), etc.).
8. **Configure Classifiers:** Create an MLP classifier for each architecture (ReLU, Adam optimizer, max_iter=1000).
9. **Enable Early Stopping:** Enable early stopping with a validation set during training.
10. **Train & Record:** Train each network and record the number of training iterations.
11. **Predict:** Generate predictions for the testing dataset using each model.
12. **Evaluate:** Calculate accuracy, precision, recall, F1-score, and training iterations for each.
13. **Compare Models:** Store metrics in a comparison table and identify the architecture with the highest F1-score.
14. **Generate Report:** Generate a detailed classification report for the best-performing model.
15. **Confusion Matrix:** Display the confusion matrix to compare actual and predicted classes.
16. **Plot Results:** Plot average accuracy and F1-score against the number of hidden layers to analyze depth influence.

## Dataset

- **Name:** Wine Quality (Red) Dataset.
- **Source:** UCI Machine Learning Repository.
- **Features:** 11 chemical properties of wine (e.g., fixed acidity, chlorides, pH, alcohol).
- **Target (Constructed):** `is_good_wine` (Binary).
- **Obtaining it:** The script automatically downloads the `winequality-red.csv` file from UCI.

## Algorithm

- **TensorFlow Keras Sequential API**: Used to stack dense (fully connected) layers.
- **Optimizer:** Adam (Adaptive Moment Estimation).
- **Loss Function:** Binary Crossentropy.

## Expected Output

When you run the program, the console will output:

- Dataset and target information.
- A summary table of the Neural Network architecture showing layers and parameters.
- A progress bar showing the loss and accuracy for each of the 50 epochs of training.
- Final Test Accuracy and Classification Report.
- A matplotlib window showing three plots: Model Accuracy curve, Model Loss curve, and a Confusion Matrix.

## Viva Questions & Answers
1. **What is a Multilayer Perceptron (MLP)?**
   **Answer:** An MLP is a class of feedforward artificial neural network consisting of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer, utilizing non-linear activation functions.
2. **What is the role of activation functions in a neural network?**
   **Answer:** They introduce non-linearity into the network, allowing it to learn and model complex, non-linear relationships in the data. Without them, the network would essentially behave like a single-layer linear model.
3. **Explain the concept of forward propagation and backpropagation.**
   **Answer:** Forward propagation is the process of passing input data through the network layers to calculate a prediction and loss. Backpropagation calculates the gradient of the loss function with respect to weights and updates them to minimize the error.
4. **What is the difference between a hidden layer and an output layer?**
   **Answer:** Hidden layers reside between the input and output layers, learning intermediate representations and features. The output layer produces the final prediction or classification probabilities based on the hidden layers' outputs.
5. **How does the learning rate affect the training of an MLP?**
   **Answer:** The learning rate controls the step size during weight updates. A rate too large causes training to diverge or oscillate, while a rate too small makes training extremely slow and prone to getting stuck in local minima.

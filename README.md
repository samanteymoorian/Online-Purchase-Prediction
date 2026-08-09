# Online Purchase Prediction Using PyTorch Neural Networks

A machine-learning project that predicts whether an online shopping session will result in a purchase using a neural network developed with PyTorch.

## Project Overview

The goal of this project is to build a binary classification model that identifies online shopping sessions that are likely to result in a purchase.

The project demonstrates a complete machine-learning workflow, including:

- Exploratory data analysis
- Train/validation/test splitting
- Numerical feature scaling
- Categorical feature encoding
- PyTorch tensor and DataLoader creation
- Neural-network training
- Overfitting analysis using validation loss
- Classification-threshold selection
- Final test evaluation
- Comparison with Logistic Regression

## Dataset

The project uses the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

The dataset contains 12,330 online shopping sessions and 17 predictive features. The target variable, `Revenue`, indicates whether a session resulted in a purchase.

Approximately 15.5% of the sessions resulted in a purchase, making the classification problem imbalanced.

Dataset:
Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset.
UCI Machine Learning Repository.

## Data Preprocessing

The dataset is divided into:

- 70% training data
- 15% validation data
- 15% test data

The split is stratified to maintain approximately the same purchase rate across all three datasets.

Numerical variables are standardized using `StandardScaler`, while categorical variables are transformed using one-hot encoding.

To prevent data leakage, preprocessing is fitted only on the training data.

## Neural Network

The PyTorch neural network uses the following architecture:

Input Features
→ 64 neurons
→ ReLU
→ Dropout
→ 32 neurons
→ ReLU
→ 1 output

The model is trained using:

- BCEWithLogitsLoss
- Adam optimizer
- Mini-batches of 64 observations
- Learning rate of 0.001

Validation loss was used to study overfitting and select the number of training epochs.

## Classification Threshold

Instead of automatically using a threshold of 0.50, several thresholds were evaluated using the validation set.

The threshold that produced the highest validation F1 score was selected for the final neural-network predictions.

Selected threshold: 0.35

## Results
The neural network achieved a ROC-AUC of 0.917 and an F1 score of 0.664 on the held-out test set.
Logistic Regression was included as a baseline model for comparison.

## Technologies

Python, PyTorch, pandas, NumPy, scikit-learn, Matplotlib

## Repository Structure

- `Online Purchase Prediction Using PyTorch Neural Networks.ipynb` — complete analysis and modeling
- `online_shoppers_intention.csv` — dataset

## Author

Saman Teymoorianmotlagh

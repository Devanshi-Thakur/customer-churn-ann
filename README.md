# Customer Churn Prediction using ANN

A machine learning project that predicts whether a bank customer is likely to leave the bank using an Artificial Neural Network built with TensorFlow/Keras.

## Project Overview

The goal of this project is to predict customer churn based on customer information such as age, credit score, balance, activity status, geography, and other banking features.

## Dataset

The project uses the `Churn_Modelling.csv` dataset containing 10,000 bank customers.

Target variable:

- `Exited = 0` → customer stayed
- `Exited = 1` → customer churned

## Preprocessing

- Removed `RowNumber`, `CustomerId`, and `Surname`
- Encoded `Gender`
- One-hot encoded `Geography`
- Split data into training and testing sets
- Used stratified splitting
- Standardized features using `StandardScaler`

## ANN Architecture

- 12 input features
- Hidden layer: 16 neurons, ReLU
- Hidden layer: 8 neurons, ReLU
- Output layer: 1 neuron, Sigmoid

Training setup:

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 20
- Batch size: 32

## Results

### Baseline ANN

- Test Accuracy: ~85.7%
- Churn Precision: ~75%
- Churn Recall: ~45%
- Churn F1-score: ~56%

### Class-Weighted ANN

- Accuracy: ~75%
- Churn Precision: ~43%
- Churn Recall: ~81%
- Churn F1-score: ~57%

## Project Structure

```text
customer-churn-ann/
├── data/
│   └── Churn_Modelling.csv
├── models/
│   ├── baseline_churn_ann.keras
│   ├── weighted_churn_ann.keras
│   ├── scaler.pkl
│   └── feature_names.pkl
├── 01_data_understanding.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Jupyter Notebook

## Future Improvements

- Keras Tuner
- Threshold optimization
- Compare ANN with traditional ML models
- ROC-AUC evaluation
- Streamlit interface
- Deployment

## Acknowledgement

This project is based on the customer churn ANN problem from Krish Naik's `Ineuron-ANN` repository and was rebuilt independently as a learning project with additional preprocessing, evaluation, and class-imbalance experimentation.
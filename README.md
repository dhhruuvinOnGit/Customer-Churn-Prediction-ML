# Customer Churn Prediction

This project aims to predict customer churn based on customer usage patterns and service plan details in a telecom dataset. By predicting churn, businesses can take proactive measures to retain customers.

## Project Overview

The main goal of this project is to build a machine learning model that predicts whether a customer will churn (`yes` or `no`). The project workflow includes data preprocessing, feature engineering, class imbalance handling, model training, and evaluation.

## Dataset

The dataset includes the following key features:

- **state**: The U.S. state where the customer is located.
- **account length**: Number of days the account has been active.
- **area code**: The area code of the customer.
- **international plan**: Whether the customer has an international plan (`yes` or `no`).
- **voice mail plan**: Whether the customer has a voice mail plan (`yes` or `no`).
- **number of vmail messages**: Number of voice mail messages.
- **total day minutes**: Total minutes of calls during the day.
- **total day calls**: Total number of calls during the day.
- **total day charge**: Total charge for calls during the day.
- **total eve minutes**: Total minutes of calls in the evening.
- **total night minutes**: Total minutes of calls at night.
- **total international minutes**: Total minutes of international calls.
- **customer service calls**: Number of calls made to customer service.
- **churn**: Whether the customer churned (`yes` or `no`).

## Project Workflow

1. **Data Preprocessing**:
   - Handled missing values and dropped irrelevant features (e.g., `phone number`).
   - Applied one-hot encoding to categorical variables like `international plan` and `voice mail plan`.
   - Scaled numerical features for machine learning model compatibility.

2. **Feature Engineering**:
   - Focused on key features impacting churn: `total day minutes`, `total day charge`, `total evening minutes`, `total evening charge`, `customer service calls`, `international plan`, and `voice mail plan`.

3. **Class Imbalance Handling**:
   - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

4. **Modeling**:
   - Various machine learning models were tested:
     - **Random Forest**: Achieved ~94% accuracy.
     - **Gradient Boosting**: Achieved ~96% accuracy with an AUC score of 0.98.

5. **Model Evaluation**:
   - Evaluated models based on:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - ROC-AUC Score

   - AUC-ROC curve for the Gradient Boosting model is shown below, with an AUC score of 0.98.

## Installation & Requirements

Install the necessary dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

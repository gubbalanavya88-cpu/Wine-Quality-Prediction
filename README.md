# 🍷 Wine Quality Prediction

## 📌 Project Overview

Wine Quality Prediction is a Machine Learning project that predicts the quality of red wine based on different chemical properties of the wine.

The project uses chemical information such as:

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol

The dataset is preprocessed using data cleaning, duplicate removal, feature transformation, SMOTE, and feature scaling.

Multiple Machine Learning classification algorithms were trained and evaluated. Logistic Regression and Random Forest were used for prediction, and the Random Forest model was saved for deployment.

The trained model is integrated into a Streamlit web application.

The application allows users to enter wine chemical properties and get the predicted wine quality.

---

## 🎯 Project Objective

The main objective of this project is to:

* Understand and analyze wine quality data.
* Perform Exploratory Data Analysis (EDA).
* Clean and preprocess the dataset.
* Remove duplicate records.
* Apply feature transformation using logarithmic transformation.
* Handle class imbalance using SMOTE.
* Scale features using StandardScaler.
* Train Machine Learning classification models.
* Compare model performance using Accuracy.
* Save the trained Random Forest model and scaler.
* Build an interactive web application using Streamlit.
* Deploy the Machine Learning application online.

---

## 📂 Dataset

The project uses the **Wine Quality Red Dataset**, which contains information about the chemical properties of red wine and its corresponding quality score.

### Features

| **Feature**            | **Description**              |
| ---------------------- | ---------------------------- |
| `fixed acidity`        | Fixed acidity of the wine    |
| `volatile acidity`     | Volatile acidity of the wine |
| `citric acid`          | Citric acid content          |
| `residual sugar`       | Amount of residual sugar     |
| `chlorides`            | Chloride concentration       |
| `free sulfur dioxide`  | Free sulfur dioxide level    |
| `total sulfur dioxide` | Total sulfur dioxide level   |
| `density`              | Density of the wine          |
| `pH`                   | pH value of the wine         |
| `sulphates`            | Sulphate concentration       |
| `alcohol`              | Alcohol percentage           |

### Target Variable

The target variable used for prediction is:

```text
quality
```

The `quality` value represents the quality score of the wine.

---

## 🔄 Data Preprocessing

The dataset is first loaded and analyzed using Pandas.

The following preprocessing steps are performed:

* Checking the dataset shape and columns.
* Checking data types and statistical information.
* Checking missing values.
* Removing duplicate records.
* Analyzing the distribution of wine quality.

---

## 🔄 Feature Transformation

Logarithmic transformation is applied to selected features:

* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Sulphates

This transformation is applied to reduce skewness in selected features.

---

## ⚖️ Handling Class Imbalance

SMOTE (**Synthetic Minority Over-sampling Technique**) is used to handle the imbalance between different wine quality classes.

The SMOTE technique generates synthetic samples for minority classes and creates a more balanced training dataset.

---

## 📏 Feature Scaling

`StandardScaler` is used to scale the input features.

The scaler is fitted on the training data and then used to transform the test data.

This helps bring the features to a comparable scale before training the Machine Learning models.

---

## 🤖 Machine Learning Models

The following classification algorithms are used:

### 1. Logistic Regression

Logistic Regression is trained to classify the wine quality.

### 2. Random Forest Classifier

Random Forest Classifier is trained using the preprocessed and scaled training data.

The Random Forest model is saved and used in the Streamlit application for prediction.

---

## 💾 Model Saving

The trained Random Forest model and StandardScaler are saved using Python's built-in `pickle` module.

The saved files are:

```text
model_rf.pkl
scaler.pkl
```

These files are loaded by the Streamlit application during prediction.

---

## 🌐 Streamlit Application

A Streamlit web application is created for wine quality prediction.

Users can enter the following wine properties:

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol

After entering the values and clicking **Predict Wine Quality**, the application displays the predicted wine quality.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Streamlit
* Pickle

---

## 📂 Project Files

```text
wine-quality-prediction/
│
├── app.py
├── winequality-red.csv
├── model_rf.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Deployment

The Streamlit application is deployed online using Streamlit Community Cloud.

The application can be accessed through the deployed Streamlit link.

---



# 🍷 Wine Quality Prediction

## 📌 Project Overview

Wine Quality Prediction is a Machine Learning project that predicts the quality of red wine based on its chemical properties.

The project uses the **Wine Quality Red Dataset** and applies data preprocessing, feature transformation, feature scaling, SMOTE for handling class imbalance, and Machine Learning classification algorithms.

A **Streamlit web application** is also created so users can enter wine properties and get a predicted wine quality.

---

## 🎯 Objective

The main objective of this project is to:

* Analyze the chemical properties of red wine.
* Preprocess and clean the dataset.
* Transform selected features using logarithmic transformation.
* Handle imbalanced quality classes using SMOTE.
* Scale the features using StandardScaler.
* Train Machine Learning classification models.
* Compare Logistic Regression and Random Forest.
* Save the trained Random Forest model and scaler.
* Deploy the prediction application using Streamlit.

---

## 📂 Project Structure

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

### Files Description

| File                  | Description                                       |
| --------------------- | ------------------------------------------------- |
| `app.py`              | Streamlit application for wine quality prediction |
| `winequality-red.csv` | Dataset used for training                         |
| `model_rf.pkl`        | Saved Random Forest model                         |
| `scaler.pkl`          | Saved StandardScaler                              |
| `requirements.txt`    | Required Python libraries                         |
| `README.md`           | Project documentation                             |

---

## 📊 Dataset

The project uses the **Red Wine Quality Dataset**.

The dataset contains chemical properties of red wine and a target variable called `quality`.

### Input Features

The model uses the following 11 features:

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

### Target

```text
quality
```

The target represents the quality score of the wine.

---

# 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Remove Duplicates
   ↓
Exploratory Data Analysis
   ↓
Feature Transformation
   ↓
Separate X and y
   ↓
Train-Test Split
   ↓
SMOTE
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model + Scaler
   ↓
Streamlit Application
   ↓
Prediction
```

---

# 🧹 Data Preprocessing

## 1. Load Dataset

The dataset is loaded using Pandas.

```python
data = pd.read_csv("winequality-red.csv")
```

## 2. Check Dataset

The dataset is examined using:

```python
data.head()
data.shape
data.columns
data.info()
data.describe()
data.isnull().sum()
```

These functions help understand the dataset, its columns, data types, statistics, and missing values.

## 3. Remove Duplicate Records

Duplicate rows are removed using:

```python
data = data.drop_duplicates()
```

This helps avoid repeated records during model training.

## 4. Check Missing Values

Missing values are checked using:

```python
data.isnull().sum()
```

---

# 🔄 Feature Transformation

Some numerical features can have skewed distributions.

Therefore, logarithmic transformation is applied to selected features:

```python
data["residual sugar"] = np.log(data["residual sugar"])
data["chlorides"] = np.log(data["chlorides"])
data["free sulfur dioxide"] = np.log(data["free sulfur dioxide"])
data["total sulfur dioxide"] = np.log(data["total sulfur dioxide"])
data["sulphates"] = np.log(data["sulphates"])
```

### Why Log Transformation?

Log transformation is used to reduce skewness and make the distribution of selected features more suitable for Machine Learning.

---

# 🎯 Feature and Target Separation

The input features are separated from the target variable.

```python
X = data.drop("quality", axis=1)
y = data["quality"]
```

Where:

* `X` = input features
* `y` = target variable (`quality`)

---

# ✂️ Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)
```

### Split

* **80%** → Training data
* **20%** → Testing data

The training data is used to train the models, while the testing data is used to evaluate their performance.

---

# ⚖️ Handling Class Imbalance with SMOTE

The wine quality classes are not equally distributed.

To handle class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

SMOTE creates synthetic samples for minority classes instead of simply duplicating existing samples.

This helps provide a more balanced training dataset.

---

# 📏 Feature Scaling

StandardScaler is used to scale the features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_fit = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)
```

### Why Scaling?

The input features have different numerical ranges.

For example:

```text
Age-like values → small range
Sulfur dioxide → larger range
```

StandardScaler transforms the features to a common scale based on their mean and standard deviation.

### Important

The scaler is:

```python
scaler.fit_transform(X_resampled)
```

on training data, while the test data uses:

```python
scaler.transform(X_test)
```

This avoids fitting the scaler separately on the test data.

---

# 🤖 Machine Learning Models

Two classification models are trained.

## 1. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()

model_lr.fit(
    X_train_fit,
    y_resampled
)
```

The predictions are generated using:

```python
y_pred_lr = model_lr.predict(X_test)
```

Accuracy is calculated using:

```python
accuracy_lr = accuracy_score(
    y_test,
    y_pred_lr
)
```

---

## 2. Random Forest Classifier

Random Forest is also trained:

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()

model_rf.fit(
    X_train_fit,
    y_resampled
)
```

Predictions:

```python
y_pred_rf = model_rf.predict(X_test)
```

Accuracy:

```python
accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)
```

---

# 💾 Model Saving

The trained Random Forest model and StandardScaler are saved using Python's built-in `pickle` module.

```python
import pickle

pickle.dump(
    model_rf,
    open("model_rf.pkl", "wb")
)

pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)
```

These files are later loaded by the Streamlit application.

> **Note:** `pickle` is a built-in Python module, so `pickle` should **not** be added to `requirements.txt`.

---

# 🌐 Streamlit Application

The project includes a Streamlit web application.

Users can enter:

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

After clicking:

```text
Predict Wine Quality
```

the application processes the input and displays the predicted wine quality.

---

# 🔮 Prediction Process

The Streamlit application follows these steps:

```text
User Input
    ↓
Convert to NumPy Array
    ↓
Reshape Input
    ↓
Load Saved Scaler
    ↓
Scale Input
    ↓
Load Random Forest Model
    ↓
Predict Quality
    ↓
Display Result
```

The saved model is loaded using:

```python
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)
```

The saved scaler is loaded using:

```python
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
```

---

# 📦 Requirements

The main libraries required for this project are:

```text
streamlit
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

`pickle` is **not** included because it is part of Python's standard library.

---

# ▶️ How to Run Locally

## Step 1: Clone the Repository

```bash
git clone https://github.com/gubbalanavya88-cpu/wine-quality-prediction.git
```

## Step 2: Open the Project

```bash
cd wine-quality-prediction
```

## Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

## Step 4: Run Streamlit

```bash
streamlit run app.py
```

The application will open in the browser.

---

# 🚀 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment steps:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Select the repository.
5. Select `app.py` as the main file.
6. Deploy the application.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Imbalanced-learn**
* **Random Forest**
* **Logistic Regression**
* **Streamlit**
* **GitHub**

---

# 📌 Key Concepts Used

### Data Preprocessing

Cleaning and preparing raw data for Machine Learning.

### Feature Transformation

Applying mathematical transformations such as logarithmic transformation to selected features.

### Feature Scaling

Bringing numerical features to a comparable scale using StandardScaler.

### SMOTE

Handling class imbalance by generating synthetic minority samples.

### Classification

Predicting wine quality classes using Machine Learning algorithms.

---

# 🎓 Conclusion

This project demonstrates an end-to-end Machine Learning workflow for wine quality prediction.

The workflow includes:

**Data Collection → Data Preprocessing → Feature Transformation → SMOTE → Feature Scaling → Model Training → Evaluation → Model Saving → Streamlit Deployment**

The trained Random Forest model is integrated into a Streamlit application, allowing users to enter wine chemical properties and receive a predicted wine quality.

---



B.Tech – Artificial Intelligence / Computer Science related field

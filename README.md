# 🩺 Breast Cancer Prediction Using Machine Learning

This project is focused on building and evaluating multiple **machine learning models** to classify **breast tumors** as **malignant or benign** using the **Wisconsin Breast Cancer Dataset**.  
Multiple algorithms like Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, and Random Forest are compared to determine the best performer.

---

## 🔍 Project Overview

- ✅ Data Cleaning and Preprocessing
- 📊 Exploratory Data Analysis (EDA) using **Seaborn & Matplotlib**
- 📈 Feature Scaling with `StandardScaler`
- 🤖 Training 7 ML Models on the dataset
- 🔎 Evaluation using Confusion Matrix and Classification Report

---

## 🧠 Algorithms Used

| Algorithm                          | Accuracy (Training) |
|-----------------------------------|----------------------|
| Logistic Regression               | 90% |
| K-Nearest Neighbors (KNN)         | 91% |
| Support Vector Machine (Linear)   | 93% |
| Support Vector Machine (RBF)      | 94% |
| Gaussian Naive Bayes              | 92% |
| Decision Tree                     | 95% |
| Random Forest                     | 97% |

---

## 🚀 Model Training & Evaluation

- Trained 7 models using `train_test_split()`
- Evaluated with `confusion_matrix`, `classification_report`, and `accuracy_score`
- Compared **testing accuracy** of all models

```python
Model[6] Random Forest Classifier
Confusion Matrix:
[[TN, FP],
 [FN, TP]]
Testing Accuracy = 97.89%

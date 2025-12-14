# Rossmann Store Sales Prediction using XGBoost

## 📌 Project Overview

This project implements an end-to-end machine learning pipeline to predict daily sales for Rossmann drug stores using historical data.  
The solution is inspired by the **Rossmann Store Sales Kaggle competition** and focuses on building a **robust, interpretable, and scalable regression model**.

The project covers the complete workflow starting from raw data ingestion to feature engineering, exploratory data analysis (EDA), model training with gradient boosting, cross-validation, hyperparameter tuning, dimensionality analysis using PCA, and final prediction generation.

---

## 🎯 Problem Statement

Rossmann operates over 3,000 drug stores across Europe. Store managers must predict daily sales up to six weeks in advance. Sales are affected by several factors such as:

- Promotions  
- Competition distance and duration  
- Store type and assortment  
- Holidays and seasonality  
- Temporal patterns (day, week, month, year)

The objective of this project is to **predict daily sales accurately** using machine learning techniques, while handling real-world challenges such as missing data, categorical variables, and non-linear relationships.

---

## 🗂 Dataset

**Source:** Kaggle – Rossmann Store Sales  
https://www.kaggle.com/c/rossmann-store-sales

**Files Used:**
- `train.csv`
- `test.csv`
- `store.csv`
- `sample_submission.csv`

---

## 🧠 Approach & Methodology

### 1️⃣ Data Loading & Merging
- Training and test datasets are merged with store-level metadata.
- Ensures availability of competition, promotion, and store-type information.

### 2️⃣ Exploratory Data Analysis (EDA)
- Distribution analysis of sales  
- Correlation analysis between numerical features  
- Identification of skewness, seasonality, and feature relationships  

### 3️⃣ Feature Engineering
Custom features were created using domain knowledge:

**Date-based features**
- Year  
- Month  
- Day  
- WeekOfYear  

**Competition features**
- Months since competitor opened  

**Promotion features**
- Duration of Promo2  
- Whether Promo2 is active in the current month  

**Store availability**
- Rows with closed stores removed from training data  

### 4️⃣ Data Preprocessing
- Numerical features scaled using `MinMaxScaler`
- Categorical features encoded using **One-Hot Encoding**
- Missing values handled logically (e.g., competition distance)

---

## 📐 PCA (Principal Component Analysis)

PCA is used **analytically** in this project to:

- Understand variance distribution across engineered features  
- Identify redundancy and correlation in high-dimensional feature space  
- Demonstrate dimensionality reduction concepts  

ℹ️ PCA is **not used for final model training**, as tree-based models like XGBoost do not require feature scaling or orthogonal components.

---

## 🚀 Model Used

### Gradient Boosting with XGBoost (`XGBRegressor`)

**Why XGBoost?**
- Handles non-linear relationships efficiently  
- Robust to outliers  
- Built-in regularization  
- High performance on structured/tabular data  

---

## 🔁 Model Evaluation Strategy

### K-Fold Cross Validation
- 5-fold cross validation used to assess generalization  
- Prevents overfitting on a single validation split  

### Metric
**RMSE (Root Mean Squared Error)**  
Chosen because it penalizes large prediction errors more heavily.

---

## ⚙️ Hyperparameter Tuning

The following hyperparameters were tuned experimentally:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

The final model balances bias and variance while maintaining strong predictive performance.

---

## 📊 Feature Importance

XGBoost’s built-in feature importance was used to:

- Identify the most influential features affecting sales  
- Improve model interpretability  
- Validate feature engineering decisions  

**Top contributing features include:**
- Promotions  
- Competition-related features  
- Temporal variables (WeekOfYear, Month)  
- Store and assortment characteristics  

---

## 📦 Final Output

- Predictions generated for the test dataset  
- Adjustments applied to ensure `Sales = 0` when a store is closed  
  




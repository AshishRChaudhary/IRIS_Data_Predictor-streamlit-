# IRIS_Data_Predictor-streamlit-
---

# 🌸 IRIS Data Analysis and Predictor

This project explores the well-known **IRIS dataset**, focusing on understanding its structure, cleaning and preparing the data, building classification models, and predicting flower species using an interactive web application.

Developed with a modular structure and a user-friendly interface, the application enables seamless exploration of data science workflows — from raw data inspection to intelligent prediction — all within a single, dynamic platform.

---

## 🔍 Key Features

### 1️⃣ View Data

* Display raw dataset sample
* Summary statistics
* Null value count
* Feature and target overview

### 2️⃣ Preprocessing

* Handle missing and zero values via median imputation
* Remove duplicate records
* Interactive data visualizations:

  * Boxplots with insights
  * Histograms
  * Correlation heatmaps
* User-controlled visualization options with dropdowns and feature selectors

### 3️⃣ Train and Evaluate

* Multiple classifier options: Logistic Regression, Random Forest, and more
* Train/test split with adjustable ratio
* Model accuracy, confusion matrix, and classification report
* Cross-validation and overfitting control
* Clean integration with preprocessing pipeline

### 4️⃣ Prediction

* Real-time prediction based on user-input features
* Model confidence displayed with class probabilities
* Select your preferred model for prediction
* Works with preprocessed and retrained data

---

## 📁 Project Structure

```
IRIS_Data_Analysis_and_Predictor/
│
├── app.py                # Main Streamlit app  
├── data_analysis.py      # Data handling, cleaning, and visualization  
├── prediction.py         # Model training, evaluation, and prediction logic  
├── files/                # (Optional) Folder for saved models or additional files  
└── requirements.txt      # Python dependencies  
```

---

## 🚀 Getting Started

### ✅ Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📚 What You’ll Learn

* Data inspection and cleaning techniques
* Visual data analysis using Seaborn and Matplotlib
* Building ML models with Scikit-learn
* Evaluating model performance
* Delivering a complete data science pipeline via a web interface

Whether you're refining your data skills or showcasing your workflow with interactive dashboards, this project brings together multiple essential concepts in one compact environment.

---

## 📦 Future Enhancements

* Model persistence with `joblib` (saving and loading trained models)
* Upload CSV for batch predictions
* Add more ML models (e.g., SVM, KNN)
* Expand visual options (e.g., pairplots, violin plots)

---

## 🙌 Credits

* Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
* Libraries: `pandas`, `numpy`, `seaborn`, `scikit-learn`, `matplotlib`, `streamlit`

---

## 🎓 Learning Purpose

This project was built purely for **learning and educational purposes**, with the goal of gaining hands-on experience in the **Data Science pipeline** — from raw data to model predictions — while also building an **interactive and modular web application using Streamlit**.


import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_analysis import get_cleaned_data

def train_and_evaluate_section(df,iris):

    st.subheader("🧠 Train and Evaluate Model")
    df,iris=get_cleaned_data() 

    # Prepare data
    X = df[iris.feature_names]
    y = df['target']

    test_size = st.slider("Test set size (%)", 10, 50, step=5, value=20)

    model_option = st.selectbox("Select Model", [
        "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors",
        "Decision Tree", "Random Forest"
    ])

    # Display model-specific hyperparameters
    params = dict()
    if model_option == "Support Vector Machine":
        params['C'] = st.slider("Regularization (C)", 0.01, 10.0, value=1.0)
        params['kernel'] = st.selectbox("Kernel", ['linear', 'rbf', 'poly'])
    elif model_option == "K-Nearest Neighbors":
        params['n_neighbors'] = st.slider("Number of Neighbors (k)", 1, 15, value=5)
    elif model_option == "Decision Tree":
        params['max_depth'] = st.slider("Max Depth", 1, 10, value=3)
    elif model_option == "Random Forest":
        params['n_estimators'] = st.slider("Number of Trees", 10, 200, value=100, step=10)
        params['max_depth'] = st.slider("Max Depth", 2, 10, value=5)

    if st.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

        # Model initialization
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_option == "Support Vector Machine":
            model = SVC(C=params['C'], kernel=params['kernel'])
        elif model_option == "K-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=params['max_depth'])
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

        # Cross-validation (5-fold)
        cv_score = cross_val_score(model, X, y, cv=5)
        st.info(f"📊 5-Fold CV Accuracy: {cv_score.mean() * 100:.2f}% ± {cv_score.std() * 100:.2f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap="YlGnBu", axis=0))

def prediction_section(df,iris):
    st.subheader("🔮 Make a Prediction")

    df, iris = get_cleaned_data()
    X = df[iris.feature_names]
    y = df['target']

    # --- Let user input values ---
    st.write("### Enter Feature Values:")
    input_data = []
    for feature in iris.feature_names:
        val = st.slider(
            label=feature,
            min_value=float(X[feature].min()),
            max_value=float(X[feature].max()),
            value=float(X[feature].mean()),
            step=0.1
        )
        input_data.append(val)

    input_array = np.array(input_data).reshape(1, -1)
    
    model_option = st.selectbox("Select Model", [
        "Logistic Regression", "Random Forest"
    ])

    if st.button("🎯 Predict"):
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)

        model.fit(X, y)  # retrain on full preprocessed data
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0]

        st.success(f"🌼 Predicted Species: **{iris.target_names[prediction]}**")
        st.write("### 🔎 Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Species': iris.target_names,
            'Probability': proba
        })
        st.dataframe(prob_df.style.background_gradient(cmap="YlOrBr", axis=0))

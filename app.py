import streamlit as st
from data_analysis import view_data_section, preprocessing_section
from models import train_and_evaluate_section, prediction_section
from sklearn.datasets import load_iris



st.set_page_config(page_title="Iris Data Analyzer and Predictor", layout="centered")
st.title(" IRIS Data Analyzer ")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["View Data", "Preprocessing", "Train and Evaluate", "Prediction"])

#data loading
iris = load_iris(as_frame=True)
df= iris.frame.copy()
# Show only the selected section
if section == "View Data":
    view_data_section(df,iris)
elif section == "Preprocessing":
    st.subheader("🧹 Preprocessing")
    preprocessing_section(df)
elif section == "Train and Evaluate":
    st.subheader("📊     Train and Evaluate")
    train_and_evaluate_section(df,iris)
elif section == "Prediction":
    st.subheader("🔮 Prediction")
    prediction_section(df,iris)
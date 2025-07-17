import streamlit as st
import numpy as np
import pandas as pd    
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def view_data_section(df,iris):
    st.subheader(" View Data")
    # Feature names
    st.write("### Feature Names")
    st.write(iris.feature_names)

    # Target classes
    st.write("### Target Classes")
    st.write(iris.target_names.tolist())

    # Display sample of dataset
    st.write("### Sample of Dataset")
    st.dataframe(df.head())

    # Dataset shape
    st.write("### Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Summary statistics
    st.write("### Summary Statistics")
    st.dataframe(df.describe())

    # Null values
    st.write("### Null Value Check")
    st.write(df.isnull().sum())

def visualize(df):
    df['species'] = df['target'].map({
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    })

    features = df.columns[:-2].tolist()  # Exclude target & species

    st.markdown("## 🎨 Interactive Visualizations")

    chart_type = st.selectbox(
        "Select Visualization Type",
        ["Boxplot", "Histogram", "Correlation Heatmap", "Pairplot", "Scatterplot"]
    )

    if chart_type == "Boxplot":
        selected_feature = st.selectbox("Select Feature", features)
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='species', y=selected_feature, hue='species', palette='Set2', ax=ax, legend=False)
        sns.stripplot(data=df, x='species', y=selected_feature, color='black', size=3, jitter=0.2, ax=ax)
        ax.set_title(f'Boxplot of {selected_feature} by Species')
        st.pyplot(fig)
        st.info("Boxplot shows the distribution and spread. Black dots are actual data points (stripplot).")

    elif chart_type == "Histogram":
        selected_feature = st.selectbox("Select Feature", features)
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=selected_feature, hue='species', kde=True, palette='husl', ax=ax)
        ax.set_title(f'Histogram of {selected_feature}')
        st.pyplot(fig)
        st.info("Histogram shows the distribution and overlap of values among species.")

    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        st.info("Strong correlation between petal length and width; helps in feature selection.")

    elif chart_type == "Pairplot":
        diag_kind = st.selectbox("Diagonal Plot Type", ["hist", "kde"])
        fig = sns.pairplot(df, hue='species', corner=True, diag_kind=diag_kind, palette='Set1')
        st.pyplot(fig)
        st.info("Useful to visually separate classes using feature combinations.")

    elif chart_type == "Scatterplot":
        x_feature = st.selectbox("X-axis Feature", features, key="scatter_x")
        y_feature = st.selectbox("Y-axis Feature", df.columns.tolist(), key="scatter_y")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='species', palette='Set1', ax=ax)
        ax.set_title(f'{x_feature} vs {y_feature}')
        st.pyplot(fig)
        st.info("Scatterplots show feature relationships and help spot separation between classes.")

def preprocessing_section(df):
    st.write("### Original Data Preview")
    st.dataframe(df.head())

    st.write("### Missing Value Count (Before Imputation)")
    st.write(df.isnull().sum())

    # Impute missing values with median
    df.fillna(df.median(), inplace=True)

    st.write("### Missing Value Count (After Imputation)")
    st.write(df.isnull().sum())

    # Remove duplicates
    before = df.shape[0]
    df.drop_duplicates()
    after = df.shape[0]

    st.write(f"### Removed {before - after} duplicate rows")
    st.write("### Cleaned Data Preview")
    st.dataframe(df.head())

    if st.checkbox("📊 Show Basic Visualizations"):
        visualize(df)

def get_cleaned_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    df.fillna(df.median(), inplace=True)
    df.drop_duplicates(inplace=True)
    return df,iris





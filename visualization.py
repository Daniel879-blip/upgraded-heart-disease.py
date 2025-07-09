import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Set Seaborn style
sns.set(style="whitegrid")

def show_class_distribution(df):
    st.subheader("Class Distribution (Heart Disease)")
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, hue='target', palette='viridis', ax=ax, legend=False)
    ax.set_title('Distribution of Target Classes (0 = No Disease, 1 = Disease)')
    ax.set_xlabel('Target Class')
    ax.set_ylabel('Count')
    st.pyplot(fig)

def show_correlation_heatmap(df):
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

def show_feature_distributions(df):
    st.subheader("Feature Distributions")
    selected_features = st.multiselect(
        "Select features to display distribution plots:",
        options=df.columns.tolist(),
        default=["age", "chol", "thalach"]
    )
    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {feature}')
        st.pyplot(fig)

def show_pairplot(df):
    st.subheader("Pairplot of Selected Features")
    selected_features = st.multiselect(
        "Select features for pairplot (max 5):",
        options=df.columns.tolist(),
        default=["age", "thalach", "chol", "cp", "target"]
    )
    if len(selected_features) > 1:
        fig = sns.pairplot(df[selected_features], hue="target", palette="viridis", diag_kind="kde")
        st.pyplot(fig)
    else:
        st.warning("Please select at least 2 features to generate a pairplot.")

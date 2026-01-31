import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#importing python file functions.
from logistic import logistic_model
from decision_tree import decision_classifier
from knn import knn_classifier
from naiveBayes import nb_classifier
from random_forest import rf_classifier
from XGB import xgb_classifier
from classification import preprocess_data

st.set_page_config(page_title="MachineLearning Model Eval App", layout="wide")
st.title("📈ML Model Evaluation Dashboard")

st.markdown(
    """
    Upload **test data only (CSV)** and select different classification models from dropdown menu.
    """
)

# (a) Dataset upload (CSV)
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Test Dataset (CSV)",
    type=["csv"]
)
result = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data set uploaded successfully!")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    X, y, target_column = preprocess_data(df, target_column=None)
    
    # --------------------------------------------------
    # (b) Model selection dropdown
    # --------------------------------------------------
    model_name = st.selectbox(
        "Select Classification Model",
        [
            "Select the Model",
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Navie Bayes Classification",
            "Random Forest",
            "XGB Classifier"
        ]
    )
    y_test = None
    y_pred = None
    if model_name == "Logistic Regression":
        st.subheader("Logistic Regression")
        output = logistic_model(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    elif model_name == "Decision Tree":
        st.subheader("Decision Tree")
        output = decision_classifier(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    elif model_name == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors")
        output = knn_classifier(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    elif model_name == "Navie Bayes Classification":
        st.subheader("Navie Bayes Classification")
        output = nb_classifier(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    elif model_name == "Random Forest":
        st.subheader("Random Forest")
        output = rf_classifier(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    elif model_name == "XGB Classifier":
        st.subheader("XGB Classifier")
        output = xgb_classifier(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    
    if model_name != "Select the Model":#printing model 
        #printing confusion matrix.
        st.subheader("Reports and matrix")
        col1,spacer, col2 = st.columns([1,0.3,2,0.3,3])
        cm = confusion_matrix(y_test, y_pred)
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.subheader("Confusion Matrix")
            st.pyplot(fig)

        with col2:
            #Classification Report
            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test, y_pred))
        with col3:
            st.subheader("📈 Evaluation Matrix")
            if result is not None:
                dataf = pd.DataFrame(
                result.items(),
                columns=["Metric", "Value"]
                )
                st.dataframe(dataf)        
       

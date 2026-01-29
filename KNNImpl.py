import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_squared_error
)

st.set_page_config(page_title="KNN Classifier & Regressor", layout="wide")
st.title("KNN Model Trainer")
st.sidebar.header("⚙️ Configuration")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["KNN Classifier", "KNN Regressor"]
)

k = st.sidebar.slider("Number of Neighbors (K)", 1, 10, 3)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

if model_type == "KNN Classifier":
    X, y = make_classification(
        n_samples=1000,
        n_features=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
else:
    X, y = make_regression(
        n_samples=1000,
        n_features=3,
        noise=10,
        random_state=42
    )

df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
df["Target"] = y

st.subheader("Dataset Preview")
st.dataframe(df.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
if st.button("Train Model"):

    if model_type == "KNN Classifier":
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(" Model Performance")
        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

        st.subheader(" Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader(" Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

    else:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(" Model Performance")
        st.write("**R² Score:**", r2_score(y_test, y_pred))
        st.write("**Mean Squared Error:**", mean_squared_error(y_test, y_pred))

        st.subheader(" Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
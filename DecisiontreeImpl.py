import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Iris Decision Tree", layout="wide")
st.title("Iris Dataset – Decision Tree Classifier")

iris = load_iris()
X = iris.data
y = iris.target 

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

st.sidebar.header("Model Settings")

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)

use_grid = st.sidebar.checkbox("Use GridSearchCV")

st.subheader("Iris Dataset")
st.dataframe(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

if st.button("Train Model"):
    
    if not use_grid:
        dt = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=42
        )
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        st.subheader("Model Performance")
        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
        st.pyplot(fig)

    else:
        st.info("Running GridSearchCV... ")

        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2']
        }

        dt = DecisionTreeClassifier(random_state=42)
        grid = GridSearchCV(dt, param_grid, cv=5)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        st.subheader("Best Parameters")
        st.json(grid.best_params_)

        st.subheader("Model Performance")
        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Best Decision Tree")
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(best_model, filled=True,
                  feature_names=iris.feature_names,
                  class_names=iris.target_names)
        st.pyplot(fig)
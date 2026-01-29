import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

st.title("Naive Bayes Classifier - Iris Dataset")

iris = load_iris()
X = iris.data
Y = iris.target

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Naive Bayes Model",
    ("Gaussian Naive Bayes", "Bernoulli Naive Bayes")
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=test_size, random_state=42
)

if model_choice == "Gaussian Naive Bayes":
    model = GaussianNB()
else:
    model = BernoulliNB()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader(" Model Performance")
st.write(f"**Accuracy Score:** {accuracy:.2f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))
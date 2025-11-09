import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris
from comet_ml import Experiment

# Initialize Comet experiment
experiment = Experiment(
    api_key="hUBFLPDx011lDAfMssveLX3vl",       # 🔑 or leave blank to use environment variable
    project_name="iris-demo",
    workspace="anush-g-7825"
)

# Load model and data
model = joblib.load("model.pkl")
iris = load_iris()

st.title("🌸 Iris Flower Classifier")
st.write("This app predicts the species of an Iris flower using your trained model and logs predictions to Comet.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    predicted_class = iris.target_names[prediction]

    # Display result
    st.success(f"🌼 Predicted Iris species: **{predicted_class.capitalize()}**")

    # Log to Comet
    experiment.log_metric("prediction_index", int(prediction))
    experiment.log_text(f"Predicted class: {predicted_class}")
    experiment.log_table("inputs", {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
        "predicted_class": predicted_class
    })

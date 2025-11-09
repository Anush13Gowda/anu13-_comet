import streamlit as st
import joblib
import numpy as np
import os
from comet_ml import Experiment

# ---------------------------
# Initialize Comet Experiment
# ---------------------------

api_key = os.getenv("COMET_API_KEY")

if api_key:
    experiment = Experiment(
        api_key=hUBFLPDx011lDAfMssveLX3vl,
        project_name="iris-classifier",
        workspace="anush-g"  # replace with your Comet workspace username
    )
    st.sidebar.success("✅ Connected to Comet successfully!")
else:
    experiment = None
    st.sidebar.warning("⚠️ COMET_API_KEY not found. Logging disabled.")

# ---------------------------
# Load Trained Model
# ---------------------------

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

# ---------------------------
# Streamlit App UI
# ---------------------------

st.title("🌸 Iris Flower Classifier")
st.write("This app predicts the species of an Iris flower using a trained model and logs experiments to Comet ML.")

st.header("Input Flower Measurements")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    # Prepare input data
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    classes = ["Setosa", "Versicolor", "Virginica"]
    result = classes[int(prediction[0])]

    # Display result
    st.success(f"🌼 Predicted Species: **{result}**")
    st.write("Prediction Probabilities:")
    st.write({classes[i]: round(prob, 3) for i, prob in enumerate(prediction_proba[0])})

    # ---------------------------
    # Log Data to Comet
    # ---------------------------
    if experiment:
        experiment.log_parameters({
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        })
        experiment.log_metric("predicted_class", int(prediction[0]))
        experiment.log_metric("Setosa_prob", float(prediction_proba[0][0]))
        experiment.log_metric("Versicolor_prob", float(prediction_proba[0][1]))
        experiment.log_metric("Virginica_prob", float(prediction_proba[0][2]))

        st.sidebar.info("📡 Prediction logged to Comet.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed with ❤️ | Powered by Streamlit & Comet ML")

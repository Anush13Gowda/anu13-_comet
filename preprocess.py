from comet_ml import Experiment
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Initialize Comet experiment
experiment = Experiment(
    project_name="iris-demo",
    workspace="anush-g-7825"  # 👈 your actual workspace
)

# Load and preprocess data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data
joblib.dump((X_scaled, y), "processed_data.pkl")
joblib.dump(scaler, "scaler.pkl")

# Log preprocessing details
experiment.log_other("num_features", X.shape[1])
experiment.log_other("num_samples", X.shape[0])
experiment.log_other("scaling", "StandardScaler")

print("✅ Data preprocessed and saved as processed_data.pkl")

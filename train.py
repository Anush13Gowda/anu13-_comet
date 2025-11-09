from comet_ml import Experiment
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Initialize Comet experiment
experiment = Experiment(
    project_name="iris-demo",
    workspace="anush-g-7825"  # 👈 your actual workspace name
)

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {accuracy:.2f}")

# Log to Comet
experiment.log_metric("accuracy", accuracy)
experiment.log_parameters(model.get_params())

# 💾 Save model to file
joblib.dump(model, "model.pkl")
print("💾 Model saved as model.pkl")

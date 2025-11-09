from comet_ml import Experiment
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Load saved model
model = joblib.load("model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log results to Comet
experiment = Experiment(
    project_name="iris-demo",
    workspace="anush-g-7825"  # 👈 replace with your actual workspace (e.g. "anush-g-7825")
)
experiment.log_metric("evaluation_accuracy", accuracy)

print(f"✅ Model evaluated successfully with accuracy: {accuracy:.2f}")

import subprocess

print("🚀 Starting Comet ML pipeline...")

# Step 1: Train model
subprocess.run(["python", "train.py"], check=True)

# Step 2: Evaluate model
subprocess.run(["python", "evaluate.py"], check=True)

print("🎯 Pipeline finished successfully!")

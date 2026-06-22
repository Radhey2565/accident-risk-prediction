import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Generate synthetic dataset
np.random.seed(42)

n = 10000

speed = np.random.randint(20, 150, n)
weather = np.random.randint(0, 4, n)
road_type = np.random.randint(0, 3, n)
traffic = np.random.randint(0, 100, n)

risk = (
    (speed > 100) |
    (weather >= 2) |
    (traffic > 70)
).astype(int)

df = pd.DataFrame({
    "Speed": speed,
    "Weather": weather,
    "RoadType": road_type,
    "TrafficDensity": traffic,
    "Risk": risk
})

X = df.drop("Risk", axis=1)
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "risk_model.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("Saved to:", MODEL_PATH)

print("Model Saved Successfully") 
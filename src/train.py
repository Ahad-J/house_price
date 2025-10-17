import os
import re
import json
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Helper function
# --------------------------
def slugify(name):
    """Convert feature names to safe field identifiers."""
    name = str(name).strip()
    return re.sub(r'[^0-9a-zA-Z_]+', '_', name)

# --------------------------
# Load parameters
# --------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]
n_estimators = params["model"]["n_estimators"]

# --------------------------
# Load and clean data
# --------------------------
df = pd.read_csv("data/Entities.csv")  
df_clean = df.dropna(axis=0, how='any').reset_index(drop=True)
df_clean = df_clean.iloc[:, [4,5,7,8,9,10,11,12,13,17]]  # select relevant columns

# Encode categorical features
string_cols = ["property_type", "city", "province_name", "purpose"]
encoders = {}
for col in string_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    encoders[col] = le

# --------------------------
# Split data
# --------------------------
X = df_clean.drop("price", axis=1)
y = df_clean["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# --------------------------
# Train model
# --------------------------
model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)

# --------------------------
# Evaluate
# --------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R²: {r2}")

# --------------------------
# Save artifacts
# --------------------------
OUTPUT_DIR = "models/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

features_list = X.columns
feature_field_map = {col: slugify(col) for col in features_list}

joblib.dump(model, os.path.join(OUTPUT_DIR, "house_price_model.pkl"))
joblib.dump(features_list, os.path.join(OUTPUT_DIR, "model_features.pkl"))
joblib.dump(encoders, os.path.join(OUTPUT_DIR, "label_encoders.pkl"))
joblib.dump(feature_field_map, os.path.join(OUTPUT_DIR, "feature_field_map.pkl"))

print(f"Saved model and artifacts to {OUTPUT_DIR}")
print("Files:")
for f in os.listdir(OUTPUT_DIR):
    print(" -", f)

# --------------------------
# Save metrics for DVC tracking
# --------------------------
metrics = {"mse": float(mse), "r2": float(r2)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
print("✅ Metrics saved to metrics.json")


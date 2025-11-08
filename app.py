from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ======================
# 1. データ読み込み＆学習
# ======================
df = pd.read_csv("training_features.csv")

X = df[["mean_acc", "std_acc", "max_acc", "min_acc", "energy"]]
y = df["label_id"]
label_map = dict(zip(df["label_id"], df["label"]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ 正解率: {acc:.4f}")

joblib.dump(model, "model.pkl")

# ======================
# 2. Flask API
# ======================
app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def index():
    return "Activity Recognition API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON input:
    {
      "features": [mean_acc, std_acc, max_acc, min_acc, energy]
    }
    """
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        pred_id = int(model.predict(features)[0])
        pred_label = label_map[pred_id]
        return jsonify({
            "label_id": pred_id,
            "label": pred_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

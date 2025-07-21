from flask import Flask, request, jsonify
import joblib
import requests

# === Load ML model ===
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("cv_encoder.pkl")

# === Telegram Bot Token ===
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = Flask(__name__)

# Health check
@app.route("/", methods=["GET"])
def home():
    return {"status": "ok", "message": "✅ Sentiment API + Telegram webhook running!"}

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    return jsonify({
        "prediction": str(pred),
        "confidence": round(float(prob), 2)
    })

# === Telegram webhook route ===
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    data = request.json
    chat_id = data["message"]["chat"]["id"]
    text = data["message"]["text"]

    # Run prediction
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    reply = f"Prediction: {pred} (Confidence: {prob:.2f})"

    # Send reply back to Telegram
    requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": reply})
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


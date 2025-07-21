from flask import Flask, request, jsonify, render_template_string
import joblib

# Load model & vectorizer
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("cv_encoder.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {
        "status": "ok",
        "message": "✅ News Sentiment API is running! Use POST /predict"
    }

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

@app.route("/test", methods=["GET"])
def test_form():
    html = """
    <h1>📰 News Sentiment API Test</h1>
    <form method="post" action="/predict" onsubmit="event.preventDefault();predict()">
      <textarea id="input" rows="4" cols="50" placeholder="Type your news headline here"></textarea><br><br>
      <button type="submit">Predict Sentiment</button>
    </form>
    <p id="result"></p>
    <script>
      async function predict() {
        const text = document.getElementById("input").value;
        const res = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text: text})
        });
        const data = await res.json();
        document.getElementById("result").innerHTML = JSON.stringify(data);
      }
    </script>
    """
    return render_template_string(html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

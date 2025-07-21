import os
from flask import Flask, request
from dotenv import load_dotenv
import requests
import joblib

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
WEBHOOK_URL = os.getenv('WEBHOOK_URL')
url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/'

encoder = joblib.load("cv_encoder.pkl")
model = joblib.load("lr_model.pkl")

app = Flask(__name__)

with app.app_context():
    # Delete old webhook
    delete_webhook_url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook'
    requests.post(delete_webhook_url, json={'drop_pending_updates': True})

    # Set new webhook properly
    set_webhook_url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook'
    response = requests.post(set_webhook_url, json={'url': f'{WEBHOOK_URL}/webhook'})
    print(f"Webhook set: {response.json()}")

@app.route('/', methods=['GET'])
def index():
    return 'Flask is running...'

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    data = request.get_json()
    message = data.get('message', {})
    chat_id = message.get('chat', {}).get('id')
    message_text = message.get('text', '')

    if not chat_id:
        return 'No chat ID', 400

    if message_text.lower() == '/start':
        requests.get(url + f'sendMessage?chat_id={chat_id}&text=Send me some text: (or type /quit)')
        return 'OK', 200
    elif message_text.lower() == '/quit':
        requests.get(url + f'sendMessage?chat_id={chat_id}&text=Good Bye!')
        return 'OK', 200
    else:
        requests.get(url + f'sendChatAction?chat_id={chat_id}&action=typing')

        X_emb = encoder.transform([message_text])
        pred = model.predict(X_emb)

        result = "Not Spam" if pred[0] == "ham" else "Spam"

        requests.get(url + f'sendMessage?parse_mode=markdown&chat_id={chat_id}&text={result}')
        requests.get(url + f'sendMessage?parse_mode=markdown&chat_id={chat_id}&text=Send me some text: (or type /quit)')
        return 'OK', 200

# For Render, you don't necessarily need this, but if you want local debug:
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

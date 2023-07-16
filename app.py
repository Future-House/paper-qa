from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    chat_log = request.get_json()
    last_question = chat_log[-1]['user'] if chat_log else ''
    return f"Response to: {last_question}", 200

if __name__ == "__main__":
    app.run(debug=True, port=5001)

# app_paper_qa.py

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    chat_log = request.get_json(force=True)  # extract the chat log from the POST request
    question = chat_log[-1]['user'] if chat_log else ""
    response = f"Response to: {question}"
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5001)

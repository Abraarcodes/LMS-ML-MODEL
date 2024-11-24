from flask import Flask, request, jsonify
from evaluate import evaluate_answers
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.json  # Get the data from frontend (questions and answers)
    result = evaluate_answers(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys
import chatbot

# sys.path.insert(0, '/home/amninder/Desktop/Folder_2')
dataset = pd.read_csv("dataset_with_predicted_intents.csv")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    # Get user input from the request
    user_input = request.form['user_input']

    # Call your chatbot function with the user input
    bot_response = chatbot_response(user_input)

    # Return the bot response as JSON
    return jsonify({'bot_response': bot_response})

def chatbot_response(user_input):
    responses = []
    response = get_response(user_input, dataset)
    if response:
        responses.extend(response)

    if not responses:
        return "Sorry, I do not understand the question."

    return responses

if __name__ == '__main__':
    app.run(debug=True)
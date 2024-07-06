from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import ast
import re
from collections import Counter
from nltk.corpus import stopwords
import json
import nltk
from nltk.stem import WordNetLemmatizer
import uuid
from flask_migrate import Migrate

lemmatizer = WordNetLemmatizer()
import random

from keras.preprocessing.sequence import pad_sequences
import pickle

from keras.models import load_model

model = load_model("modelN.keras")

import numpy as np

dataset = pd.read_csv("dataset_with_predicted_intents.csv")

app = Flask(__name__)

# Database Config ---------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ws50.db"
app.secret_key = b'_17dy2p"Fh7aw\nj8a]/'

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define a model
class UserType(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type_name = db.Column(db.String(80), unique=True, nullable=False)

    def __repr__(self):
        return f"<UserType {self.type_name}>"


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    user_type = db.Column(db.Integer, db.ForeignKey("user_type.id"), nullable=False)

    user_type_relation = db.relationship(
        "UserType", backref=db.backref("users", lazy=True)
    )

    def __repr__(self):
        return f"<User {self.username}>"


class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    context = db.Column(db.String(120), nullable=False)
    response = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    session_id = db.Column(db.String(120), nullable=False)
    feedback = db.Column(db.Integer)
    predicted_intent = db.Column(db.String(120), nullable=False ,default="unknown")

    user_relation = db.relationship(
        "User", backref=db.backref("chat_history", lazy=True)
    )

    def __repr__(self):
        return f"<ChatHistory {self.id} for User {self.user_relation.username}>"


@app.route("/init_db")
def init_db():
    with app.app_context():
        db.create_all()
        if not UserType.query.first():
            admin = UserType(type_name="Admin")
            regular = UserType(type_name="Regular")
            db.session.add(admin)
            db.session.add(regular)
            db.session.commit()
    return "Database initialized successfully."


# ------------------------------------------------------


# Auth ------------------------------------------------
# Register User
@app.route("/register/new_user", methods=["POST"])
def register_new_user():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    username = data.get("name")
    email = data.get("email")
    password = data.get("password")
    user_type_name = "Regular"  # assuming a default user type for registration

    result = add_user(username, email, password, user_type_name)

    if "successfully" in result:
        return jsonify({"message": result}), 201
    else:
        return jsonify({"error": result}), 400


def add_user(username, email, password, user_type_name):
    try:
        with app.app_context():
            user_type = UserType.query.filter_by(type_name=user_type_name).first()
            if not user_type:
                return "User type not found."
            new_user = User(
                username=username,
                email=email,
                password=password,
                user_type=user_type.id,
            )
            db.session.add(new_user)
            db.session.commit()
        return "User added successfully."
    except Exception as e:
        return f"An error occurred: {e}"


# Login User
@app.route("/login", methods=["POST"])
def login_user():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    email = data.get("email")
    password = data.get("password")

    result = auth_user(email, password)

    if "successfully" in result:
        return jsonify({"message": result}), 201
    else:
        return jsonify({"error": result}), 400


def auth_user(email, password):
    try:
        user = User.query.filter_by(email=email).first()
        if not user:
            return "User not found."

        if not user.password == password:
            return "Incorrect password."

        session["user_id"] = user.id
        session["username"] = user.username
        session["user_type"] = user.user_type

        return "User authenticated successfully."
    except Exception as e:
        return f"An error occurred: {e}"


# User chat history
@app.route("/send_feedback")
def send_feedback():
    data = request.get_json()
    session_id= data.get("sessionid")
    response= data.get("response")
    feedback= data.get("feedback")

    if not data:
        return jsonify({"error": "Something has went wrong."}), 400
    
    chat_history = ChatHistory.query.filter_by(session_id=session_id).all()


# -----------------------------------------------------


# Page Routes ------------------------------------------
@app.route("/")
def home():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/chatbot")
def chatbot():
    session_id = str(uuid.uuid4())
    # Redirect to the URL with the session ID
    return redirect(url_for("chatbot_with_id", session_id=session_id))


@app.route("/chatbot/<session_id>")
def chatbot_with_id(session_id):
    return render_template("index.html", session_id=session_id)


# ---------------------------------------------------


# ChatBot API --------------------------------------
@app.route("/send_message", methods=["POST"])
def send_message():
    # Get user input from the request
    user_input = request.json.get("user_input")
    session_id = request.json.get("session_id")

    if not user_input or not session_id:
        return jsonify({"error": "No user input provided"}), 400
    
    chat_history = ChatHistory.query.filter_by(session_id=session_id).all()
    history_context = " ".join([f"{record.context}" for record in chat_history])

    history_keywords = keyword_extraction(history_context)

    predicted_intent = get_predicted_intent(user_input)
    bot_response = chatbot_response(user_input, history_keywords)

    if bot_response:
        try:
            with app.app_context():
                new_record = ChatHistory(
                    context=user_input,
                    response=bot_response[0],
                    user_id=session["user_id"],
                    session_id=session_id,
                    feedback= "",
                    predicted_intent= predicted_intent
                )
                db.session.add(new_record)
                db.session.commit()
        except Exception as e:
            return f"An error occurred: {e}"

    # Return the bot response as JSON
    return jsonify({"bot_response": bot_response, "history_context": history_keywords})


# ------------------------------------------------------


# Chatbot functions--------------------------------------
def chatbot_response(user_input, history_keywords):
    responses = []
    response = get_response(user_input, dataset, history_keywords)
    if response:
        responses.extend(response)

    if not responses:
        return "Sorry, I do not understand the question."

    return responses


def keyword_extraction(user_input):
    with open("keywords.json", "r") as file:
        data = json.load(file)
        mental_health_keywords = data["MENTAL_HEALTH_KEYWORDS"]

    user_input_tokens = user_input.lower().split()

    matching_keywords = []

    user_input_tokens = [re.sub(r"[^\w\s]", "", token) for token in user_input_tokens]

    for token in user_input_tokens:
        if token.lower() in mental_health_keywords:
            matching_keywords.append(token)

    if matching_keywords == []:
        return None

    return matching_keywords

def get_predicted_intent(user_input):
    words = pickle.load(open("texts.pkl", "rb"))
    labels = pickle.load(open("labels.pkl", "rb"))

    lemmatized_sentence = [
        lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)
    ]

    # Create bag-of-words representation
    input_bag = [1 if word in lemmatized_sentence else 0 for word in words]

    # Reshape input for prediction
    input_bag = np.array(input_bag).reshape(1, -1)

    # Make prediction
    predicted_probabilities = model.predict(input_bag)
    predicted_class_index = np.argmax(predicted_probabilities[0])
    predicted_class = labels[predicted_class_index]

    if predicted_class:
        return predicted_class
    
    return None    

def get_response(user_input, dataset, history_keywords):
    intents = json.loads(open("chatbot/intents.json").read())
    words = pickle.load(open("texts.pkl", "rb"))
    labels = pickle.load(open("labels.pkl", "rb"))

    lemmatized_sentence = [
        lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)
    ]

    # Create bag-of-words representation
    input_bag = [1 if word in lemmatized_sentence else 0 for word in words]

    # Reshape input for prediction
    input_bag = np.array(input_bag).reshape(1, -1)

    # Make prediction
    predicted_probabilities = model.predict(input_bag)
    predicted_class_index = np.argmax(predicted_probabilities[0])
    predicted_class = labels[predicted_class_index]

    responses = []
    print(predicted_class)
    intent_list = intents["intents"]
    for i in intent_list:
        if i["tag"] == predicted_class:
            response = random.choice(i["responses"])
            responses.append(response)
            if predicted_class != "greeting":
                keywords = keyword_extraction(user_input)
                advice = get_advice(dataset, predicted_class, keywords)
                cul_advice = get_cul_advice(dataset, history_keywords)
                if advice:
                    responses.append(advice)
                if cul_advice:
                    responses.append(cul_advice)
                return responses
            return responses
    return None


def get_advice(dataset, predicted_class, found_keywords):
    matching_responses = []
    # Iterate over each row in the dataset
    for index, row in dataset.iterrows():
        keywords = ast.literal_eval(row["mental_health_keywords"])
        if row["predicted_intent"] == predicted_class and keywords == found_keywords:
            print("ADVICE", found_keywords)
            matching_responses.append(row["Response"])

    if matching_responses:
        return random.choice(matching_responses)
    else:
        return None
    
def get_cul_advice(dataset, found_keywords):
    matching_responses = []
    # Iterate over each row in the dataset
    for index, row in dataset.iterrows():
        keywords = ast.literal_eval(row["mental_health_keywords"])
        if keywords == found_keywords:
            print("CUL_ADVICE", found_keywords)
            matching_responses.append(row["Response"])

    if matching_responses:
        return random.choice(matching_responses)
    else:
        return None   


# ------------------------------------------------------------


if __name__ == "__main__":
    app.run(debug=True)

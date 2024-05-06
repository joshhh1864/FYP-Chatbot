import pandas as pd
import ast
import joblib
from keywords import MENTAL_HEALTH_KEYWORDS
import re
from collections import Counter
from nltk.corpus import stopwords
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random

lemmatizer = WordNetLemmatizer()
from keras.preprocessing.sequence import pad_sequences
import pickle

from keras.models import load_model
model = load_model("modelN.keras")

import numpy as np


def find_response_by_intent(user_input, dataset):
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

    intent_list = intents["intents"]
    for i in intent_list:
        if i["tag"] == predicted_class:
            response = random.choice(i["responses"])
            return response

    return None


def find_response_by_keywords(user_input, dataset):
    user_input_tokens = user_input.lower().split()

    matching_keywords = []

    user_input_tokens = [re.sub(r"[^\w\s]", "", token) for token in user_input_tokens]

    for token in user_input_tokens:
        # Check if the token is a mental health keyword
        if token.lower() in MENTAL_HEALTH_KEYWORDS:
            matching_keywords.append(token)

    if matching_keywords == []:
        return None

    matching_responses = []

    # Iterate over each row in the dataset
    for index, row in dataset.iterrows():
        # Parse the string representation of keywords into a list
        keywords = ast.literal_eval(row["mental_health_keywords"])

        if keywords == matching_keywords:
            matching_responses.append(row["Response"])

    if matching_responses:
        print("Rule2")
        return matching_responses[0]
    else:
        return None


def find_response_by_context(user_input, dataset):
    negated_tokens= handle_negation(user_input)
    stop_words = set(stopwords.words("english"))

    # Count the occurrence of each token in user input
    user_input_counter = Counter(negated_tokens)
    best_match_count = 0
    best_match_response = None

    # Iterate over each row in the dataset
    for index, row in dataset.iterrows():
        # Split the cleaned context into tokens
        context_tokens = row["Cleaned_Context"].lower().split()

        # Remove punctuation from context tokens
        context_tokens = [re.sub(r"[^\w\s]", "", token) for token in context_tokens]

        # Remove stopwords from context tokens
        context_tokens = [token for token in context_tokens if token not in stop_words]

        # Count the occurrence of each token in the context
        context_counter = Counter(context_tokens)

        # Calculate the number of shared words between user input and context
        shared_words_count = sum((user_input_counter & context_counter).values())

        # Update best match if the current row has more shared words
        if shared_words_count > best_match_count:
            best_match_count = shared_words_count
            best_match_response = row["Response"]
    return best_match_response

def handle_negation(user_input):
    NEGATION_TERMS = ["not", "no", "never"]

    user_input_tokens = user_input.lower().split()

    negated_tokens = []

    # Initialize a flag to track negation state
    negation_flag = False

    # Iterate through tokens
    for token in user_input_tokens:
        # Check if token is a negation term
        if token in NEGATION_TERMS:
            # Toggle the negation flag
            negation_flag = not negation_flag
            continue

        # If the current token is not negated, add it to the list of negated tokens
        if not negation_flag:
            # Remove punctuation from the token
            token = re.sub(r"[^\w\s]", "", token)
            negated_tokens.append(token)
        else:
            # Prefix the negated word with "not_"
            negated_tokens.append("not_" + token)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    negated_tokens = [word for word in negated_tokens if word not in stop_words]

    return negated_tokens


def default_response():
    return "Sorry, I do not understand the question."


def chatbot_response(user_input, dataset):
    # Rule 1: If keywords not found, check response based on predicted intent
    response = find_response_by_intent(user_input, dataset)
    if response:
        return response

    # Rule 2: Check if the user input contains keywords and find response
    response = find_response_by_keywords(user_input, dataset)
    if response:
        return response

    # Rule 3: If no appropriate response found, try based on cleaned context
    response = find_response_by_context(user_input, dataset)
    if response:
        return response

    # Rule 4: If all else fails, return default response
    return default_response()


dataset = pd.read_csv("dataset_with_predicted_intents.csv")

print("Hello! I'm a simple chatbot. How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break
    else:
        response = chatbot_response(user_input, dataset)
        print("Bot:", response)

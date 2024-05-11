import pandas as pd
import ast
from keywords import MENTAL_HEALTH_KEYWORDS
import re
from collections import Counter
from nltk.corpus import stopwords
import json
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import random

from keras.preprocessing.sequence import pad_sequences
import pickle

from keras.models import load_model

model = load_model("modelN.keras")

import numpy as np

dataset = pd.read_csv("dataset_with_predicted_intents.csv")

def keyword_extraction(user_input):
    with open("keywords.json", 'r') as file:
        data = json.load(file)
        mental_health_keywords = data['MENTAL_HEALTH_KEYWORDS']

    user_input_tokens = user_input.lower().split()

    matching_keywords = []

    user_input_tokens = [re.sub(r"[^\w\s]", "", token) for token in user_input_tokens]

    for token in user_input_tokens:
        if token.lower() in mental_health_keywords:
            matching_keywords.append(token)

    if matching_keywords == []:
        return None

    return matching_keywords

def get_response(user_input, dataset):
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
                if advice:
                    responses.append(advice)
                return responses
            return responses
    return None


def get_advice(dataset, predicted_class, found_keywords):
    matching_responses = []
    print(found_keywords)
    # Iterate over each row in the dataset
    for index, row in dataset.iterrows():
        keywords = ast.literal_eval(row["mental_health_keywords"])
        if row["predicted_intent"] == predicted_class and keywords == found_keywords:
            print("yes",found_keywords)
            matching_responses.append(row["Response"])
        
    if matching_responses:
        return random.choice(matching_responses)
    else:
        return None


def default_response():
    return "Sorry, I do not understand the question."


def chatbot_response(user_input, dataset):
    responses = []
    response = get_response(user_input, dataset)
    if response:
        responses.extend(response)

    if not responses:
        return default_response()

    return responses


print("Hello! I'm a simple chatbot. How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break
    else:
        responses = chatbot_response(user_input, dataset)
        if responses:
            print("Bot:")
            print(responses[0])
            if (len(responses)==2):
                print("Professional's Advice: ", responses[1]   )

        else:
            print("Bot: Sorry, I couldn't understand that.")

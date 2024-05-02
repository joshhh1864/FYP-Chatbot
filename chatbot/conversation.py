import pandas as pd
import ast
import joblib
from keywords import MENTAL_HEALTH_KEYWORDS
import re
from collections import Counter
from nltk.corpus import stopwords
import json
import random


def find_response_by_intent(user_input, dataset):
    intents = json.loads(open('chatbot/intents.json').read())
    # Load the trained SVM model
    model = joblib.load("svm_model.joblib")

    # Load vectorizer used during training
    vectorizer = joblib.load("vectorizer.joblib")

    input_strings =[user_input]
    
    # Vectorize the input strings
    input_strings_vec = vectorizer.transform(input_strings)

    # Predict intents
    predicted_intents = model.predict(input_strings_vec)

    print(predicted_intents)

    intent_list=[]
    intent_list.append({"intent": predicted_intents})

    if intent_list:
        tag= intent_list[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                # Return the responses associated with the tag
                result= random.choice(i['responses'])
                break
        return result
    else:
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
    # Define negation terms
    NEGATION_TERMS = ["not", "no", "never"]

    # Split user input into tokens
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

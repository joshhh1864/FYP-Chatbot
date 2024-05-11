def train_model_svm():
    import pandas as pd
    import json
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    import joblib

    # Load Dataset
    with open("chatbot/intents.json", "r") as f:
        data = json.load(f)

    # Create DataFrame From Intents
    df = pd.DataFrame(data["intents"])

    dictionary = {"tag": [], "patterns": [], "responses": []}
    for i in range(len(df)):
        ptrns = df[df.index == i]["patterns"].values[0]
        rspns = df[df.index == i]["responses"].values[0]
        tag = df[df.index == i]["tag"].values[0]
        for j in range(len(ptrns)):
            dictionary["tag"].append(tag)
            dictionary["patterns"].append(ptrns[j])
            dictionary["responses"].append(rspns)

    # Create a working dictionary
    df = pd.DataFrame.from_dict(dictionary)

    X = df["patterns"]
    y = df["tag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    svm_model = SVC()

    # Grid Search for Hyperparameter Tuning
    param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf", "poly"]}
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train_vec, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Predictions
    y_pred = grid_search.predict(X_test_vec)

    # Evaluate the model
    print(classification_report(y_test, y_pred, zero_division=1))

    # Save the trained model
    joblib.dump(grid_search, "svm_model.joblib")

    # Save the vectorizer
    joblib.dump(vectorizer, "vectorizer.joblib")


def train_model_RNN():
    import json
    import numpy as np
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
    from keras.optimizers import SGD
    from keras.utils import to_categorical
    import random
    import pickle

    with open("chatbot/intents.json", "r") as file:
        data = json.load(file)

    avoid_words=['?','!']

    words = []
    labels = []
    documents = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in labels: 
                labels.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in avoid_words]
    words = sorted(list(set(words)))

    labels = sorted(list(set(labels)))

    pickle.dump(words,open('texts.pkl','wb'))
    pickle.dump(labels,open('labels.pkl','wb'))

    training = []

    for doc in documents:
       # Initialize bag of words
        bag = []
        # Tokenize and lemmatize the words in the pattern
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        # Create bag of words array
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)
        
        # Initialize output row
        output_row = list([0] * len(labels))
        # Set the index corresponding to the intent tag to 1
        output_row[labels.index(doc[1])] = 1
        
        # Append bag of words and output row to training data
        training.append([bag, output_row])

    random.shuffle(training)
    training=np.array(training, dtype=object)

    train_x = list(training[:,0])
    train_y = list(training[:,1])

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('modelN.keras', hist)
    print("model created")


def predict_intent():
    import pandas as pd
    import joblib
    import pickle
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    import numpy as np
    from keras.models import load_model
    modelKeras = load_model("modelN.keras")

    other_dataset_df = pd.read_csv("datasets/train.csv")

    words = pickle.load(open("texts.pkl", "rb"))
    labels = pickle.load(open("labels.pkl", "rb"))

    # Preprocess text data
    other_dataset_df["Cleaned_Context"] = other_dataset_df["Context"].apply(
        preprocess_text
    )

    other_dataset_df["mental_health_keywords"] = other_dataset_df["Context"].apply(
        keyword_extraction
    )

    predictions = []
    for index, row in other_dataset_df.iterrows():
        lemmatized_sentence = [
            lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(row['Context'])
        ]

        # Create bag-of-words representation
        input_bag = [1 if word in lemmatized_sentence else 0 for word in words]

        # Reshape input for prediction
        input_bag = np.array(input_bag).reshape(1, -1)

        # Make prediction
        predicted_probabilities = modelKeras.predict(input_bag)
        predicted_class_index = np.argmax(predicted_probabilities[0])
        predicted_class = labels[predicted_class_index]
        
        predictions.append(predicted_class)

    # Add predicted intents to the dataset
    other_dataset_df["predicted_intent"] = predictions

    # Save the dataset with predicted intents
    other_dataset_df.to_csv("dataset_with_predicted_intents.csv", index=False)

    # Visualize predicted intents
    # predicted_intent_visualisation(other_dataset_df)


def preprocess_text(text):
    import string
    from nltk.corpus import stopwords

    # Define negation terms
    NEGATION_TERMS = ["not", "no", "never"]

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text_tokens = text.split()

    negated_tokens = []

    # Initialize a flag to track negation state
    negation_flag = False

    # Iterate through tokens
    for token in text_tokens:
        # Check if token is a negation term
        if token in NEGATION_TERMS:
            # Toggle the negation flag
            negation_flag = not negation_flag
            continue

        # If the current token is not negated, add it to the list of negated tokens
        if not negation_flag:
            negated_tokens.append(token)
        else:
            # Prefix the negated word with "not_"
            negated_tokens.append("not_" + token)

    # Remove stopwords from negated tokens
    negated_tokens = [word for word in negated_tokens if word not in stop_words]

    # Join the tokens back into a string
    processed_text = " ".join(negated_tokens)

    return processed_text


def predicted_intent_visualisation(df):
    unique_intents_count = df["predicted_intent"].nunique()

    # Check for null values
    null_count = df["predicted_intent"].isnull().sum()

    print(f"Number of unique intents: {unique_intents_count}")
    print(f"Intents Values: {df['predicted_intent'].unique()}")
    print(f"Number of null values: {null_count}")


def keyword_extraction(text):
    import string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    import json

    with open("keywords.json", 'r') as file:
        data = json.load(file)
        mental_health_keywords = data['MENTAL_HEALTH_KEYWORDS']

    extracted_mental_health_keywords = mental_health_keywords

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords and mental health keywords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Extract mental health keywords from filtered tokens
    mental_health_filtered_tokens = [
        word for word in filtered_tokens if word in extracted_mental_health_keywords
    ]

    keyword_counts = Counter(mental_health_filtered_tokens)

    meaningful_keywords = [keyword for keyword, count in keyword_counts.items()]

    return meaningful_keywords


# train_model_svm()
predict_intent()
# train_model_RNN()

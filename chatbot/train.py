def train_model():
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


def predict_intent():
    import pandas as pd
    import joblib

    other_dataset_df = pd.read_csv("datasets/train.csv")

    # Preprocess text data
    other_dataset_df["Cleaned_Context"] = other_dataset_df["Context"].apply(
        preprocess_text
    )

    # Load the trained SVM model
    model = joblib.load("svm_model.joblib")

    # Load vectorizer used during training
    vectorizer = joblib.load("vectorizer.joblib")

    # Prepare data for prediction
    X_other = other_dataset_df["Cleaned_Context"]

    # Vectorize the features
    X_other_vec = vectorizer.transform(X_other)

    # Predict intents
    predicted_intents = model.predict(X_other_vec)

    # Add predicted intents to the dataset
    other_dataset_df["predicted_intent"] = predicted_intents

    other_dataset_df["mental_health_keywords"] = other_dataset_df["Context"].apply(
        keyword_extraction
    )

    # Save the dataset with predicted intents
    other_dataset_df.to_csv("dataset_with_predicted_intents.csv", index=False)

    # Visualize predicted intents
    predicted_intent_visualisation(other_dataset_df)


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
    from keywords import MENTAL_HEALTH_KEYWORDS

    mental_health_keywords = MENTAL_HEALTH_KEYWORDS

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
        word for word in filtered_tokens if word in mental_health_keywords
    ]

    keyword_counts = Counter(mental_health_filtered_tokens)

    meaningful_keywords = [keyword for keyword, count in keyword_counts.items()]

    return meaningful_keywords


# train_model()
predict_intent()


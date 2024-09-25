import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
import nltk
from nltk.tokenize import word_tokenize

import re
nltk.download('punkt_tab')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Loads data from the SQLite database.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        X (pd.Series): Feature data (text).
        Y (pd.DataFrame): Labels (categories).
        category_names (list): List of category names for classification.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    X = df['message'].astype(str)
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)  # Drop non-label columns
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text data and normalizes it (lowercasing, removing punctuation, etc.).

    Args:
        text (str): Input text to tokenize.

    Returns:
        list: A list of cleaned and tokenized words.
    """
    # Normalization: Lowercase, remove punctuation, etc.
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # Remove non-alphabetic characters
    tokens = word_tokenize(text)
    
    # Optionally, you can remove stopwords and apply stemming/lemmatization
    return tokens


def build_model():
    """
    Builds a machine learning pipeline and performs a grid search.

    Returns:
        GridSearchCV: The machine learning model with grid search.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, token_pattern=None)),  # Set token_pattern=None
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    # Parameter grid for GridSearchCV
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__alpha': [0.1, 1, 10]
    }

    # Grid search
    model = GridSearchCV(pipeline, param_grid, cv=5, verbose=3, n_jobs=-1)
    return model
    #return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model performance on test data.

    Args:
        model: Trained machine learning model.
        X_test (pd.Series): Test data features.
        Y_test (pd.DataFrame): Test data labels.
        category_names (list): List of category names for classification.
    """
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saves the trained model as a pickle file.

    Args:
        model: Trained machine learning model.
        model_filepath (str): Path to save the pickle file.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print(X.head())
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
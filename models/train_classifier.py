import sys
import re
import numpy as np
import pandas as pd 
from sqlalchemy import create_engine

import nltk
nltk.download(['stopwords','wordnet', 'punkt'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Response', con=engine)
    X = df.message.values
    Y = df[df.columns.difference(['id', 'message', 'original', 'genre'])].values
    category_names = df.columns.difference(['id', 'message', 'original', 'genre'])
    
    return X, Y, category_names


def tokenize(text):
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # Remove stop words
        if tok not in stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('cff', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    # initial a dictionary for the metrics
    test_result = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []}

    # get the metric values from classification_report
    for i in np.arange(len(category_names)):
        report = classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True)
        test_result['accuracy'].append(report['accuracy'])
        test_result['precision'].append(report['weighted avg']['precision'])
        test_result['recall'].append(report['weighted avg']['recall'])
        test_result['f1_score'].append(report['weighted avg']['f1-score'])
        #print(report)
        #print('\n')

    df_test_result = pd.DataFrame(test_result, index=category_names)
    print('\nThe average accuracy is {}'.format(df_test_result['accuracy'].mean()))   


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
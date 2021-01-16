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
    '''
    Load dataset from the sqlite database,
    define feature and target variables X and Y

        Parameters:
                database_filepath (str): The filepath for the database

        Returns:
                X (numpy.array): The messages column, feature for the classification model
                Y (numpy.array): The categories columns, the target variables
                category_names (list): A list of category names
    '''
    # Load dataset
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Response', con=engine)
    # Define feature and target variables
    X = df.message.values
    Y = df[df.columns[4:]].values
    # Create a list of category names
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    '''
    Process text data: normalization, tokenization, stop words removal,
    and lemmatization

        Parameter:
                text (str): The text data need process
        
        Return:
                clean_token (list): A list of clean tokens
    '''
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenization
    tokens = word_tokenize(text)
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # Remove stop words
        if tok not in stopwords.words('english'):
            # Lemmatization
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build machine learning pipeline: 
        NLP, multi-output classification model, and grid search.

        No parameters

        Return:
                pipeline: A Pipeline object
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('cff', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search

    return pipeline


def highlight_mean(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_mean = s == s.loc['mean_score']
    return ['background-color: yellow' if v else '' for v in is_mean]


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print the accuracy, precision, recall, and f1 score of the model

        Parameters:
                model: A machine learning model / pipeline
                X_test: Test set of X
                Y_test: Test set of Y
                category_names: The list of category names
        
        No returns
    '''
    # Predict values using X_test
    Y_pred = model.predict(X_test)

    # Initial a dictionary for the metrics
    test_result = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []}

    # Get the metric values from classification_report
    for i in np.arange(len(category_names)):
        report = classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True)
        test_result['accuracy'].append(report['accuracy'])
        test_result['precision'].append(report['weighted avg']['precision'])
        test_result['recall'].append(report['weighted avg']['recall'])
        test_result['f1_score'].append(report['weighted avg']['f1-score'])
        #print(report)
        #print('\n')
    # Store the metric scores in a dataframe
    df_test_result = pd.DataFrame(test_result, index=category_names)
    df_test_result.loc['mean_score'] = df_test_result.mean()
    # Highlight the mean score row
    df_test_result.style.apply(highlight_mean)
    # Print the scores dataframe
    print('\n')
    print(df_test_result)
    print('\n')
    #print('\nThe average accuracy is {}'.format(df_test_result['accuracy'].mean()))   


def save_model(model, model_filepath):
    '''
    Save the model as a pickle file
    '''
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
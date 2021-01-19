import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load datasets and merge into one dataframe

        Parameters:
                messages_filepath (str): Filepath of the messages dataset
                categories_filepath (str): Filepath of the categories dataset

        Returns:
                df (pandas.DataFrame): A merged dataframe of the two datasets
    '''
    # Read datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the two datasets using the common id
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    '''
    Split the categories column into separate, clearly named columns, 
    convert values to binary, and drop duplicates.

        Parameters:
                df (pandas.DataFrame): The returned dataframe from load_data()
        
        Returns:
                df (pandas.DataFrame): A cleaned dataframe
    '''
    # Split the categories column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    # Extract a list of new column names for categories
    colnames = re.sub(r"-[0-9]", "", df["categories"][0]).split(";")
    # Rename the columns of `categories`
    categories.columns = colnames
    
    # Convert category values to 0 or 1
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert values that are not 0 to 1
        categories[column] = categories[column].replace('[^0]+', '1', regex=True)
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original categories column
    df.drop('categories', axis=1, inplace=True)
    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)    
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Save the dataset into a sqlite database

        Parameters:
                df (pandas.DataFrame): The cleaned dataset returned from clean_data()
                database_filename (str): Name of the database
        
        No returns
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # Load and merge datasets
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        # Clean the dataset
        print('Cleaning data...')
        df = clean_data(df)
        # Save the dataset into a sqlite database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
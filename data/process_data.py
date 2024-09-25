# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge datasets from filepaths.

    Args:
    messages_filepath (str): Filepath for the messages CSV file.
    categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    df (DataFrame): Merged DataFrame containing messages and categories.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    """
    Clean the data by splitting categories into individual columns,
    converting values to binary, and removing duplicates.

    Args:
    df (DataFrame): The merged DataFrame to be cleaned.

    Returns:
    df (DataFrame): Cleaned DataFrame with separate category columns.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string and convert to integer
        categories[column] = categories[column].str.split('-').str[1].astype(int)

        # ensure binary values (0 or 1)
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to a SQLite database.

    Args:
    df (DataFrame): Cleaned DataFrame to be saved.
    database_filename (str): The filename for the SQLite database.
    """
    # create sql alchemy engine
    engine = create_engine(f'sqlite:///{database_filename}')

    # write data to database
    df.to_sql('messages', engine, if_exists='replace', index=False)


def main():
    """
    Main function that orchestrates the loading, cleaning, and saving of data.

    It reads the filepaths for the messages, categories, and database from 
    command-line arguments and performs the ETL pipeline.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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
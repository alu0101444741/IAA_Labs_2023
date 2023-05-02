#!/usr/bin/env python
"""
    Utility functions

"""
from typing import Dict
import pandas as pd
import math
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)

K_value = 3
output_folder_path = './output/'
input_folder_path = './input/'
vocabulary_path = output_folder_path + 'vocabulario.txt'
vocabulary_words_count = len(open(vocabulary_path, mode = "r").readlines())
csv_main_file = input_folder_path + 'F75_train_FIXED.csv'
classes = ['positive', 'negative', 'neutral']

spell_checker = SpellChecker()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def preprocess_token(token: str, stemming: bool = True, lemmatization: bool = False):
    token = token.strip(' ') # Removing blank spaces
    token = token.lower() # To lower case                    
    token = spell_checker.correction(token) # Spellchecking
    if (token is None): return(None)
    # Must not be a one-letter-word, a stopword, numeric, ...
    if ((len(token) == 1) or token in stop_words or has_numbers(token) or has_special_characters(token)):
        return(None)
    # Stemming
    if (stemming): token = stemmer.stem(token)
    # Lemmatization
    if (lemmatization): token = lemmatizer.lemmatize(token) 
    return(token)

special_characters = ['(', ')', '.', ',', '-', '`', '"', '\'', ':', '&', '%', '+', '!', '=']

def has_special_characters(word: str):
    """
    Checks if there is a special character on a given word

    Parameters
    ----------
    word: str
        Word extracted from the dataset
    """
    return any(char in special_characters for char in word)

def has_numbers(word: str):
    """
    Checks if there is a number on a given word

    Parameters
    ----------
    word: str
        Word extracted from the dataset
    """
    return any(char.isdigit() for char in word)

def create_test_csv(rows: int):
    csv_file = pd.read_csv('./input/F75_train_FIXED.csv', sep=",")
    test_csv = csv_file[rows:len(csv_file)]
    return(test_csv)

def create_fixed_csv():
    csv_file = pd.read_csv('./input/F75_train.csv', sep=",")
    news = pd.DataFrame({'News': csv_file.iloc[:, 0], 'Classification': csv_file.iloc[:, 1]})
    first_useless_row = -1
    for i in range(0, len(news)):
        if ((type(news['News'][i]) is float) and (math.isnan(float(news['News'][i])))):
            first_useless_row = i
            break
    if (first_useless_row != -1):
        news = news.drop(index = range(first_useless_row, len(news)))
        news.to_csv('./input/F75_train_FIXED.csv', index = False)

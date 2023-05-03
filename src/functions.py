#!/usr/bin/env python
"""
    Utility functions

"""
from typing import Dict
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords

from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('wordnet2021', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)

# Program constants
output_folder_path = './output/'
input_folder_path = './input/'
csv_main_file = input_folder_path + 'F75_train_FIXED.csv'
vocabulary_path = output_folder_path + 'vocabulario.txt'
vocabulary_words_count = len(open(vocabulary_path, mode = "r").readlines())
classes = ['positive', 'negative', 'neutral']

K_value = 2
spell_checker = SpellChecker()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_token(token: str, stemming: bool = True, lemmatization: bool = False):
    # token = token.strip(' ') # Removing blank spaces --> Doesn't change the accuracy
    # token = token.lower() # To lower case. --> Drop accuracy (aprox. 10%)               
    # token = spell_checker.correction(token) # Spellchecking
    if (token is None): return(None)
    token = token.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    token = re.sub(r'[0-9]', '', token) # Remove numbers
    # Must not be a one-letter-word, a stopword, numeric, ...
    if ((len(token) <= 1) or (token in stop_words)): # or (not wordnet2021.synsets(token)) or has_numbers(token) or has_special_characters(token)):
        return(None)
    
    if (token[0].lower() != token[0]): return(token) # is a name --> Improve accuracy in some cases
    # Stemming
    #if (stemming): token = stemmer.stem(token) # --> Drop accuracy (aprox. 10%) 
    # Lemmatization
    #if (lemmatization): token = lemmatizer.lemmatize(token) # --> 
    return(token)


#!/usr/bin/env python
"""
    Preprocesser class implementation

"""
# ***************** pip install ******************
from typing import Dict
import math
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)
from functions import *
# ********************************************************

class Preprocesser:
    """
    Class to create a vocabulary from a dataset

    Attributes
    ----------
    csv_file_path: str
        Path to the CSV file which contains the dataset
    vocabulary_file_path: str
        Path/name of the txt file that will have the vocabulary
    spell_checker: SpellChecker
        Object to re-write those words that are not spelled correctly
    stemmer: PorterStemmer
        Object to reduce an inflected word down to its word stem
    lemmatizer: WordNetLemmatizer
        Object to reduce inflected words to their root word
    stop_words: list[str]
        A bunch of words to ignore
    """
    def __init__(self, csv_file_path: str, vocabulary_file_path: str = "vocabulario.txt"):
        """
        Parameters
        ----------
        csv_file_path: str
            Path to the CSV file which contains the dataset
        vocabulary_file_path: str
            Path/name of the txt file that will have the vocabulary
        """
        self.csv_file_path: str = csv_file_path
        self.vocabulary_file_path: str = vocabulary_file_path
        self.vocabulary: Dict[str, int] = {}

        self.spell_checker = SpellChecker()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
        

    def create_vocabulary(self, stemming: bool = True, lemmatization: bool = False):
        """
        Method to create the Dictionary using the stored dataset

        Parameters
        ----------
        stemming: bool
            Set to false to skip the stem
        lemmatization: bool
            Set to false to skip the lemmatization
        """
        csv_file = pd.read_csv(self.csv_file_path, sep=",")
        
        for article in csv_file['News']:
            tokens = nltk.word_tokenize(article)
            for token in tokens:
                token = token.strip(' ') # Removing blank spaces
                token = token.lower() # To lower case                    
                token = self.spell_checker.correction(token) # Spellchecking
                if (token is None):
                    continue
                # Must not be a one-letter-word, a stopword, numeric, ...
                if ((len(token) == 1) or token in self.stop_words or has_numbers(token) or has_special_characters(token)):
                    continue
                # Stemming
                if (stemming):
                    token = self.stemmer.stem(token)
                # Lemmatization
                if (lemmatization):
                    token = self.lemmatizer.lemmatize(token)
                # Adding to dictionary
                if token not in self.vocabulary:
                    self.vocabulary[token] = 1
                self.vocabulary[token] = self.vocabulary[token] + 1
                
        self.__write_to_file()

    def __write_to_file(self):
        """
        Create the vocabulary txt file and writes the stored Dictionary on it
        """
        vocabulary_file = open(self.vocabulary_file_path, mode="w")
        
        # Sorting the vocabulary
        vocabulary_keys = list(self.vocabulary.keys())
        vocabulary_keys.sort()
        vocabulary_sorted = {i: self.vocabulary[i] for i in vocabulary_keys}
        vocabulary = vocabulary_sorted
        print('Start writing to file')
        # Write to file
        for word in vocabulary:
            n = vocabulary_file.write(word)
            # n = self.vocabulary_file.write(': ' + str(vocabulary[word]))
            n = vocabulary_file.write('\n')
        vocabulary_file.close()    
    
    def __create_fixed_csv(self):
        csv_file = pd.read_csv('./input/F75_train.csv', sep=",")
        news = pd.DataFrame({'News': csv_file.iloc[:, 0], 'Classification': csv_file.iloc[:, 1]})
        first_useless_row = -1
        for i in range(0, len(news)):
            if ((type(news['News'][i]) is float) and (math.isnan(float(news['News'][i])))):
                # print("First trash row at: " + str(i))
                first_useless_row = i
                break
        if (first_useless_row != -1):
            news = news.drop(index = range(first_useless_row, len(news)))
            news.to_csv('./input/F75_train_FIXED.csv')

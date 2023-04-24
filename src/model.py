#!/usr/bin/env python
"""
    Corpus and LanguajeModel classes implementation

"""
from typing import Dict
import pandas as pd
import numpy as np
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

class Corpus:
    def __init__(self, classification: str, csv_file_path: str):
        self.classification = classification
        self.csv_file_path = csv_file_path
        self.words: Dict[str, int] = {}
        self.document_count = 0

        if (self.classification == 'neutral'):
            self.file_path = "corpusT.txt"
        else:
            self.file_path = "corpus" + self.classification[0].upper() + ".txt"

        self.spell_checker = SpellChecker()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
        self.__create_corpus()
    
    def __create_corpus(self, stemming: bool = True, lemmatization: bool = False):
        csv_file = pd.read_csv(self.csv_file_path, sep=",")

        for i in range(0, len(csv_file)):
            if (csv_file['Classification'][i] != self.classification):
                continue
            self.document_count += 1
            tokens = nltk.word_tokenize(csv_file['News'][i])
            for token in tokens:
                token = token.strip(' ') # Removing blank spaces
                token = token.lower() # To lower case                    
                token = self.spell_checker.correction(token) # Spellchecking
                if (token is None):
                    continue
                # Must not be a one-letter-word, a stopword, numeric, ...
                if ((len(token) == 1) or (token in self.stop_words) or has_numbers(token) or has_special_characters(token)):
                    continue
                # Stemming
                if (stemming):
                    token = self.stemmer.stem(token)
                # Lemmatization
                if (lemmatization):
                    token = self.lemmatizer.lemmatize(token)
                # Adding to dictionary
                if token not in self.words:
                    self.words[token] = 1
                self.words[token] = self.words[token] + 1

    def write_to_file(self):
        """
        Create the corpus txt file and writes the stored Dictionary on it
        """
        vocabulary_file = open(self.file_path, mode="w")
        # Sorting the vocabulary
        vocabulary_keys = list(self.words.keys())
        vocabulary_keys.sort()
        vocabulary_sorted = {i: self.words[i] for i in vocabulary_keys}
        vocabulary = vocabulary_sorted

        # Write to file
        for word in vocabulary:
            n = vocabulary_file.write(word + ' ' + str(vocabulary[word]))
            n = vocabulary_file.write('\n')
        vocabulary_file.close() 

class LanguageModel:
    def __init__(self, corpus: Corpus, vocabulary_path: str):
        self.corpus = corpus
        self.vocabulary_path = vocabulary_path 
        if (self.corpus.classification == 'neutral'):
            self.file_path = "modelo_lenguaje_T.txt"
        else:
            self.file_path = "modelo_lenguaje_" + self.corpus.classification[0].upper() + ".txt"  
        self.__create_model()
    
    def __create_model(self):
        corpus = self.corpus.words # create_dictionary_from_corpus_file(self.corpus.file_path)
        word_count = 0
        for word in corpus:
            word_count += corpus[word]
        vocabulary_words_number = len(open(self.vocabulary_path, mode = "r").readlines())

        model_file = open(self.file_path, mode = "w")
        model_file.write('Número de documentos (noticias) del corpus: ' + str(self.corpus.document_count) + '\n')
        model_file.write('Número de palabras del corpus: ' + str(word_count) + '\n')

        for word in corpus:
            word_probability = (corpus[word] + 1) / (len(corpus) + vocabulary_words_number)
            model_file.write('Palabra: ' + word + ' Frec.:' + str(corpus[word]) + ' LogProb: ' + str(np.log(word_probability)) + '\n')
        # UNK
        corpus['UNK'] = 0
        model_file.write('Palabra: UNK Frec.: 0 LogProb: ' + str(np.log(1 / (len(corpus) + vocabulary_words_number + 1))))
        model_file.close()

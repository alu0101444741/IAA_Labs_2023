#!/usr/bin/env python
"""
  Script to run the preprocessing

  Usage: main.py
"""
# import sys
from vocabulary import *
from model import *

# preprocesser = Preprocesser('./input/F75_train_FIXED.csv')
# preprocesser.create_vocabulary()
vocabulary_path = 'vocabulario.txt'
# csv_file = pd.read_csv('./input/F75_train_FIXED.csv', sep=",")

positive_corpus = Corpus('positive', './input/F75_train_FIXED.csv')
neutral_corpus = Corpus('neutral', './input/F75_train_FIXED.csv')
negative_corpus = Corpus('negative', './input/F75_train_FIXED.csv')

# positive_corpus.write_to_file()
# neutral_corpus.write_to_file()
# negative_corpus.write_to_file()

positive_model = LanguageModel(positive_corpus, vocabulary_path)
neutral_model = LanguageModel(neutral_corpus, vocabulary_path)
negative_model = LanguageModel(negative_corpus, vocabulary_path)

#!/usr/bin/env python
"""
  Script to create output files: vocabulary, corpuses and language models.

"""

# import sys
from vocabulary import Vocabulary
from corpus import Corpus
from language_model import LanguageModel
from functions import input_folder_path, vocabulary_path

# Create vocabulary
vocabulary = Vocabulary('./input/F75_train_FIXED.csv')
print('Writing vocabulary to file...')
vocabulary.write_to_file()

# Creating Corpus<P, N, T>
positive_corpus = Corpus('positive', input_folder_path + 'F75_train_FIXED.csv')
neutral_corpus = Corpus('neutral', input_folder_path + 'F75_train_FIXED.csv')
negative_corpus = Corpus('negative', input_folder_path + 'F75_train_FIXED.csv')
for corpus in [positive_corpus, neutral_corpus, negative_corpus]: corpus.write_to_file()

# Creating LanguageModel<P, N, T>
positive_model = LanguageModel(positive_corpus, vocabulary_path)
neutral_model = LanguageModel(neutral_corpus, vocabulary_path)
negative_model = LanguageModel(negative_corpus, vocabulary_path)
for model in [positive_model, neutral_model, negative_model]: model.write_to_file()

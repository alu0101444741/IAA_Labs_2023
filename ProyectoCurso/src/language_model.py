#!/usr/bin/env python
"""
    LanguajeModel class implementation
"""
from corpus import Corpus
from functions import Dict, output_folder_path, vocabulary_words_count
import numpy as np
DOCUMENTS_COUNT = 2500

class LanguageModel:
    """
    Class to create a language model from a corpus and a vocabulary

    Attributes
    ----------    
    corpus: Corpus
        Corpus object from which the model will be created
    vocabulary_path: str
        Vocabulary file name/path
    model: Dict[str, (int, float)]
        Dictionary to store a word with it frecquency and probability
    classification_probability: float
        Number of news on the corpus divided by the total
    """
    def __init__(self, corpus: Corpus, vocabulary_path: str):
        self.model: Dict[str, (int, float)] = {} # [word, (frec, prob)]
        self.corpus = corpus
        self.classification_probability = 0        
        self.vocabulary_path = vocabulary_path
        if (self.corpus.classification == 'neutral'):
            self.file_path = output_folder_path + "modelo_lenguaje_T.txt"
        else:
            self.file_path = output_folder_path + "modelo_lenguaje_" + self.corpus.classification[0].upper() + ".txt"  
        self.__create_model()

    def __create_model(self):
        """
        Private method where the model is created.
        """
        self.word_count = 0        
        for word in self.corpus.words:
            self.word_count += self.corpus.words[word]
            self.model[word] = self.__compute_word_stats(word)
        # UNK        
        self.model['UNK'] = (0, np.log(1 / (len(self.corpus.words) + vocabulary_words_count + 1)))
        self.classification_probability = round(np.log(self.corpus.document_count / DOCUMENTS_COUNT), 2)

    def write_to_file(self):
        """
        Create the language model txt file and writes the stored model (Dict) on it
        """
        model_file = open(self.file_path, mode = "w")
        model_file.write('Número de documentos (noticias) del corpus: ' + str(self.corpus.document_count) + '\n')
        model_file.write('Número de palabras del corpus: ' + str(self.word_count) + '\n')
        for word in self.model:
            model_file.write('Palabra: ' + word + ' Frec.:' + str(self.model[word][0]) + ' LogProb: ' + str(self.model[word][1]) + '\n')
        model_file.close()
    
    def __compute_word_stats(self, word: str):
        """
        Private method to create a pair (2-tuple) of a word frecquency and probability   
        """
        word_probability = round(np.log((self.corpus.words[word] + 1) / (len(self.corpus.words) + vocabulary_words_count)), 2)
        return((self.corpus.words[word], word_probability))

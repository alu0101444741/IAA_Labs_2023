#!/usr/bin/env python
"""
    LanguajeModel class implementation
"""
from corpus import Corpus
from functions import Dict, output_folder_path, vocabulary_words_count
import numpy as np
DOCUMENTS_COUNT = 2500

class LanguageModel:
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
        self.word_count = 0        
        for word in self.corpus.words:
            self.word_count += self.corpus.words[word]
            self.model[word] = self.__compute_word_stats(word)
        # UNK        
        self.model['UNK'] = (0, np.log(1 / (len(self.corpus.words) + vocabulary_words_count + 1)))
        self.classification_probability = np.log(self.corpus.document_count / DOCUMENTS_COUNT)

    def write_to_file(self):
        model_file = open(self.file_path, mode = "w")
        model_file.write('Número de documentos (noticias) del corpus: ' + str(self.corpus.document_count) + '\n')
        model_file.write('Número de palabras del corpus: ' + str(self.word_count) + '\n')
        for word in self.model:
            model_file.write('Palabra: ' + word + ' Frec.:' + str(self.model[word][0]) + ' LogProb: ' + str(self.model[word][1]) + '\n')
        model_file.close()
    
    def __compute_word_stats(self, word: str):
        word_probability = np.log((self.corpus.words[word] + 1) / (len(self.corpus.words) + vocabulary_words_count))
        return((self.corpus.words[word], word_probability))

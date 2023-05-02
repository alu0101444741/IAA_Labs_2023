#!/usr/bin/env python
"""
    Vocabulary class implementation

"""
from functions import Dict, nltk, pd, preprocess_token, output_folder_path

class Vocabulary:
    """
    Class to create a vocabulary from a dataset

    Attributes
    ----------
    csv_file_path: str
        Path to the CSV file which contains the dataset
    file_path: str
        Path/name of the txt file that will have the vocabulary
    spell_checker: SpellChecker
        Object to re-write those words that are not spelled correctly
    stemmer: PorterStemmer
        Object to reduce an inflected word down to its word stem
    lemmatizer: WordNetLemmatizer
        Object to reduce inflected words to their root word
    """
    def __init__(self, csv_file_path: str, file_path: str = output_folder_path + "vocabulario.txt", stemming: bool = True, lemmatization: bool = False):
        """
        Parameters
        ----------
        csv_file_path: str
            Path to the CSV file which contains the dataset
        file_path: str
            Path/name of the txt file that will have the vocabulary
        """
        self.csv_file_path: str = csv_file_path
        self.file_path: str = file_path
        self.vocabulary: Dict[str, int] = {}
        self.__create_vocabulary(stemming, lemmatization)

    def __create_vocabulary(self, stemming: bool, lemmatization: bool):
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
                token = preprocess_token(token, stemming, lemmatization)
                if (token is None): continue
                # Adding to dictionary
                if token not in self.vocabulary: self.vocabulary[token] = 1
                self.vocabulary[token] = self.vocabulary[token] + 1                

    def write_to_file(self):
        """
        Create the vocabulary txt file and writes the stored Dictionary on it
        """
        vocabulary_file = open(self.file_path, mode="w")
        
        # Sorting the vocabulary
        vocabulary_keys = list(self.vocabulary.keys())
        vocabulary_keys.sort()
        vocabulary_sorted = {i: self.vocabulary[i] for i in vocabulary_keys}
        vocabulary = vocabulary_sorted
        print('Start writing to file')
        # Write to file
        for word in vocabulary:
            vocabulary_file.write(word)
            vocabulary_file.write('\n')
        vocabulary_file.close() 
    

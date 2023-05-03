#!/usr/bin/env python
"""
    Corpus class implementation
"""
from functions import Dict, preprocess_token, output_folder_path, nltk, pd, K_value
import concurrent.futures

class Corpus:
    """
    Class to create a corpus from a dataset that contains news

    Attributes
    ----------
    classification: str
        One in: oositive, negative or neutral
    csv_file_path: str
        Path to the CSV file which contains the dataset
    from_file: bool
        Set to false to skip the default construction
    """
    def __init__(self, classification: str, csv_file_path: str, from_file: bool = True):
        self.classification = classification
        self.csv_file_path = csv_file_path
        self.words: Dict[str, int] = {}
        self.document_count = 0

        if (self.classification == 'neutral'):
            self.file_path = output_folder_path + "corpusT.txt"
        else:
            self.file_path = output_folder_path + "corpus" + self.classification[0].upper() + ".txt"
        if (from_file): self.__create_corpus()
    
    def __create_corpus(self, stemming: bool = True, lemmatization: bool = False):
        """
        Private method in which the Corpus is created.

        Parameters
        ----------
        stemming: bool
            True to stem all the words
        lemmatization: bool
            True to lemmatize all the words
        """
        csv_file = pd.read_csv(self.csv_file_path, sep=",")

        for i in range(0, len(csv_file)):
            if (csv_file['Classification'][i] != self.classification):
                continue
            self.document_count += 1
            tokens = nltk.word_tokenize(csv_file['News'][i])

            # Define the list of words to preprocess
            # words = ['apple', 'orange', 'banana', 'pear', 'kiwi', 'pineapple'] # <-- tokens

            # Create a thread pool with 4 worker threads
            with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
                # Submit the preprocessing tasks to the thread pool
                preprocessing_tasks = [executor.submit(preprocess_token, word, stemming, lemmatization) for word in tokens]
            
                # Collect the results from the completed tasks
                preprocessed_words = [task.result() for task in concurrent.futures.as_completed(preprocessing_tasks)]

            for token in preprocessed_words:
                if (token is None): continue
                # Adding to dictionary
                if token not in self.words:
                    self.words[token] = 1
                else:
                    self.words[token] = self.words[token] + 1

            #afor token in tokens:
            #a    token = preprocess_token(token, stemming, lemmatization)
            #a    if (token is None): continue
            #a    # Adding to dictionary
            #a    if token not in self.words:
            #a        self.words[token] = 1
            #a    else:
            #a        self.words[token] = self.words[token] + 1


    def set_words(self, words: Dict[str, int], document_count: int):
        """
        Alternative method to create the Corpus object.
        Overwrites the stored words and frecquencies and the document count.

        Parameters
        ----------
        words: Dict[str, int]
            New words
        document_count: int
            New document counter
        """
        self.words = words
        self.document_count = document_count

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
            vocabulary_file.write(word + ' ' + str(vocabulary[word]) + '\n')
        vocabulary_file.close() 

def create_corpuses_efficiently(csv_file_path: str) -> list[Corpus]:
    """
    Specific function for the current project.
    Creates three Corpus objects simultaneously to avoid looping three times on the same input file.
    Returns the three corpuses.

    Parameters:
    ----------
    csv_file_path: str
        Input file
    """
    positive_corpus = Corpus('positive', csv_file_path, from_file = False)    
    negative_corpus = Corpus('negative', csv_file_path, from_file = False)
    neutral_corpus = Corpus('neutral', csv_file_path, from_file = False)
    positive_words: Dict[str, int] = {}
    negative_words: Dict[str, int] = {}
    neutral_words: Dict[str, int] = {}
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    csv_file = pd.read_csv(csv_file_path, sep=",")
    for i in range(0, len(csv_file)):
        tokens = nltk.word_tokenize(csv_file['News'][i])
        for token in tokens:
            token = preprocess_token(token, True, False)
            if (token is None): continue
            # Adding to dictionary
            if (csv_file['Classification'][i] == 'positive'):
                if token not in positive_words:
                    positive_words[token] = 1
                else:
                    positive_words[token] += 1
                positive_count += 1
            elif (csv_file['Classification'][i] == 'negative'):
                if token not in negative_words:
                    negative_words[token] = 1
                else:
                    negative_words[token] += 1
                negative_count += 1
            else:
                if token not in neutral_words:
                    neutral_words[token] = 1
                else:
                    neutral_words[token] += 1
                neutral_count += 1
    positive_words['UNK'] = 1
    negative_words['UNK'] = 1
    neutral_words['UNK'] = 1

    # Remove words with frecquency less than K
    for words in [positive_words, negative_words, neutral_words]:
        words_to_delete = []
        for word in words:
            if (words[word] <= K_value):
                words['UNK'] += words[word]
                words_to_delete.append(word)
        for w in words_to_delete: del words[w]
    
    positive_corpus.set_words(positive_words, positive_count)
    negative_corpus.set_words(negative_words, negative_count)
    neutral_corpus.set_words(neutral_words, neutral_count)            
    return([positive_corpus, negative_corpus, neutral_corpus])
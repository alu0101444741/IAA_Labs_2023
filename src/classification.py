#!/usr/bin/env python
"""
    Functions to classify a document

"""
import csv
import time
# from vocabulary import Vocabulary
from corpus import Corpus, create_corpuses_efficiently
from language_model import LanguageModel
from functions import preprocess_token, input_folder_path, output_folder_path, vocabulary_path, csv_main_file, pd, math, nltk

train_set_path = input_folder_path + 'F75_train_1.csv'
test_set_path = input_folder_path + 'F75_train_2.csv'
classification_file_path = output_folder_path + 'clasificacion_alu0101444741.csv'
summary_file_path = output_folder_path + 'resumen_alu0101444741.csv'
#a = pd.read_csv(csv_main_file, sep=",")

# (1) Divide CSV into train and test
def split_csv(csv_file_path: str, train_set_size: float):
    csv_file = pd.read_csv(csv_file_path, sep=",")
    train_size = math.floor(len(csv_file) * train_set_size)
    train_set = csv_file.iloc[0:train_size]
    test_set  = csv_file.iloc[train_size:len(csv_file)]    
    train_set.to_csv(path_or_buf = train_set_path, index = False)
    test_set.to_csv(path_or_buf = test_set_path, index = False)
    return(train_set_path, test_set_path)

# (2) Using the train half create:
#   * Vocabulary
#   * <P, N, T> Corpus
#   * <P, N, T> LanguageModels
def create_language_models(train_set_path: str) -> list[LanguageModel]:
    #print('Creating vocabulary...')
    #vocabulary = Vocabulary(train_set_path, output_folder_path + 'train_vocabulary.txt')
    models = []
    print('Creating corpuses...')
    start_time = time.time()

    corpuses = create_corpuses_efficiently(train_set_path) #[]

    end_time = time.time()
    print('Done. Time:' + str(end_time - start_time))
    
    #for classification in classes:
    #    print('Creating ' + classification + ' corpus...')
    #    corpuses.append(Corpus(classification, train_set_path))
    print('Creating models...')
    start_time = time.time()

    for corpus in corpuses:        
        models.append(LanguageModel(corpus, vocabulary_path))
    
    end_time = time.time()
    print('Done. Time:' + str(end_time - start_time))
    return(models)

# (3) Classify the test half ignoring the Classification column
def classify_all(csv_file_path: str, language_models: list[LanguageModel]):
    csv_file = pd.read_csv(csv_file_path, sep=",")
    classification_file_content = []
    summary_file_content = []
    for document in csv_file['News']:
        current_line = []
        classification = classify_document(document, language_models)
        # First ten characters
        current_line.append(document[0:10] + ' ')
        # Prob. given by each model
        for probability in classification: current_line.append("%.2f" % probability) 
        # Classify by the maximum value
        current_classification = language_models[classification.index(max(classification))].corpus.classification
        current_line.append(current_classification)
        # Output files content
        classification_file_content.append(current_line)
        summary_file_content.append(current_classification)

    # Write the classification and summary files
    write_csv_file(classification_file_content, classification_file_path, ['FirstTenChar', 'LogP', 'LogN', 'LogT', 'Classification'])
    write_csv_file(summary_file_content, summary_file_path, ['Classification'])

# (4) Compute accuracy: (Success / Test_half_size * 100)
def compute_accuracy(test_set_path: str, summary_file_path: str):
    test_file = pd.read_csv(test_set_path, sep=",")
    summary_file = pd.read_csv(summary_file_path, sep=",")
    success = 0
    for index in range(0, len(test_file) - 1):
        if (test_file['Classification'][index] == summary_file['Classification'][index]):
            success += 1
    return((success / len(test_file)) * 100)

def classify_document(document: str, models: list[LanguageModel]):
    """
    Provides the probability that a document will be classified in a certain way based on given language models.

    Parameters
    ----------
    document: str
        The document with only words from the corpus
    models: LanguageModel
        Language model that stores the probabilities of each word
    """
    model_probabilities = []
    # Document preprocessing
    preprocessed_document = nltk.word_tokenize(document)    
    for token in preprocessed_document:
        token = preprocess_token(token)

    # Classification using each language model
    for model in models:
        model_probabilities.append(get_document_probability(preprocessed_document, model))
    
    return(model_probabilities)



def get_document_probability(preprocessed_document: str, model: LanguageModel):
    """
    Compute the probability of a document by the sum of all it words probabilities

    Parameters
    ----------
    preprocessed_document: str
        The document with only words from the corpus
    model: LanguageModel
        Language model that stores the probabilities of each word
    """
    current_sum = 0 # model.probability
    for word in preprocessed_document:
        if word not in model.corpus.words:
            current_sum += model.model['UNK'][1]
        else:
            current_sum += model.model[word][1]
    return(current_sum + model.classification_probability)

def write_csv_file(content: list[list[str]], file_path: str, header: list[str]):
    """
    Writes a given content on a csv file

    Parameters
    ----------
    content: list[list[str]]
        String matrix
    file_path: str
        File name/path
    """
    with open(file_path, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in content: writer.writerow(row)
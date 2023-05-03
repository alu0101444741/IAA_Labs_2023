from functions import nltk, Dict, pd
from classification import math

special_characters = ['(', ')', '.', ',', '-', '`', '"', '\'', ':', '&', '%', '+', '!', '=']

def has_special_characters(word: str):
    """
    Checks if there is a special character on a given word

    Parameters
    ----------
    word: str
        Word extracted from the dataset
    """
    return any(char in special_characters for char in word)

def has_numbers(word: str):
    """
    Checks if there is a number on a given word

    Parameters
    ----------
    word: str
        Word extracted from the dataset
    """
    return any(char.isdigit() for char in word)

def create_test_csv(rows: int):
    csv_file = pd.read_csv('./input/F75_train_FIXED.csv', sep=",")
    test_csv = csv_file[rows:len(csv_file)]
    return(test_csv)

def create_fixed_csv():
    csv_file = pd.read_csv('./input/F75_train.csv', sep=",")
    news = pd.DataFrame({'News': csv_file.iloc[:, 0], 'Classification': csv_file.iloc[:, 1]})
    first_useless_row = -1
    for i in range(0, len(news)):
        if ((type(news['News'][i]) is float) and (math.isnan(float(news['News'][i])))):
            first_useless_row = i
            break
    if (first_useless_row != -1):
        news = news.drop(index = range(first_useless_row, len(news)))
        news.to_csv('./input/F75_train_FIXED.csv', index = False)

def create_corpus_from_file(file_path: str):
    """
    Create a corpus from a file. Each word and its frecquency must be on a single line. i.e: park 12

    Parameters
    ----------
    file_path: str
        Words file name/path

    Return
    ----------
    dictionary: Dict[str, int]
        Words and it frecquency
    """
    dictionary: Dict[str, int] = {}
    vocabulary_file = open(file_path, mode="r")
    words = vocabulary_file.readlines()

    for word in words:
        tokens = nltk.word_tokenize(word)
        dictionary[tokens[0]] = int(tokens[1])

    return(dictionary)     

def create_dictionary_from_file(file_path: str):
    """
    Create a dictionary from a corpus file. Each word and its frecquency must be on a single line. i.e: park 12

    Parameters
    ----------
    file_path: str
        Words file name/path

    Return
    ----------
    dictionary: Dict[str, int]
        Words and it frecquency
    """
    dictionary: Dict[str, int] = {}
    vocabulary_file = open(file_path, mode="r")
    words = vocabulary_file.readlines()

    for word in words:
        tokens = nltk.word_tokenize(word)
        dictionary[tokens[0]] = int(tokens[1])

    return(dictionary)
    


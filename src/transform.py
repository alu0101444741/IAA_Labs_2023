from functions import nltk, Dict

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
    


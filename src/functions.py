#!/usr/bin/env python
"""
    Utility functions

"""
from typing import Dict
import nltk

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

def create_dictionary_from_corpus_file(file_path: str):
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
    


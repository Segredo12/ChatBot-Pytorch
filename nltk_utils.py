# nltk_utils.py
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

"""
This is the path for the file. !Atention! If you're having problens creating
file, you should create folder manually.
"""
FILE_PATH = "compiled_data/data.pth"
DEBUG = False  # You can use this onto 'True' or 'False' to see some logs on terminal.

def tokenize(sentence):
    """
    This function will divide the sentence onto an array of words/tokens
    a token is a word, number or a puncturation character.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    This function will try to find the root form of the word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    This function will return a bag of words as and array.
    """
    # Stem each word:
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word:
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
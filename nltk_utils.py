import nltk
import numpy as np
#nltk.download('punkt')  # package with pre-trained tokenizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    # splitting the sentence into tokens
    return nltk.word_tokenize(sentence)

def stem(word):
    # getting the root form of the word (chopping the endings)
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    array with number appended in the word pos ,i.e,

    sentence = ["hello","how","are","you"]
    words = ["hi","hello","I","you","bye","thank","cool"]
    bag   = [ 0,     1,    0,   1,     0,   0,  0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag 




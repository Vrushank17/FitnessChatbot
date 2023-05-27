import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

stemmer = PorterStemmer()

def tokenize(sentence):
    return tokenizer.tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = []
    for w in all_words:
        if w in tokenized_sentence:
            bag.append(1)
        else:
            bag.append(0)
    
    return np.array(bag, dtype=np.float32)
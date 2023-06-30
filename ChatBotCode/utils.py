import nltk
import torch
from nltk.stem.porter import PorterStemmer
import numpy as np

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', padding=True)

MAX_ID_LENGTH = 100

stemmer = PorterStemmer()

def encode(input):
    # encoding = tokenizer.encode(sentence)
    # return tokenizer.convert_ids_to_tokens(encoding)
    tokenized_inputs = tokenizer.encode_plus(
                            input, 
                            add_special_tokens=True, 
                            max_length=MAX_ID_LENGTH, 
                            padding='max_length', 
                            return_tensors="pt"
                        )
    return tokenized_inputs

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = []
    for w in all_words:
        if w in tokenized_sentence:
            bag.append(1)
        else:
            bag.append(0)
    
    return np.array(bag)

def create_input_ids(input):
    return encode(input)['input_ids']

def create_attention_mask(input):
    return encode(input)['attention_mask']

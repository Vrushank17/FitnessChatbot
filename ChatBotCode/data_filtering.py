import json
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy_vals = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokenized_sentence = tokenize(pattern)
        all_words.extend(tokenized_sentence)
        xy_vals.append((tokenized_sentence, tag))
    
ignore_punc = ['?', '.', ';', ',', '!']
all_words = [stem(word) for word in all_words if word not in ignore_punc]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

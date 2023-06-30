import json
from utils import create_input_ids, create_attention_mask

with open('intents.json', 'r') as f:
    intents = json.load(f)

# all_input_ids = []
# tags = []
X_train = []
y_train = []
attention_masks = []

tag_index = 0

for intent in intents['intents']:
    for pattern in intent['patterns']:
        input_ids = create_input_ids(pattern)
        attention_masks.append(create_attention_mask(pattern))
        X_train.append(input_ids)
        y_train.append(tag_index)
    tag_index += 1

def get_data():
    return X_train, y_train, attention_masks
    
# ignore_punc = ['?', '.', ';', ',', '!']
# all_words = [stem(word) for word in all_words if word not in ignore_punc]
# all_words = [word for word in all_words if word not in ignore_punc]
# all_words = sorted(set(all_words))
tags = sorted(set(y_train))

from utils import tokenize, bag_of_words
import torch
from data_filtering import *
from train import model
import random

def return_response(user_input):
    user_input = tokenize(user_input)
    X = bag_of_words(user_input, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() >= 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                return response
    else:
        return "Sorry, I don't understand what you saying"
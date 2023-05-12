# chat.py
import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, DEBUG, FILE_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)  # Reload the json file;

data = torch.load(FILE_PATH)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Segredo"  # This is the name of the ChatBot.
close_bot_sentence = "quit"  # You can change this to any word you would like so the ChatBot will stop working.
print(f"Hi my name is {bot_name}, let's have a talk ? (type '{close_bot_sentence}' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    if DEBUG:
        print(f"[DEBUG] Sentence value: {sentence}")
    X = bag_of_words(sentence, all_words)
    if DEBUG:
        print(f"[DEBUG] All words value: {all_words}")
        print(f"[DEBUG] Bag of Words value: {X}")
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    if DEBUG:
        print(f"[DEBUG] Output value: {output}")
    _, predicted = torch.max(output, dim=1)
    if DEBUG:
        print(f"[DEBUG] Predicted value: {predicted} having as an item: {predicted.item()}")
    tag = tags[predicted.item()]
    if DEBUG:
        print(f"[DEBUG] Tag value: {tag}")
    probs = torch.softmax(output, dim=1)
    if DEBUG:
        print(f"[DEBUG] Probs value: {probs}")
    prob = probs[0][predicted.item()]
    if DEBUG:
        print(f"[DEBUG] Prob.item value: {prob.item()}")
    if prob.item() > 0.999:  # Chance of the Bot anwsering
        awnser = False
        if DEBUG:
            print(f"[DEBUG] Intentes list: {intents['intents']}")
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                awnser = True  # Checks that the ChatBot found the awnser.
        if not awnser:  # In case bot doesn't find the awnser he will send a custom message.
            print(f"{bot_name}: I do not understand...")
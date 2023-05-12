# train.py
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from nltk_utils import DEBUG
from nltk_utils import FILE_PATH

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem

with open('intents.json', 'r') as f:
    intents = json.load(f)


class ChatDataset(Dataset):

    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Support indexing such that dataset[i] can be used to get i-th sample:
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # We can call len(dataset) to return the size:
    def __len__(self):
        return self.n_samples


# Function that will Start to read intents file and words, and will try to ignore some words.
# It will prepare Hyper-parameters:
def first_part_of_training():
    all_words = []
    tags = []
    xy = []
    # Loop through each sentence in out intents patterns
    for intent in intents['intents']:
        tag = intent['tag']
        # Add to tag list:
        tags.append(tag)
        for pattern in intent['patterns']:
            # Tokensize each word in the sentence:
            w = tokenize(pattern)
            # Add to our words list:
            all_words.extend(w)
            # Add to xy pair:
            xy.append((w, tag))

    # Stem and lower each word
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    # Remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        # X: bag of words for each pattern_sentence:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        # y: PyTorch CrossEntropyLoss need only class labels, not one-hot:
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Hyper-parameters
    num_epochs = 1000  # Number of times that the training will occur.
    batch_size = 8  # Number of units manufactured in a production run.
    learning_rate = 0.01  # Speed of the machine learning.
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Calls out the second part of the training:
    second_part_of_training(num_epochs, train_loader, device, model, criterion, optimizer, input_size, hidden_size,
                            output_size, all_words, tags)


# Function that will train the model:
def second_part_of_training(num_epochs, train_loader, device, model, criterion, optimizer, input_size, hidden_size,
                            output_size, all_words, tags):
    if DEBUG:
        print(f"[DEBUG] num_epochs range: {range(num_epochs)}")
        print(f"[DEBUG] Train_loader len: {len(train_loader)}")
    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            if DEBUG:
                print(f"[DEBUG] Words value: {words}")
            labels = labels.to(device)
            if DEBUG:
                print(f"[DEBUG] Labels value: {labels}")
            # Forward pass:
            outputs = model(words)
            if DEBUG:
                print(f"[DEBUG] Outputs value: {outputs}")
            # If y would be one-hot, we must apply:
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels.type(torch.int64))
            if DEBUG:
                print(f"[DEBUG] Loss value: {loss}")

            # Backward and optimize:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')

    # We now call out third part of the training:
    third_part_of_training(model, input_size, hidden_size, output_size, all_words, tags)


# Function that will end the training execution storing the data information onto the file:
def third_part_of_training(model, input_size, hidden_size, output_size, all_words, tags):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    if DEBUG:
        # print("[DEBUG] Model State created: ", data["model_state"]) # Array of tensor;
        print("[DEBUG] Input Size created: ", data["input_size"])
        print("[DEBUG] Hidden Size created: ", data["hidden_size"])
        print("[DEBUG] Output Size created: ", data["output_size"])
        print("[DEBUG] All Words created: ", data["all_words"])
        print("[DEBUG] Tags created: ", data["tags"])

    torch.save(data, FILE_PATH)

    print(f'Training complete. file saved to {FILE_PATH}')


# Function that is going to execute all code for the training:
def execution_of_training():
    first_part_of_training()


if __name__ == '__main__':
    execution_of_training()

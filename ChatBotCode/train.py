import json
import numpy as np
from data_filtering import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import BertForSequenceClassification

X_train, y_train, attention_masks = get_data()

X_train = torch.cat(X_train, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
y_train = torch.tensor(y_train)

# print(f"Input: {X_train[0]}")
# print(f"Output: {y_train}")

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train, attention_masks):
        self.n_samples = len(X_train)
        self.X_train = X_train
        self.y_train = y_train
        self.attention_masks = attention_masks
    
    def __getitem__(self, index):
        X = self.X_train[index]
        y = self.y_train[index]
        attention_mask = self.attention_masks[index]

        return {
            'input_ids': X,
            'tag': y,
            'attention_mask': attention_mask,
        }

    def __len__(self):
        return self.n_samples

batch_size = 8
learning_rate = 0.001
epochs = 10

dataset = ChatDataset(X_train, y_train, attention_masks)

train_loader = DataLoader(
    dataset = dataset,
    batch_size = batch_size,
    shuffle = True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(y_train))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

progress_bar = tqdm(total=len(train_loader), desc='Training Progress')

model.train()
for epoch in range(epochs):
    for batch in train_loader:
        X = batch['input_ids'].to(device)
        y = batch['tag'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Compute prediction and loss
        output = model(input_ids=X, attention_mask=attention_mask)

        loss = loss_fn(output.logits, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)

    if (epoch + 1) % 100 == 0:
        print(f"loss: {loss.item():.4f}  [{epoch+1}/{epochs}]")

print(f"final_loss, loss={loss.item():.4f}")

progress_bar.close()

# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }

# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('model_scripted.pt') # Save


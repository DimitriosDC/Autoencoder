from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim, empty
from sklearn import decomposition
from sklearn.datasets import fetch_openml, make_regression
import statistics
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import Subset
import torch.nn.init as init

import os
import sys
import pickle

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed =  int(sys.argv[1])
torch.manual_seed(seed) # torch seed
random.seed(seed) # Removed this to set seed globally for whole file
np.random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

sheet_sizes = np.array(([30,30],[30,30])) # sizes of each input sheet
receptor_densities = [1,1] # densitity of receptors in each sheet

input_sheets1 = input_sheets(sheet_sizes=sheet_sizes, sheet_densities=receptor_densities)
input_sheets1.create_input_sheets() # create input sheets

num_inputs = 30000 # total number of inputs for model

load_path = '/home/pca20dd/Autoencoder/data/dataset_generate.pkl'      #Test Dataset
with open(load_path, 'rb') as f:
    loaded_dataset = pickle.load(f)

dataset = loaded_dataset

sheet_sizes = 30
receptors_each_region = input_sheets1.receptors_each_region


# Load dataset
train_data_full = dataset.all_samples[:20000,:]
validation_data_full = dataset.all_samples[20000:25000,:]
test_data_full = dataset.all_samples[25000:,:]

test_1_full = test_data_full[test_data_full[:, :receptors_each_region[0]].sum(axis=1) > 0]
test_2_full = test_data_full[test_data_full[:, receptors_each_region[0]:].sum(axis=1) > 0]

###Add to Non-Linear
test_1_full = test_1_full[:1000,:]
test_2_full = test_2_full[:1000,:]

print(test_2_full.shape)

# Convert the scaled data back to torch tensors
train_data = torch.from_numpy(train_data_full)
validation_data = torch.from_numpy(validation_data_full)
test_data = torch.from_numpy(test_data_full)

test_1 = torch.from_numpy(test_1_full)
test_2 = torch.from_numpy(test_2_full)

# Define the data loaders
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=100, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

test_1_loader = torch.utils.data.DataLoader(test_1, batch_size=100, shuffle=False)
test_2_loader = torch.utils.data.DataLoader(test_2, batch_size=100, shuffle=False)

class AutoencoderUntied(nn.Module):
    def __init__(self, input_shape, h, enc_weights, dec_weights):
        super().__init__()
        self.encoder = nn.Linear(input_shape,h, bias=False)
        self.decoder = nn.Linear(h, input_shape, bias=False)

        self.encoder.weight = nn.Parameter(enc_weights)
        self.decoder.weight = nn.Parameter(dec_weights)

    def forward(self, x):
        encoded_feats = self.encoder(x) #noisy_input)
        reconstructed_output = self.decoder(encoded_feats)
        return reconstructed_output

# h = input_sheets1.total_receptors ###### FOR MULTIPLE numbers of hidden layers
h = 50
input_shape = dataset[0].shape[0]
enc_weights = torch.empty(h, input_shape)
dec_weights = torch.empty(input_shape, h)
nn.init.kaiming_uniform_(dec_weights, a=math.sqrt(5))

nn.init.xavier_uniform_(enc_weights)
enc_weights = enc_weights.to(device)
dec_weights = dec_weights.to(device)
model = AutoencoderUntied(input_shape, h, enc_weights.float(), dec_weights.float()).to(device)

# # Define the loss function
criterion = nn.MSELoss()

# # Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# WEIGHTS
weights_enc = model.encoder.weight.data.cpu()
weights_dec = model.decoder.weight.data.cpu()

encod_weights = torch.eye(weights_enc.shape[1])
dot_products = torch.matmul(weights_enc, weights_enc.T) - torch.eye(weights_enc.shape[0])

track_loss = []
num_epochs = 50
val_loss = []
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for data3 in train_loader:
        data3 = data3.float().to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        recon_batch = model(data3)
        # loss = criterion(recon_batch.to(device), data.float())
        loss = criterion(recon_batch, data3)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  * input_shape # Accumulate the loss within the epoch
    track_loss.append(epoch_loss / len(train_loader))  # Average loss per epoch

        # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss_data = 0
    with torch.no_grad():
        for data in validation_loader:
            data = data.float().detach()
            data = data.to(device)
            recon = model(data)
            loss_data = criterion(recon.to(device), data.float())
            val_loss_data += loss_data.item() * input_shape
    val_loss.append(val_loss_data / len(validation_loader))

plt.plot(range(num_epochs), track_loss, label='Training Loss')
plt.plot(range(num_epochs), val_loss, label='Validation Loss')
plt.xlabel('Number of Training Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.savefig(f'/home/pca20dd/Autoencoder/Plots/loss_plot_{seed}.png')


# Test phase
model.eval()  # Set the model to evaluation mode
test_loss_data = 0

with torch.no_grad():
    for data in test_loader:
        data = data.float().to(device)  # Move data to the same device as the model
        recon = model(data)
        loss_data = criterion(recon, data)
        test_loss_data += loss_data.item() * input_shape

mse_test = test_loss_data / len(test_loader)
print("MSE on test set:", mse_test)

# Test phase for test dataset 1
model.eval()  # Set the model to evaluation mode
test_loss_data_1 = 0

with torch.no_grad():
    for data in test_1_loader:
        data = data.float().to(device)  # Move data to the same device as the model
        recon = model(data)
        loss_data = criterion(recon, data)
        test_loss_data_1 += loss_data.item() * input_shape

mse_test_1 = test_loss_data_1 / len(test_1_loader)
print("MSE on test set 1:", mse_test_1)

# Test phase for test dataset 2
model.eval()  # Set the model to evaluation mode
test_loss_data_2 = 0

with torch.no_grad():
    for data in test_2_loader:
        data = data.float().to(device)  # Move data to the same device as the model
        recon = model(data)
        loss_data = criterion(recon, data)
        test_loss_data_2 += loss_data.item() * input_shape

mse_test_2 = test_loss_data_2 / len(test_2_loader)
print("MSE on test set 2:", mse_test_2)

 # WEIGHTS
weights_enc = model.encoder.weight.data
weights_dec = model.decoder.weight.data
weights_enc_cpu = model.encoder.weight.detach().cpu().numpy()
weights_dec_cpu = model.decoder.weight.detach().cpu().numpy()

np.save(np.append(mse_test, mse_test_1, mse_test_2),f'/home/pca20dd/Autoencoder/results/{seed}mse_results.npy')

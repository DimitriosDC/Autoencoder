from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import math
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim, empty
from sklearn import decomposition
from sklearn.datasets import fetch_openml, make_regression
import statistics
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import Subset
import torch.nn.init as init


import os
import sys
from google.colab import drive
import pickle

from functions_input_sheet import input_sheets
from plotting_functions import plot_reconstructed_input, plot_grids


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

class Autoencoder(nn.Module):
    def __init__(self, sheet_sizes, h, enc_weights, dec_weights):
        #print(h)
        super().__init__()
        self.encoder = nn.Linear(input_shape,h, bias=True)
        self.decoder = nn.Linear(h, input_shape, bias=True)

        self.encoder.weight = nn.Parameter(enc_weights)
        self.decoder.weight = nn.Parameter(dec_weights)
        #self.sigmoid = nn.Sigmoid()  # Add a sigmoid activation function
        #self.decoder.weight = nn.Parameter(weights.transpose(0, 1))


    def forward(self, x):#, noise_factor=0.2):
        #encoded_feats = self.encoder(x)
        #sigmoid_out = self.sigmoid(encoded_feats)  # Apply sigmoid activation
        #reconstructed_output = self.decoder(sigmoid_out)

        # Add noise to the input
        #noisy_input = add_noise(x, noise_factor)      ###ADDED NOISE####

        encoded_feats = self.encoder(x)
        encoded_feats = torch.sigmoid(encoded_feats)  # Apply sigmoid activation to hidden layer
        reconstructed_output = self.decoder(encoded_feats)


        return reconstructed_output

## set h to be no bottleneck- each number of receptors to hidden units
h = 50
input_shape = dataset[0].shape[0]

enc_weights = torch.empty(h, input_shape)
dec_weights = torch.empty(input_shape, h)
#weights = torch.randn(h, input_shape)
#weights /= torch.sum(weights,1)[:,None]

# Create a linear layer
# weights = nn.Linear(h, input_shape)

# Initialize the weights with the default method
#nn.init.kaiming_uniform_(enc_weights)
nn.init.kaiming_uniform_(dec_weights, a=math.sqrt(5))

nn.init.xavier_uniform_(enc_weights)
#nn.init.xavier_uniform_(dec_weights)
enc_weights = enc_weights.to(device)
dec_weights = dec_weights.to(device)


model = Autoencoder(input_shape, h, enc_weights.float(), dec_weights.float()).to(device)
#model = AutoencoderSeq(input_shape, h, weights.float()).to(device) # seq

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer

#optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)


# Train the model
val_loss = []
track_loss = []
num_epochs = 50

for epoch in range(num_epochs):
  epoch_loss = 0
  model.train()
  for data3 in train_loader:
        data3 = data3.float().to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        recon_batch = model(data3)
        loss = criterion (recon_batch.to(device), data3).float()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * input_shape
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

plt.savefig(f'/home/pca20dd/Autoencoder/Plots/Non_linear_plots/loss_plot_{seed}.png')

# # Test the model
track_loss_data = 0
#with torch.no_grad():
for data in test_loader:
    data = data.float().to(device)
    enc = model.encoder(data)
    recon = model(data)
    loss_data = criterion(recon, data)
    track_loss_data += loss_data.item() * input_shape

mse = track_loss_data / len(test_loader)
print("MSE:", mse)

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
weights_enc_cpu = weights_enc.cpu().detach().numpy()
weights_dec_cpu = weights_dec.cpu().detach().numpy()

np.save(f'/home/pca20dd/Autoencoder/results/non_linear_results/{seed}_mse_results.npy', np.array([mse_test, mse_test_1, mse_test_2]))

u, s, vh = np.linalg.svd(weights_enc_cpu.T, full_matrices=False)
u1, s1, vh1 = np.linalg.svd(weights_dec_cpu, full_matrices=False)
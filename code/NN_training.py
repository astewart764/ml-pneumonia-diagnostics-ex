from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, SubsetRandomSampler
from sys import getsizeof
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
import multiprocessing as mp

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        categories = {0 : 'Healthy', 1 : 'Pneumonia'}
        return (self.data[idx], self.labels[idx])

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.conv1      =   self.compress_block(in_channels, 32, 7, 3)
        self.conv2      =   self.compress_block(32, 64, 3, 1)
        self.conv3      =   self.compress_block(64, 128, 3, 1)

        self.upconv1    =   self.expand_block(128, 64, 3, 1)
        self.upconv2    =   self.expand_block(64 * 2, 32, 3, 1)
        self.upconv3    =   self.expand_block(32 * 2, out_channels, 3, 1)

        self.fc1        =   nn.Linear(32 * 2 * out_channels, 64)
        self.fc2        =   nn.Linear(64, 8)
        self.fc3        =   nn.Linear(8, 1)

    def compress_block(self, in_channels, out_channels, kernel_size, padding):

        compress =  nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
                        )

        return compress

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand =    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
                        )

        return expand

    def forward(self, x):

        conv1   =   self.conv1(x)
        conv2   =   self.conv2(conv1)
        conv3   =   self.conv3(conv2)

        upconv1 =   self.upconv1(conv3)
        upconv2 =   self.upconv2(torch.cat([upconv1, conv2], 1))
        upconv3 =   self.upconv3(torch.cat([upconv2, conv1], 1))

        flat    =   nn.flatten(upconv3)

        fc1     =   F.relu(self.fc1(flat))
        fc2     =   F.relu(self.fc2(fc1))
        output  =   F.sigmoid(fc2)

        return output

class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super(ConvNet, self).__init__()

        self.conv1  =   self.conv_block(in_channels, 1, 7, 3)
        self.conv2  =   self.conv_block(1, 1, 3, 1)
        self.conv3  =   self.conv_block(1, 1, 3, 1)
        self.fc1    =   nn.Linear(4800, 256)
        self.fc2    =   nn.Linear(256, 8)
        self.out    =   nn.Linear(8, 1)

    def conv_block(self, in_channels, out_channels, kernel_size, padding):

        conv_down =  nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
                        )

        return conv_down

    def forward(self, x):

        x   =   self.conv1(x)
        x   =   self.conv2(x)
        x   =   self.conv3(x)
        x   =   x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x   =   F.relu(self.fc1(x))
        x   =   F.relu(self.fc2(x))
        x   =   nn.sigmoid(self.out(x))

        return x

def LoadDataset(category):
    npy_directory = str(Path(__file__).parent.parent) + '\\npy_dataset\\'
    image_data = torch.Tensor(np.load(npy_directory + category + '_images.npy')[:2000])
    label_data = torch.Tensor(np.load(npy_directory + category + '_labels.npy')[:2000])

    return image_data, label_data

def LoadBatch(batch_no, device):
    batch_directory = str(Path(__file__).parent.parent) + '\\training_batches\\'
    print("\nLoading training_batch{:02d}.npy onto {}\n".format(batch_no, torch.cuda.get_device_name()))
    batch = np.load(batch_directory + 'training_batch{:02d}.npy'.format(batch_no))

    img_batch, label_batch = np.expand_dims(np.array([_ for _ in batch[:, 0]]), axis = 1), np.expand_dims(np.array([_ for _ in batch[:, 1]]), axis = 1)
    return torch.tensor(img_batch, dtype = torch.float32, device = device), torch.tensor(label_batch, dtype = torch.float32, device = device)

def get_device():
    device_name = torch.cuda.get_device_name()
    if torch.cuda.is_available():
        device = 'cuda:0'
        print("CUDA device available : Using " + device_name + "\n")

    else:
        device = 'cpu'
        print("CUDA device unavailable : Using " + device_name + "\n")

    return device, device_name

def train(network, kwargs, epochs, device):
    criterion   =   nn.BCEWithLogitsLoss()
    optimizer   =   optim.SGD(network.parameters(), lr = 0.001, momentum = 0.5)

    Path(str(Path(__file__).parent.parent) + '\\model_saves\\').mkdir(parents = True, exist_ok = True)

    for epoch in range(epochs):
        running_loss    =   0
        iter_counter    =   0

        for jdx, file in enumerate(Path(str(Path(__file__).parent.parent) + '\\training_batches\\').glob('**/*.npy')):
            train_X, train_Y    =   LoadBatch(jdx + 1, device)
            traindataset        =   ImageDataset(train_X, train_Y)
            trainloader         =   DataLoader(traindataset, **kwargs)

            for inputs, labels in trainloader:
                optimizer.zero_grad()

                outputs         =   network(inputs)
                loss            =   criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss    +=  loss.item()
                iter_counter    +=  1

                if iter_counter % 4 == 3:
                    print("Epoch: {:3d} | Ieration : {:3d} | Loss : {:3.3f}".format(epoch + 1, iter_counter + 1, running_loss))
                    running_loss    =   0

        torch.save(network.state_dict(), str(Path(__file__).parent.parent) + '\\model_saves\\pneumonia_cnn.pt')

def main():
    DEVICE, device_name = get_device()

    kwargs = {'batch_size'  : 32,
              'shuffle'     : False,
              'num_workers' : 0,
              'drop_last'   : True,
              'pin_memory'  : False,
             }

    pneumonia_network   =   UNet(1, 2).to(DEVICE)
    conv_network        =   ConvNet(1).to(DEVICE)

    epochs  =   5

    train(conv_network, kwargs, epochs, DEVICE)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()

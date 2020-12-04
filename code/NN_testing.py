from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
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
        x   =   torch.sigmoid(self.out(x))

        return x

def get_device():
    device_name = torch.cuda.get_device_name()
    if torch.cuda.is_available():
        device = 'cuda:0'
        print("CUDA device available : Using " + device_name + "\n")

    else:
        device = 'cpu'
        print("CUDA device unavailable : Using " + device_name + "\n")

    return device, device_name

def get_images(count):
    testFolderNormal    =   str(Path(__file__).parent.parent) + "\\scaled_dataset\\test\\NORMAL\\"
    testFolderPneumonia =   str(Path(__file__).parent.parent) + "\\scaled_dataset\\test\\PNEUMONIA\\"

    testNormalLimit     =   len(os.listdir(testFolderNormal)) - 1
    testPneumoniaLimit  =   len(os.listdir(testFolderPneumonia)) - 1

    samples     =   np.random.randint(0, 2, count)
    label_dict  =   {0 : 'Healthy', 1 : 'Pneumonia'}
    labels      =   [label_dict[lab] for lab in samples]

    test_batch  =   np.empty((1, 1, 480, 640))
    for sample in samples:
        if sample == 0:
            limit   =   testNormalLimit
            dir     =   testFolderNormal

        elif sample == 1:
            limit   =   testPneumoniaLimit
            dir     =   testFolderPneumonia

        else:
            print("Sample index not in range of 'Healthy' or 'Pneumonia'")
            sys.exit(1)

        test_img    =   np.array(Image.open(dir + os.listdir(dir)[random.randint(0, limit)])).reshape((1, 1, 480, 640))
        if test_batch.sum() ==  0:
            test_batch  =   test_img

        else:
            test_batch = np.concatenate((test_batch, test_img), axis = 0)

    return test_batch, labels

def test(network, images, device):

    images  =   torch.tensor(images, dtype = torch.float32).to(device)

    with torch.no_grad():
        outputs  =   network(images)
        predictions  =   torch.round(outputs).cpu().numpy().astype(int)

    return predictions

def main():
    DEVICE, device_name = get_device()

    conv_network    =   ConvNet(1)
    conv_network.load_state_dict(torch.load(str(Path(__file__).parent.parent) + '\\model_saves\\pneumonia_cnn.pt'))
    conv_network.to(DEVICE)

    # for param in conv_network.parameters():
    #     print(param.data.shape)

    test_batch, labels  =   get_images(20)
    predictions =   test(conv_network, test_batch, DEVICE)
    predictions.reshape(20)
    # Convert prediction ints to strings using dictioary apprach in 'get_images'

    fig1, axes  =   plt.subplots(nrows = 5, ncols = 4, gridspec_kw = {'hspace' : 0.5, 'wspace' : 0})
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.imshow(test_batch[idx, 0], cmap = 'gray')
        ax.set_title("Label : {} | Prediction : {}".format(labels[idx]))    # Add prediction tag to titles once converted to string
        ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()

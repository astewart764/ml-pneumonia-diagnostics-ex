<<<<<<< HEAD
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

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def CalcConvs(input_height, input_width, padding, kernel_size, stride):
    output_height   =   (input_height + padding * 2 - kernel_size) / stride + 1
    output_width    =   (input_width + padding * 2 - kernel_size) / stride + 1
    print("Output Height : " + str(output_height) + "\nOutput Width : " + str(output_width))
    return output_width, output_height

def CalcMaxPools(input_height, input_width, padding, kernel_size, stride):
    output_height   =   (input_height + padding * 2 - kernel_size) / stride + 1
    output_width   =   (input_width + padding * 2 - kernel_size) / stride + 1
    print("Output Height : " + str(output_height) + "\nOutput Width : " + str(output_width))
    return output_width, output_height

def LoadBatch(batch_no):
    batch_directory = str(Path(__file__).parent.parent) + '\\training_batches\\'
    print("Loading training_batch{:02d}.npy...".format(batch_no))
    batch = np.load(batch_directory + 'training_batch{:02d}.npy'.format(batch_no))

    img_batch, label_batch = np.array([_ for _ in batch[:, 0]]), np.array([_ for _ in batch[:, 1]])

    return torch.Tensor(img_batch), torch.Tensor(label_batch)

X, y = LoadBatch(2)

print(X.shape)

[print(idx) for idx in range(1,1304) if 1304 % idx == 0]

print(1304 / 8)

W, H = 640, 480
W, H = CalcConvs(input_width = W, input_height = H, padding = 3, kernel_size = 7, stride = 1)
W, H = CalcMaxPools(input_width = W, input_height = H, padding = 0, kernel_size = 2, stride = 2)
W, H = CalcConvs(input_width = W, input_height = H, padding = 1, kernel_size = 3, stride = 1)
W, H = CalcMaxPools(input_width = W, input_height = H, padding = 0, kernel_size = 2, stride = 2)

print(W * H)
=======
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

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def CalcConvs(input_height, input_width, padding, kernel_size, stride):
    output_height   =   (input_height + padding * 2 - kernel_size) / stride + 1
    output_width    =   (input_width + padding * 2 - kernel_size) / stride + 1
    print("Output Height : " + str(output_height) + "\nOutput Width : " + str(output_width))
    return output_width, output_height

def CalcMaxPools(input_height, input_width, padding, kernel_size, stride):
    output_height   =   (input_height + padding * 2 - kernel_size) / stride + 1
    output_width   =   (input_width + padding * 2 - kernel_size) / stride + 1
    print("Output Height : " + str(output_height) + "\nOutput Width : " + str(output_width))
    return output_width, output_height

def LoadBatch(batch_no):
    batch_directory = str(Path(__file__).parent.parent) + '\\training_batches\\'
    print("Loading training_batch{:02d}.npy...".format(batch_no))
    batch = np.load(batch_directory + 'training_batch{:02d}.npy'.format(batch_no))

    img_batch, label_batch = np.array([_ for _ in batch[:, 0]]), np.array([_ for _ in batch[:, 1]])

    return torch.Tensor(img_batch), torch.Tensor(label_batch)

X, y = LoadBatch(2)

print(X.shape)

[print(idx) for idx in range(1,1304) if 1304 % idx == 0]

print(1304 / 8)

W, H = 640, 480
W, H = CalcConvs(input_width = W, input_height = H, padding = 3, kernel_size = 7, stride = 1)
W, H = CalcMaxPools(input_width = W, input_height = H, padding = 0, kernel_size = 2, stride = 2)
W, H = CalcConvs(input_width = W, input_height = H, padding = 1, kernel_size = 3, stride = 1)
W, H = CalcMaxPools(input_width = W, input_height = H, padding = 0, kernel_size = 2, stride = 2)

print(W * H)
>>>>>>> 95c912283f818fcfe5f1c23e4b5d87123427c007

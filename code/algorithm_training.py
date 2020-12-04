from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def LoadDataset(category):
    npy_directory = str(Path(__file__).parent.parent) + '\\npy_dataset\\'
    image_data = np.load(npy_directory + category + '_images.npy')
    label_data = np.load(npy_directory + category + '_labels.npy')

    return image_data, label_data

print("Loading dataset...")
training_images, training_labels = LoadDataset('training')
testing_images, testing_labels = LoadDataset('testing')

print(training_images.shape)
print(testing_images.shape)

input("Continue to training? >> ")

knn = KNeighborsClassifier(n_neighbors = 10)
print("Fitting KNN classifier...")
knn.fit(trainImages[:500, :], trainLabels[:500])

print("Predicting test data...")
predictions = knn.predict(testImages[:100, :])
score = accuracy_score(testLabels[:100], predictions)
print("KNN accuracy: ", score)

fig3, axes = plt.subplots(nrows = 4, ncols = 3)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(testImages[i].reshape(480, 640))
    ax.set_title("Label: {}   |   Prediction {}".format(testLabels[i], predictions[i]))
    ax.set_axis_off()

fig3.subplots_adjust(wspace = 2)
plt.show()

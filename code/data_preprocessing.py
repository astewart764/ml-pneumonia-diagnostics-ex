from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math

class DatasetContainer:
    def __init__(self, source):
        self.normal = CustomDataset(source, "NORMAL\\")
        self.pneumonia = CustomDataset(source, "PNEUMONIA\\")

class CustomDataset:
    def __init__(self, source, ID):
        self.directory = source + ID
        self.files = list(map(lambda x : source + ID + x, os.listdir(source + ID)))

    def get(self, idx):
        img = np.asarray(Image.open(self.files[idx]))

        return img

def DatasetCreation(dataset_type):
    image_directory = str(Path(__file__).parent.parent) + "\\" + dataset_type + "\\"                    # Define directroy containing train, test, val
    train_directory = image_directory + "train\\"                                                       # Define directories containing NORMAL, PNEUMONIA
    val_directory = image_directory + "val\\"
    test_directory = image_directory + "test\\"

    train_data = DatasetContainer(train_directory)                                                      # Objects containing 'normal' and 'pneumonia' properties with nested classes of each containing 'directory' and 'files' properties
    test_data = DatasetContainer(test_directory)
    val_data = DatasetContainer(val_directory)

    return train_data, val_data, test_data

def ExampleComparison():
    example_normal =\
    Image.open(train.normal.files[np.random.randint(0, len(train.normal.files))])
    example_pneumonia =\
    Image.open(train.pneumonia.files[np.random.randint(0, len(train.pneumonia.files))])
    fig1, (ax11, ax12) = plt.subplots(1 , 2, figsize = (11, 5))
    ax11.imshow(example_normal, cmap = plt.cm.Greys)
    ax11.set_title('Healthy')
    ax12.imshow(example_pneumonia, cmap = plt.cm.Greys)
    ax12.set_title('Pneumonia')
    fig1.tight_layout()
    plt.show()

def CheckResolutions():
    array = np.asarray([])
    for X in [train, test, val]:
        for attr, obj in X.__dict__.items():
            print(obj.directory)
            for file in obj.files:
                image = Image.open(file)
                image_arr = np.asarray(image)

                if len(image_arr.shape) > 2:
                    image = ImageOps.grayscale(image)
                    image_arr = np.asarray(image)

                array = np.append(array, image_arr.shape)
            array = np.reshape(array, (-1, 2))
            print(np.max(array[:,0]), np.max(array[:,1]))                                                   # Max resolution : (height, width) : (2713, 2916)

def ResolutionStandardization():
    for X in [train, test, val]:
        for attr, obj in X.__dict__.items():
            scaled_directory = '\\'.join(obj.directory.split('\\')).replace('dataset', 'scaled_dataset')
            print(scaled_directory)
            Path(scaled_directory).mkdir(parents = True, exist_ok = True)

            for i, file in enumerate(obj.files):
                img_scaler = Image.new('RGB', (640, 480))
                standard_width, standard_height = img_scaler.size

                image = Image.open(file)
                image_width, image_height = image.size

                if image_width > image_height:
                    scaling = standard_width / image_width
                else:
                    scaling = standard_height / image_height

                newsize = tuple((scaling * np.asarray([image_width, image_height])).astype(int))
                image = image.resize(newsize)

                centre_tuple =\
                tuple((np.asarray(img_scaler.size) / 2).astype(int) - (np.asarray(image.size) / 2).astype(int))

                img_scaler.paste(image, centre_tuple)
                img_scaler = ImageOps.grayscale(img_scaler)
                img_scaler.save(scaled_directory + '{}{:04d}.jpeg'.format(attr, i))

def ScaledDirSwitch():
    for X in [train, test, val]:
        for attr, obj in X.__dict__.items():
            scaled_directory = '\\'.join(obj.directory.split('\\')).replace('dataset', 'scaled_dataset')
            obj.directory = scaled_directory

def GeneratePandasData(category):
    class_value = 0
    class_array = np.array([])
    category_array = np.array([])
    index_array = np.array([])
    name_array = np.array([])
    type_array = np.array([])
    for attr, obj in category.__dict__.items():
        directory_list = obj.directory.split('\\')
        if attr == 'normal':
            data_class = 0
        else:
            data_class = 1

        for i, file in enumerate(obj.files):
            [name, type] = file.split('\\')[-1].split('.')
            class_array = np.append(class_array, data_class).astype(int)
            category_array = np.append(category_array, directory_list[-3])
            index_array = np.append(index_array, i).astype(int)
            name_array = np.append(name_array, name)
            type_array = np.append(type_array, type)

    dataframe = pd.DataFrame(data = np.transpose([index_array, class_array, category_array, name_array, type_array]), columns = ['dir index', 'class', 'category', 'file name', 'file type'])
    dataframe['dir index'] = dataframe['dir index'].astype(int)
    dataframe['class'] = dataframe['class'].astype(int)

    return dataframe

def ShuffleDataset(category, flatten):
    trainImages = np.concatenate((np.array([category.normal.get(x) for x in range(len(category.normal.files))]), np.array([category.pneumonia.get(x) for x in range(len(category.pneumonia.files))])), axis = 0)
    trainLabels = np.concatenate((np.array([0 for x in range(len(category.normal.files))]), np.array([1 for x in range(len(category.pneumonia.files))])), axis = 0)

    if flatten:
        img_count = len(category.normal.files) + len(category.pneumonia.files)
        trainImages = trainImages.reshape(img_count, -1)

    shufflePack = list(zip(trainImages, trainLabels))
    random.shuffle(shufflePack)
    trainImages, trainLabels = zip(*shufflePack)

    trainImages = np.array(trainImages)
    trainLabels = np.array(trainLabels)

    return trainImages, trainLabels

def CombineDataset(image_data, label_data):

    combi_data = np.array(list(map(lambda idx : (image_data[idx].reshape(480, 640), label_data[idx]), range(len(image_data)))))

    return combi_data

def SaveDataset(image_array, label_array, category):
    npy_directory = str(Path(__file__).parent.parent) + '\\npy_dataset\\'
    Path(npy_directory).mkdir(parents = True, exist_ok = True)
    np.save(npy_directory + category + '_images.npy', image_array)
    np.save(npy_directory + category + '_labels.npy', label_array)

def LoadDataset(category):
    npy_directory = str(Path(__file__).parent.parent) + '\\npy_dataset\\'
    image_data = np.load(npy_directory + category + '_images.npy')
    label_data = np.load(npy_directory + category + '_labels.npy')

    return image_data, label_data

def BatchSegmentation(dataset, batch_size):
    batch_size  =   int(batch_size)
    batch_directory = str(Path(__file__).parent.parent) + '\\training_batches\\'
    Path(batch_directory).mkdir(parents = True, exist_ok = True)

    no_batches  =   math.floor(len(dataset) / batch_size)
    no_images   =   int(len(dataset) / no_batches)

    print("No. of batch files : {:d}".format(no_batches))
    print("Images per batch file : {:d}".format(no_images))
    print("Unused files : " + str(trainImages.shape[0] - (batch_size * (math.floor(len(dataset) / batch_size)))))

    input('Continue? ')

    for idx, count in enumerate(range(0, math.floor(len(dataset) / batch_size))):
        temp_arr = dataset[idx * batch_size : (idx+1) * batch_size, :]
        print("Saving training_batch{:02d}.npy...".format(idx + 1))
        np.save(batch_directory + 'training_batch{:02d}.npy'.format(idx + 1), temp_arr)

def GetFactorial(num):
    print("Factorials of " + str(num) + " :")
    [print('{}  ({})'.format(idx, int(num/idx))) for idx in range(1,num) if num % idx == 0]

print("Creating dataset reference objects...")
train, val, test = DatasetCreation('dataset')
train, val, test = DatasetCreation('scaled_dataset')

print("Creating Pandas dataframes...")
trainDataframe = GeneratePandasData(train)
valDataframe = GeneratePandasData(val)
testDataframe = GeneratePandasData(test)
testDataframe = testDataframe.append(valDataframe, ignore_index = 1)

print("Shuffling datasets...")
trainImages, trainLabels = ShuffleDataset(train, flatten = False)
testImages, testLabels = ShuffleDataset(test, flatten = False)

print(trainImages.shape)
print(trainLabels.shape)

GetFactorial(trainImages.shape[0])

zipped_data = CombineDataset(trainImages, trainLabels)
BatchSegmentation(zipped_data, 5216 / 8)

# Beginning of data pre-processing pipeline:

# 1. Data cleansing                         - remove or correct corrupt values/instances
# 2. Instance selection and partitioning    - data shuffling between 'normal' and 'pneumonia'
# 3. Feature tuning                         - scaling and normalisation data to adjust values with skewed distribution
# 4. Representation transformation          - one-hot-encoding, etc. to categorise or numerise data values
# 5. Feature extraction                     - reducing the number of features by creating lower-dimensional data (PCA, embedding, extraction, hashing)
# 6. Feature selection                      - select subset of input features that have highest correlations to data using filters or wrapper methods
# 7. Feature construction                   - Creating new features through mathematical functions (e.g. polynomial expansion) and feature crossing to capture feature interactions

# NOTE: For images                          - Clipping, resizing, cropping, gaussian blur, canary filters, scaling

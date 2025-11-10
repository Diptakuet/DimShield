#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1,2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

print("-------------------")
print("####  Built with CUDA:", tf.test.is_built_with_cuda())
print("####  GPUs found    :", tf.config.list_physical_devices('GPU'))
print("-------------------")

# Dataset name
'''OLIVETTI
   - 400 grayscale images of faces, each 64x64 pixels
   - 10 classes (0-9) representing different individuals
   - Each image is already normalized to [0,1]
'''
DATASET = "OLIVETTI"

# Model type
'''CNN
   - Convolutional Neural Network for image classification
   - Uses 1D convolutional layers since images are flattened to 1D
'''
MODEL_TYPE = "CNN"
MODEL_PATH = f"/local/kat/LESLIE/Topic_Dimshield/{MODEL_TYPE}/{DATASET}"

# Pretrained model name
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"


# Load Dataset
def fetch_data():
    print(f"Loading {DATASET} dataset...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    x_all, y_all = data.images, data.target

    # Dataset feature size
    DATASET_FEATURE = x_all.shape[1]*x_all.shape[2]
    # Number of classes
    num_classes = y_all.max() + 1

    # There are 400 examples in the dataset and 4096 dimensions
    print("Original dataset:")
    print(f"x_all: {x_all.shape} images")
    print(f"y_all: {y_all.shape} images")
    print(f"Dataset Features: {DATASET_FEATURE}")
    print(f"Number of Classes: {num_classes}")
    print("-------------------")
    return x_all, y_all, DATASET_FEATURE, num_classes


# Split Dataset
def split_data(x_all, y_all):
    ''' First split: Org_img --> 80% (train+val) / 20% (test) 
        Second split: 80% (train+val) --> 75%(train) / 25%(val)
        Resulting in 60% of Org_Img for training, 20% for validation, and 20% for testing
    '''
    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2,random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,random_state=42, shuffle=True)

    print("Splitted Dataset (60%/20%/20%):")
    print(f"  x_train : {x_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  x_val  : {x_val.shape}")
    print(f"  y_val  : {y_val.shape}")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print("-------------------")
    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test, DATASET_FEATURE, num_classes, normalize=0):
    ''' 
    Reshape data to 1D for CNN input.
    If normalize=1, divide by 255 to scale to [0,1].
    '''
    if normalize:
        x_train = x_train / 255.0
        x_val   = x_val   / 255.0
        x_test  = x_test  / 255.0

    X_train = x_train.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_val   = x_val.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_test  = x_test.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    Y_val   = tf.keras.utils.to_categorical(y_val, num_classes)
    Y_test  = tf.keras.utils.to_categorical(y_test, num_classes)
    print("Processed Dataset:")
    print(f"  X_train : {X_train.shape}")
    print(f"  Y_train : {Y_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  Y_val   : {Y_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Y_test  : {Y_test.shape}")
    print("-------------------")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def plot_img(X,Y):
    '''
    Plot images in a grid format.
    Displays the first 64 images with their corresponding labels.
    '''

    # Define the dimensions of the plot grid 
    W_grid = 8
    L_grid = 8
    
    # subplot return the figure object and axes object
    # we can use the axes object to plot specific figures at various locations
    
    axes = plt.subplots(L_grid, W_grid, figsize = (17,17))    
    axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

    n_train = len(X) # get the length of the train dataset
    # Select a random number from 0 to n_train
    for i in range(0, W_grid * L_grid): # create evenly spaces variables 
        # read and display an image with the selected index    
        axes[i].imshow(X[i,1:])
        label_index = int(Y[i])
        axes[i].set_title(f"{label_index}", fontsize = 8)
        axes[i].axis('off')
    
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    
    
   
    
def main():

    # Load Dataset
    x_all, y_all, DATASET_FEATURE, num_classes = fetch_data() 
    # Split Dataset
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)

    # Preprocess data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test, DATASET_FEATURE, num_classes, normalize=0)
    
    # Load Pretrained model    
    pretrained_model = models.load_model(model_file)

    # Evaluate on the 20% test set
    loss, acc = pretrained_model.evaluate(X_test, Y_test, verbose=0)
    print("-------------------")
    print(f"\nCNN Olivetti-face Test Accuracy: {acc:.4f}")
    print("-------------------")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR


print("#######Built with CUDA:", tf.test.is_built_with_cuda())
print("#######GPUs found    :", tf.config.list_physical_devices('GPU'))

# Load and Split OLIVETTI Data
data = fetch_olivetti_faces(shuffle=True, random_state=42)
x_all, y_all = data.images, data.target

DATASET_FEATURE = x_all.shape[1]*x_all.shape[2]
num_classes = y_all.max() + 1

# There are 400 examples in the dataset and 4096 dimensions
print("Original dataset:")
print(f"x_all: {x_all.shape} images")
print(f"y_all: {y_all.shape} images")
print(f"Dataset Features: {DATASET_FEATURE}")
print(f"Number of Classes: {num_classes}")
print("-------------------")

# First split: Org_img --> 80% (train+val) / 20% (test) 
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2,random_state=42, shuffle=True)

# Second split: 80% (train+val) --> 75%(train) / 25%(val) == 60% of Org_Img (train) / 20% of Org_img (val)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,random_state=42, shuffle=True)

print("Splitted Dataset (60%/20%/20%):")
print(f"  x_train : {x_train.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  x_val  : {x_val.shape}")
print(f"  y_val  : {y_val.shape}")
print(f"  x_test  : {x_test.shape}")
print(f"  y_test  : {y_val.shape}")


def get_data(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Preprocess Data (Flatten)
    Olivetti data is already normalized to [0,1]
    """
    X_train = x_train.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_val  = x_val .reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_test  = x_test .reshape(-1, DATASET_FEATURE, 1).astype('float32')
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    Y_val  = tf.keras.utils.to_categorical(y_val, num_classes)
    Y_test  = tf.keras.utils.to_categorical(y_test, num_classes)
    print("Processed Dataset:")
    print(f"  X_train : {X_train.shape}")
    print(f"  Y_train : {Y_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  Y_val  : {Y_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Y_test  : {Y_test.shape}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def plot_img(X,Y):

    # Let's view more images in a grid format
    # Define the dimensions of the plot grid 
    W_grid = 8
    L_grid = 8
    
    # subplot return the figure object and axes object
    # we can use the axes object to plot specific figures at various locations
    
    fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))
    
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

    
    
# plot_img(x_val,y_val)    
    
# Preprocess data
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(x_train, y_train, x_val, y_val, x_test, y_test)    
    

# Build CNN Model
model = models.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(4096, 1)))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(40, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# Define EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,              # Stop if val_loss doesn’t improve after 5 epochs
    verbose=1,
    restore_best_weights=True
)

# 7) Train the CNN on the 80% portion (with 20% of it as validation)
history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, Y_val)
#     callbacks=[early_stopping]
)

# Evaluate on the 20% val set
val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
# print(f"Val loss: {test_loss:.4f}")
print(f"Val accuracy: {val_acc:.4f}")

# Evaluate on the 20% test set
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
# print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Save the entire trained CNN model
model.save("CNN_model_OLIVETTI.h5")
print("Model saved as CNN_model_OLIVETTI.h5")


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

print("####### Built with CUDA:", tf.test.is_built_with_cuda())
print("####### GPUs found    :", tf.config.list_physical_devices('GPU'))

# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_all = np.concatenate([x_train, x_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0).flatten()

DATASET_FEATURE = x_all.shape[1] * x_all.shape[2] * x_all.shape[3]
num_classes = 10
input_shape = x_all.shape[1:]  # (32, 32, 3)

print("Original dataset:")
print(f"x_all: {x_all.shape} images")
print(f"y_all: {y_all.shape} images")
print(f"Dataset Features: {DATASET_FEATURE}")
print(f"Number of Classes: {num_classes}")
print("-------------------")

# First split: 80% (train+val) / 20% (test)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, shuffle=True)

# Second split: 80% (train+val) --> 75%(train) / 25%(val) == 60% train / 20% val
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, shuffle=True)

print("Splitted Dataset (60%/20%/20%):")
print(f"  x_train : {x_train.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  x_val   : {x_val.shape}")
print(f"  y_val   : {y_val.shape}")
print(f"  x_test  : {x_test.shape}")
print(f"  y_test  : {y_test.shape}")

def get_data(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Preprocess Data
    """
    X_train = x_train.astype('float32') / 255.0
    X_val   = x_val.astype('float32') / 255.0
    X_test  = x_test.astype('float32') / 255.0
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
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def plot_img(X, Y):
    W_grid = 8
    L_grid = 8
    fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
    axes = axes.ravel()
    for i in range(0, W_grid * L_grid):
        axes[i].imshow(X[i])
        label_index = int(np.argmax(Y[i]))
        axes[i].set_title(f"{label_index}", fontsize=8)
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.4)
    plt.show()

# Preprocess data
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(x_train, y_train, x_val, y_val, x_test, y_test)

# Build a deeper CNN Model for CIFAR10
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping]
)

val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
print(f"Val accuracy: {val_acc:.4f}")

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

model.save("CNN_model_CIFAR10.h5")
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

print("-------------------")
print("####  Built with CUDA:", tf.test.is_built_with_cuda())
print("####  GPUs found    :", tf.config.list_physical_devices('GPU'))
print("-------------------")

DATASET = "CIFAR10"
MODEL_TYPE = "CNN"
MODEL_PATH = f"/local/kat/LESLIE/Topic_Dimshield/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"

def fetch_data():
    print(f"Loading {DATASET} dataset...")
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
    print(f"Input Shape: {input_shape}")
    print("-------------------")
    return x_all, y_all, DATASET_FEATURE, num_classes, input_shape

def split_data(x_all, y_all):
    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, shuffle=True)
    print("Splitted Dataset (60%/20%/20%):")
    print(f"  x_train : {x_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  x_val   : {x_val.shape}")
    print(f"  y_val   : {y_val.shape}")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print("-------------------")
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test, DATASET_FEATURE, num_classes, normalize=1):
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_val   = x_val.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0
    X_train = x_train.astype('float32')
    X_val   = x_val.astype('float32')
    X_test  = x_test.astype('float32')
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

def main():
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=1
    )

    pretrained_model = models.load_model(model_file)

    test_loss, test_acc = pretrained_model.evaluate(X_test, Y_test, verbose=0)
    print("-------------------")
    print(f"\nCNN CIFAR10 Test Accuracy: {test_acc:.4f}")
    print("-------------------")

if __name__ == "__main__":
    main()
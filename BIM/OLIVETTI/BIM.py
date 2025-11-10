from __future__ import absolute_import
from __future__ import print_function

import copy
import pandas as pd
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Disable eager execution so that tf.compat.v1.placeholder works
tf.compat.v1.disable_eager_execution()

from tqdm import tqdm
import cleverhans
from cleverhans.utils_tf import batch_eval
import keras.backend as K
from keras.models import load_model
# from keras.datasets import OLIVETTI
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

print("-------------------")
print("####  Built with CUDA:", tf.test.is_built_with_cuda())
print("####  GPUs found    :", tf.config.list_physical_devices('GPU'))
print("-------------------")

# Root path for the project
root_path = "/local/kat/LESLIE/Topic_Dimshield/"

# Dataset name
'''OLIVETTI
   - 400 grayscale images of faces, each 64x64 pixels
   - 10 classes (0-9) representing different individuals
   - Each image is already normalized to [0,1]
'''
DATASET = "OLIVETTI"

# Attack Type
ATTACK_NAME = "BIM"

# Path to the dataset for the attack
PATH_ADV_DATA  = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets"

# Model type
'''CNN
   - Convolutional Neural Network for image classification
   - Uses 1D convolutional layers since images are flattened to 1D
'''
MODEL_TYPE = "CNN"
MODEL_PATH = f"/local/kat/LESLIE/Topic_Dimshield/{MODEL_TYPE}/{DATASET}"

# Pretrained model name
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"

# Batch Size
BATCH_SIZE = 32
        
# Load Dataset
def fetch_data():
    print(f"Loading {DATASET} dataset...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    x_all, y_all = data.images, data.target

    # Dataset feature size
    DATASET_FEATURE = x_all.shape[1]*x_all.shape[2]
    # Number of classes
    num_classes = y_all.max() + 1
    # Input shape for reshaping
    input_shape = x_all.shape[1:]  # (height, width) tuple


    # There are 400 examples in the dataset and 4096 dimensions
    print("Original dataset:")
    print(f"x_all: {x_all.shape} images")
    print(f"y_all: {y_all.shape} images")
    print(f"Dataset Features: {DATASET_FEATURE}")
    print(f"Number of Classes: {num_classes}")
    print(f"Input Shape: {input_shape}")
    print("-------------------")
    return x_all, y_all, DATASET_FEATURE, num_classes, input_shape


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

def plot_img(X, Y, plot_path=f"{root_path}/{DATASET}/AE_Reconstructed_{DATASET}.png"):
    '''
    Plot images in a grid format.
    Displays the first 16 images with their corresponding labels.
    '''
    W_grid = 4
    L_grid = 4

    _, axes = plt.subplots(L_grid, W_grid, figsize=(10, 10))
    axes = axes.ravel()  # flatten the grid into a 1D array

    # Loop through the first 16 images and plot them
    for i in range(0, W_grid * L_grid):
        axes[i].imshow(X[i], cmap='gray')  # preserve original style/colors
        label_index = int(Y[i])
        axes[i].set_title(f"{label_index}", fontsize=8)
        axes[i].axis('off')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")



# print("CleverHans version:", cleverhans.__version__)

def fgsm(x, predictions, eps, clip_min=None, clip_max=None, y=None):
    if y is None:
        y = tf.cast(
            tf.equal(predictions,
                     tf.reduce_max(predictions, 1, keepdims=True)),
            tf.float32
        )
    y = y / tf.reduce_sum(y, 1, keepdims=True)
    logits = predictions
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    )
    grad = tf.gradients(loss, x)[0]
    signed_grad = tf.sign(grad)
    adv_x = tf.stop_gradient(x + eps * signed_grad)
    if clip_min is not None and clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    return adv_x


def basic_iterative_method(sess, model, X, Y, eps, eps_iter,
                           nb_iter=10, clip_min=None, clip_max=None, batch_size=BATCH_SIZE):
    # placeholders
    x = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    y = tf.compat.v1.placeholder(tf.float32, shape=(None,) + Y.shape[1:])
    results = np.zeros((nb_iter, X.shape[0]) + X.shape[1:], dtype=np.float32)

    X_adv = X.copy()
    X_min = X - eps
    X_max = X + eps

    print(f"Running BIM for eps_iter={eps_iter}…")
    its = defaultdict(lambda: nb_iter - 1)
    out = set()

    for i in tqdm(range(nb_iter), desc="BIM iters"):
        adv_tensor = fgsm(x, model(x),
                          eps=eps_iter,
                          clip_min=clip_min,
                          clip_max=clip_max,
                          y=y)
        # run one FGSM step
        X_adv, = batch_eval(
            sess, [x, y], [adv_tensor],
            [X_adv, Y],
            feed={K.learning_phase(): 0},
            args={'batch_size': batch_size}
        )
        # project back into allowed L∞ ball
        X_adv = np.clip(X_adv, X_min, X_max)
        results[i] = X_adv

        preds = np.argmax(
            model.predict(X_adv, batch_size=batch_size, verbose=0), axis=1
        )
        mis = np.where(preds != Y.argmax(axis=1))[0]
        for idx in mis:
            if idx not in out:
                its[idx] = i
                out.add(idx)

    return its, results


def craft_bim(sess, model, X, Y, batch_size,
              eps, eps_iter, clip_min, clip_max, path_data):
    print(f"\nCrafting {ATTACK_NAME} samples for eps={eps:.3f}…")
    its, results = basic_iterative_method(
        sess, model, X, Y, eps, eps_iter,
        clip_min=clip_min, clip_max=clip_max,
        batch_size=batch_size
    )
    # pick the first misclassification iteration per sample
    X_adv = np.array(
        [results[its[i], i] for i in range(len(Y))],
        dtype=np.float32
    )

    # evaluate
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print(f"  → Adversarial accuracy: {100*acc:.3f}%")

    # save the .npy
    fname = f"Adv_{ATTACK_NAME}_eps_{eps:.3f}.npy"
    np.save(os.path.join(path_data, fname), X_adv)
    print("  → Saved", fname)

    # compute average L2
    avg_l2 = np.linalg.norm(
        X_adv.reshape((len(X), -1)) - X.reshape((len(X), -1)),
        axis=1
    ).mean()
    print(f"  → Avg L2 perturbation: {avg_l2:.4f}\n")

    return avg_l2, acc


def gen_adv_instance(model_file, X, Y, batch_size,
                     eps, eps_iter, clip_min, clip_max, path_data):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    model = load_model(model_file)
    _, base_acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    print(f"Base test accuracy: {100*base_acc:.3f}%")

    avg_l2, adv_acc = craft_bim(
        sess, model, X, Y, batch_size,
        eps, eps_iter, clip_min, clip_max, path_data
    )
    sess.close()
    return avg_l2, adv_acc



def main():
    # Fetch and split data
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data() 
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)

    # Preprocess data (1D reshape and normalization)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=0
    )

    # BIM parameters
    eps_iter   = 1.0        # step size per iteration
    CLIP_MIN   = 0.0
    CLIP_MAX   = 1.0
 
    # Sweep epsilon 
    eps_start = 0.050
    eps_end = -0.005
    eps_step = 0.005
    eps_values = np.round(np.arange(eps_start, eps_end, -eps_step), 3)  # 0.050, 0.045, ..., 0.000

    results = []

    for eps in eps_values:
        l2, acc = gen_adv_instance(
            model_file, X_test, Y_test, BATCH_SIZE,
            eps, eps_iter, CLIP_MIN, CLIP_MAX, PATH_ADV_DATA
        )
        results.append([eps, l2, acc])

    # Save summary CSV
    with open(f"{ATTACK_NAME}_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epsilon', 'Average L2', 'Accuracy'])
        writer.writerows(results)

    print(f"Done! Results in {ATTACK_NAME}_results.csv")


if __name__ == "__main__":
     main()    # Run the main function




from __future__ import absolute_import
from __future__ import print_function

import copy
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

from tqdm import tqdm

from cleverhans.utils import other_classes
from cleverhans.utils_tf import batch_eval, model_argmax
from cleverhans.attacks_tf import (jacobian_graph, jacobian,
                                   apply_perturbations, saliency_map)
import keras.backend as K
from keras.models import load_model
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("-------------------")
print("####  Built with CUDA:", tf.test.is_built_with_cuda())
print("####  GPUs found    :", tf.config.list_physical_devices('GPU'))
print("-------------------")

root_path = "/local/kat/LESLIE/Topic_Dimshield/"
DATASET = "CIFAR10"
ATTACK_NAME = "JSMA"
PATH_ADV_DATA  = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets"
MODEL_TYPE = "CNN"
MODEL_PATH = f"/local/kat/LESLIE/Topic_Dimshield/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"
BATCH_SIZE = 64

def fetch_data():
    print(f"Loading {DATASET} dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0).flatten()
    DATASET_FEATURE = x_all.shape[1]*x_all.shape[2]*x_all.shape[3]
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
    # Flatten for JSMA (1D input)
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
    W_grid = 4
    L_grid = 4
    _, axes = plt.subplots(L_grid, W_grid, figsize=(10, 10))
    axes = axes.ravel()
    for i in range(0, W_grid * L_grid):
        axes[i].imshow(X[i])
        label_index = int(Y[i])
        axes[i].set_title(f"{label_index}", fontsize=8)
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

def jsma(sess, x, predictions, grads, sample, target, theta, gamma,
         increase, nb_classes, clip_min, clip_max, verbose=False):
    adv_x = copy.copy(sample)
    nb_features = np.prod(adv_x.shape[1:])  # np.product -> np.prod
    original_shape = adv_x.shape
    adv_x = np.reshape(adv_x, (1, nb_features))
    max_iters = np.floor(nb_features * gamma / 2)
    if increase:
        search_domain = set([i for i in range(nb_features) if adv_x[0, i] < clip_max])
    else:
        search_domain = set([i for i in range(nb_features) if adv_x[0, i] > clip_min])
    iteration = 0
    adv_x_original_shape = np.reshape(adv_x, original_shape)
    adv_x_img = adv_x_original_shape.reshape(-1, 32, 32, 3)
    current = model_argmax(sess, x, predictions, adv_x_img, feed={K.learning_phase(): 0})
    while (current != target and iteration < max_iters and len(search_domain) > 1):
        adv_x_original_shape = np.reshape(adv_x, original_shape)
        adv_x_img = adv_x_original_shape.reshape(-1, 32, 32, 3)
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_img,
                                              nb_features, nb_classes,
                                              feed={K.learning_phase(): 0})
        i, j, search_domain = saliency_map(
            grads_target, grads_others, search_domain, increase)
        adv_x = apply_perturbations(
            i, j, adv_x, increase, theta, clip_min, clip_max)
        adv_x_original_shape = np.reshape(adv_x, original_shape)
        adv_x_img = adv_x_original_shape.reshape(-1, 32, 32, 3)
        current = model_argmax(sess, x, predictions, adv_x_img, feed={K.learning_phase(): 0})
        iteration += 1
    percent_perturbed = float(iteration * 2) / nb_features
    if current == target:
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        return np.reshape(adv_x, original_shape), 0, percent_perturbed

def saliency_map_method(sess, model, X, Y, theta, gamma, clip_min, clip_max, batch_chunk=100):
    nb_classes = Y.shape[1]
    X_img = X.reshape(-1, 32, 32, 3)
    x = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X_img.shape[1:])
    grads = jacobian_graph(model(x), x, nb_classes)
    X_adv = np.zeros_like(X)
    num_samples = X.shape[0]
    for chunk_start in range(0, num_samples, batch_chunk):
        chunk_end = min(chunk_start + batch_chunk, num_samples)
        print(f"Processing chunk {chunk_start} to {chunk_end-1}")
        for i in tqdm(range(chunk_start, chunk_end)):
            current_class = int(np.argmax(Y[i]))
            target_class = np.random.choice(other_classes(nb_classes, current_class))
            sample_flat = X[i:(i+1)]
            X_adv_sample, _, _ = jsma(
                sess, x, model(x), grads, sample_flat, target_class, theta=theta,
                gamma=gamma, increase=True, nb_classes=nb_classes,
                clip_min=clip_min, clip_max=clip_max
            )
            X_adv[i] = X_adv_sample
    return X_adv

def craft_jsma(sess, model, X, Y, batch_size, gamma, clip_min, clip_max, path_data, batch_chunk=100):
    print(f"Crafting {ATTACK_NAME} adversarial samples for gamma={gamma:.3f}...")
    X_adv  = saliency_map_method(sess, model, X, Y, theta=1, gamma=gamma, clip_min=clip_min, clip_max=clip_max, batch_chunk=batch_chunk)
    X_adv_img = X_adv.reshape(-1, 32, 32, 3)
    _, acc = model.evaluate(X_adv_img, Y, batch_size=batch_size, verbose=0)
    print(f"  → Adversarial accuracy: {100*acc:.3f}%")
    fname = f"Adv_{ATTACK_NAME}_gamma_{gamma:.3f}.npy"
    np.save(os.path.join(path_data, fname), X_adv_img)
    print("  → Saved", fname)
    avg_l2 = np.linalg.norm(
        X_adv_img.reshape((len(X_adv_img), -1)) - X.reshape((len(X), -1)),
        axis=1
    ).mean()
    print(f"  → Avg L2 perturbation: {avg_l2:.4f}\n")
    return avg_l2, acc

def gen_adv_instance(model_file, X, Y, batch_size,
                     gamma, clip_min, clip_max, path_data, batch_chunk=100):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    model = load_model(model_file)
    X_img = X.reshape(-1, 32, 32, 3)
    _, base_acc = model.evaluate(X_img, Y, batch_size=batch_size, verbose=0)
    print(f"Base test accuracy: {100*base_acc:.3f}%")
    avg_l2, adv_acc = craft_jsma(
        sess, model, X, Y, batch_size,
        gamma, clip_min, clip_max, path_data, batch_chunk=batch_chunk
    )
    sess.close()
    return avg_l2, adv_acc

def main():
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=1
    )

    # Reduce test set to first 1000 images for attack generation (Due to time constraints)
    # # First 500 images
    # X_test = X_test[:500]
    # Y_test = Y_test[:500]
    # Second 500 images
    X_test = X_test[500:1000]
    Y_test = Y_test[500:1000]

    CLIP_MIN   = 0.0
    CLIP_MAX   = 1.0
    gamma_values = np.array([0.022, 0.02, 0.018, 0.016, 0.014, 0.01, 0.005, 0.001])
    results = []
    for gamma in gamma_values:
        l2, acc = gen_adv_instance(
            model_file, X_test, Y_test, BATCH_SIZE,
            gamma, CLIP_MIN, CLIP_MAX, PATH_ADV_DATA, batch_chunk=100
        )
        results.append([gamma, l2, acc])
    with open(f"{root_path}/{ATTACK_NAME}/{DATASET}/{ATTACK_NAME}_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Gamma', 'Average L2', 'Accuracy'])
        writer.writerows(results)
    print(f"Done! Results in {root_path}/{ATTACK_NAME}/{DATASET}/{ATTACK_NAME}_results.csv")

if __name__ == "__main__":
    main()
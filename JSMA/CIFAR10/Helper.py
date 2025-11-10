# This is a helper file for the JSMA attack on the CIFAR-10 dataset
# Running JSMA on CIFAR-10 is very slow, and requires a lot of memory
# For our experiment we created 500 adversarial test images at a time
# and later combined them into a single file
# This file contains functions to help with that process

# This is a helper file for combining JSMA adversarial images for CIFAR-10
# and evaluating them using the pretrained model, using the same functions as JSMA.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import csv

# ---- CONFIG ----
root_path = "/local/kat/LESLIE/Topic_Dimshield"
ATTACK_NAME = "JSMA"
DATASET = "CIFAR10"
MODEL_TYPE = "CNN"
MODEL_PATH = f"{root_path}/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"

# Directories containing the first and second 500 adversarial images
DIR1 = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets/first_500_images"
DIR2 = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets/second_500_images"
OUT_DIR = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets/"
os.makedirs(OUT_DIR, exist_ok=True)

# Gamma values used
gamma_values = [0.022, 0.02, 0.018, 0.016, 0.014, 0.01, 0.005, 0.001]

def fetch_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0).flatten()
    DATASET_FEATURE = x_all.shape[1]*x_all.shape[2]*x_all.shape[3]
    num_classes = 10
    input_shape = x_all.shape[1:]  # (32, 32, 3)
    return x_all, y_all, DATASET_FEATURE, num_classes, input_shape

def split_data(x_all, y_all):
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, shuffle=True)
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test, DATASET_FEATURE, num_classes, normalize=1):
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_val   = x_val.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0
    X_train = x_train.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_val   = x_val.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_test  = x_test.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    Y_val   = tf.keras.utils.to_categorical(y_val, num_classes)
    Y_test  = tf.keras.utils.to_categorical(y_test, num_classes)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def get_y_test_combined():
    # Use the same split as in JSMA.py to get Y_test for first and second 500 images
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
    _, _, _, _, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=1
    )
    # First 1000
    Y_test_combined = Y_test[:1000]
    return Y_test_combined

def main():
    Y_test_combined = get_y_test_combined()
    model = load_model(model_file)
    results = []

    for gamma in gamma_values:
        fname = f"Adv_{ATTACK_NAME}_gamma_{gamma:.3f}.npy"
        # Try both 3 and 2 decimal places for robustness
        path1 = os.path.join(DIR1, fname)
        path2 = os.path.join(DIR2, fname)
        if not os.path.exists(path1):
            fname2 = f"Adv_{ATTACK_NAME}_gamma_{gamma:.2f}.npy"
            path1 = os.path.join(DIR1, fname2)
        if not os.path.exists(path2):
            fname2 = f"Adv_{ATTACK_NAME}_gamma_{gamma:.2f}.npy"
            path2 = os.path.join(DIR2, fname2)
        if not (os.path.exists(path1) and os.path.exists(path2)):
            print(f"Missing files for gamma={gamma}: {path1}, {path2}")
            continue

        X_adv_1 = np.load(path1)  # shape (500, 32, 32, 3)
        X_adv_2 = np.load(path2)  # shape (500, 32, 32, 3)
        X_adv_combined = np.concatenate([X_adv_1, X_adv_2], axis=0)  # shape (1000, 32, 32, 3)

        # Save combined file
        out_path = os.path.join(OUT_DIR, f"Adv_{ATTACK_NAME}_gamma_{gamma:.3f}.npy")
        np.save(out_path, X_adv_combined)
        print(f"Saved combined adversarial file: {out_path}")

        # Evaluate accuracy
        loss, acc = model.evaluate(X_adv_combined, Y_test_combined, batch_size=64, verbose=0)
        print(f"Gamma={gamma:.3f}: Combined 1000 images — Loss={loss:.4f}, Accuracy={acc:.4f}")

        # Calculate average L2 perturbation (relative to original images)
        # Get original X_test images (first 1000, normalized)
        x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
        x_test = x_test.astype('float32') / 255.0
        X_test_original = x_test[:1000]
        avg_l2 = np.linalg.norm(
            X_adv_combined.reshape((len(X_adv_combined), -1)) - X_test_original.reshape((len(X_test_original), -1)),
            axis=1
        ).mean()

        results.append([gamma, avg_l2, acc])

    # Save results to CSV in the same format as JSMA.py
    csv_out = os.path.join(root_path, ATTACK_NAME, DATASET, "JSMA_results.csv")
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Gamma', 'Average L2', 'Accuracy'])
        writer.writerows(results)
    print(f"Saved results to {csv_out}")
    print("Done.")

if __name__ == "__main__":
    main()
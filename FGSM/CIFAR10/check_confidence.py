import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import csv
import glob
import re
from sklearn.model_selection import train_test_split

# --- CONFIG ---
DATASET = "CIFAR10"
root_path = "/local/kat/LESLIE/Topic_Dimshield"
MODEL_TYPE = "CNN"
MODEL_PATH = f"{root_path}/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/CNN_model_{DATASET}.h5"
ADV_DIR = f"{root_path}/FGSM/{DATASET}/Datasets"
OUT_DIR = f"{root_path}/FGSM/{DATASET}/Conf_Result"
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_data():
    print(f"Loading {DATASET} dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0).flatten()
    DATASET_FEATURE = np.prod(x_all.shape[1:])
    input_shape = x_all.shape[1:]
    return x_all, y_all, DATASET_FEATURE, input_shape

def split_data(x_all, y_all):
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, shuffle=True)
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess_data(x_test, y_test, DATASET_FEATURE, num_classes=10):
    x_test = x_test.astype('float32') / 255.0
    X_test = x_test.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    Y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return X_test, Y_test

def write_conf_csv(out_path, y_true, preds):
    pred_labels = np.argmax(preds, axis=1)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Index', 'Original Label', 'Predicted Label'] + [f'Conf_{i}' for i in range(preds.shape[1])]
        writer.writerow(header)
        for idx, (orig_label, pred_label, conf_row) in enumerate(zip(y_true, pred_labels, preds)):
            writer.writerow([idx, int(orig_label), int(pred_label)] + list(map(float, conf_row)))

def main():
    # Load data and labels
    x_all, y_all, DATASET_FEATURE, input_shape = fetch_data()
    _, _, _, _, x_test, y_test = split_data(x_all, y_all)
    X_test, Y_test = preprocess_data(x_test, y_test, DATASET_FEATURE, num_classes=10)
    x_test_img = x_test.astype('float32') / 255.0  # shape: (N, H, W, C)

    # Load pretrained model
    model = load_model(model_file)

    # 1) Create confidence_results_original.csv for clean test data
    print("Creating confidence_results_original.csv ...")
    preds_orig = model.predict(x_test_img, verbose=0)
    write_conf_csv(os.path.join(OUT_DIR, "confidence_results_original.csv"), y_test, preds_orig)
    print(f"Saved: {os.path.join(OUT_DIR, 'confidence_results_original.csv')}")

    # 2) Create confidence_results_eps_*.csv for every adversarial file in ADV_DIR
    print("Processing adversarial files ...")
    adv_files = sorted(glob.glob(os.path.join(ADV_DIR, "Adv_FGSM_eps_*.npy")))
    eps_re = re.compile(r"Adv_FGSM_eps_([0-9]*\.?[0-9]+)\.npy$")

    if not adv_files:
        print(f"No adversarial files found in: {ADV_DIR}")
        return

    for fpath in adv_files:
        fname = os.path.basename(fpath)
        m = eps_re.search(fname)
        if not m:
            print(f"Skipping (cannot parse epsilon): {fname}")
            continue
        eps_token = m.group(1)  # keep exact formatting from filename (e.g., 0.02, 0.020)

        # Load and scale adversarial inputs
        X_adv = np.load(fpath)
        if X_adv.ndim == 2:
            X_adv = X_adv.reshape((-1,) + input_shape)
        X_adv = X_adv.astype('float32')
        # If not normalized, normalize to [0, 1]
        if X_adv.max() > 1.5:
            X_adv = X_adv / 255.0


        preds_adv = model.predict(X_adv, verbose=0)
        out_csv = os.path.join(OUT_DIR, f"confidence_results_eps_{eps_token}.csv")
        write_conf_csv(out_csv, y_test, preds_adv)
        print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
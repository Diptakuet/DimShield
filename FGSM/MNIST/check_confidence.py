import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv

# --- CONFIG ---
DATASET = "MNIST"
root_path = "/local/kat/LESLIE/Topic_Dimshield"
MODEL_TYPE = "CNN"
EPSILON = 0.20  # Example epsilon value
MODEL_PATH = f"{root_path}/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/CNN_model_{DATASET}.h5"
csv_path = f"{root_path}/FGSM/{DATASET}/Conf_Result/confidence_results_eps_{EPSILON:.2f}.csv"
DATASET_FEATURE = 28 * 28  # MNIST images are 28x28

# Path to adversarial data (change epsilon as needed)
ADV_DIR = f"{root_path}/FGSM/{DATASET}/Datasets"
adv_path = os.path.join(ADV_DIR, f"Adv_fgsm_eps_{EPSILON:.2f}.npy")

def fetch_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test

# def split_data(x_all, y_all):
#     from sklearn.model_selection import train_test_split
#     x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, shuffle=True)
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, shuffle=True)
#     return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess_data(x_test, y_test, DATASET_FEATURE, num_classes=10):
    x_test = x_test.astype('float32') / 255.0
    X_test = x_test.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    Y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return X_test, Y_test

def main():
    # Load adversarial examples
    X_adv = np.load(adv_path)
    if X_adv.ndim == 2:
        X_adv = X_adv.reshape(-1, DATASET_FEATURE, 1)

    # Load Y_test and original labels
    x_test, y_test = fetch_data()
    # _, _, _, _, x_test, y_test = split_data(x_all, y_all)
    X_test, Y_test = preprocess_data(x_test, y_test, DATASET_FEATURE, num_classes=10)

    # # Ensure the number of samples matches
    # if X_adv.shape[0] != Y_test.shape[0]:
    #     min_len = min(X_adv.shape[0], Y_test.shape[0])
    #     X_adv = X_adv[:min_len]
    #     Y_test = Y_test[:min_len]
    #     y_test = y_test[:min_len]

    # Load pretrained model
    model = load_model(model_file)

    # Evaluate accuracy
    loss, acc = model.evaluate(X_adv, Y_test, verbose=0)
    print(f"Adversarial accuracy (epsilon={EPSILON}): {acc:.4f}")

    # Get predictions and confidence matrix
    preds = model.predict(X_adv)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)  # Confidence for predicted class

    print(f"Predicted labels shape: {pred_labels.shape}")
    print(f"Confidence matrix shape: {preds.shape}")

    # Save results to CSV in the same directory as this script
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ['Index', 'Original Label', 'Predicted Label'] + [f'Conf_{i}' for i in range(preds.shape[1])]
        writer.writerow(header)
        # Data rows
        for idx, (orig_label, pred_label, conf_row) in enumerate(zip(y_test, pred_labels, preds)):
            writer.writerow([idx, orig_label, pred_label] + list(conf_row))
    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()
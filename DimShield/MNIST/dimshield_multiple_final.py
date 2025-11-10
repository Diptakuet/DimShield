import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses, Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# Paths & Hyperparameters
# ------------------------------------------------
CNN_MODEL_PATH  = "/home/seonghun/Research/Research-son/DETool_MNIST/cnn_model_80_20_split.h5"
LATENT_DIM      = 50
EPOCHS          = 50
BATCH_SIZE      = 32
MAX_TOP_FILES   = None   # None -> use all files

# Attack directories and output prefixes
ATTACKS = [
    {"name": "FGSM", "adv_dir": "/home/seonghun/Research/Research-son/FGSM"},
    {"name": "BIM",  "adv_dir": "/home/seonghun/Research/Research-son/BIM"},
]

# ------------------------------------------------
# Load clean MNIST test set once
# ------------------------------------------------
(_, _), (x_cnn_test, y_cnn_test) = tf.keras.datasets.mnist.load_data()
x_cnn_test = x_cnn_test.reshape(-1, 784).astype("float32") / 255.0
y_cnn_test = tf.keras.utils.to_categorical(y_cnn_test, 10)

# ------------------------------------------------
# Load the pretrained CNN
# ------------------------------------------------
cnn_model = load_model(CNN_MODEL_PATH)
print("Loaded CNN model from", CNN_MODEL_PATH)

# ------------------------------------------------
# Helper: build a fresh autoencoder
# ------------------------------------------------
def build_autoencoder():
    encoder_in = layers.Input((784,))
    encoded    = layers.Dense(LATENT_DIM, activation="relu")(encoder_in)
    encoder    = Model(encoder_in, encoded, name="encoder")

    decoder_in = layers.Input((LATENT_DIM,))
    decoded    = layers.Dense(784, activation="sigmoid")(decoder_in)
    decoder    = Model(decoder_in, decoded, name="decoder")

    ae_in  = layers.Input((784,))
    ae_out = decoder(encoder(ae_in))
    ae     = Model(ae_in, ae_out, name="autoencoder")
    ae.compile(optimizer="adam", loss=losses.MeanSquaredError())
    return ae

# ------------------------------------------------
# Helper: evaluate raw & denoised accuracy on a test split
# ------------------------------------------------
def eval_on_split(ae, X_test, y_test):
    raw_inputs = X_test.reshape(-1, 784, 1)
    _, raw_acc = cnn_model.evaluate(raw_inputs, y_test, verbose=0)

    recon        = ae.predict(X_test, verbose=0)
    recon_inputs = recon.reshape(-1, 784, 1)
    _, den_acc   = cnn_model.evaluate(recon_inputs, y_test, verbose=0)

    return raw_acc, den_acc

# ------------------------------------------------
# Process one attack directory
# ------------------------------------------------
def process_attack(attack_name, ADV_DIR):
    print(f"\n=== Processing {attack_name} at {ADV_DIR} ===")
    # find & sort all adv files by eps descending
    all_files = glob.glob(os.path.join(ADV_DIR, f"Adv_{attack_name.lower()}_eps_*.npy"))
    eps_files = []
    for path in all_files:
        eps_str = os.path.basename(path).split("_eps_")[1].replace(".npy", "")
        eps_val = float(eps_str)
        eps_files.append((eps_val, path))
    eps_files.sort(key=lambda x: x[0], reverse=True)

    # individual results
    ind_res = []
    for eps_val, adv_path in eps_files:
        adv = np.load(adv_path)
        if adv.ndim == 3:
            adv = adv.reshape(adv.shape[0], -1)

        idx = np.arange(len(adv))
        _, test_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
        X_test = adv[test_idx]
        y_test = y_cnn_test[test_idx]

        ae = build_autoencoder()
        ae.fit(
            adv[np.setdiff1d(idx, test_idx)],  # train on the other 80%
            x_cnn_test[np.setdiff1d(idx, test_idx)],
            epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0
        )

        raw_acc, den_acc = eval_on_split(ae, X_test, y_test)
        print(f"[Individual {attack_name}] eps={eps_val:.2f} raw={raw_acc:.4f} den={den_acc:.4f}")
        ind_res.append({"eps": eps_val, "raw_acc": raw_acc, "denoised_acc": den_acc})

    pd.DataFrame(ind_res).sort_values("eps", ascending=False) \
        .to_csv(f"{attack_name.lower()}_individual_results.csv", index=False)
    print(f"Saved {attack_name.lower()}_individual_results.csv")

    # aggregated top-k
    max_k = len(eps_files) if MAX_TOP_FILES is None else min(MAX_TOP_FILES, len(eps_files))
    comb_res = []
    for k in range(1, max_k+1):
        top_k = eps_files[:k]
        Xtr, Xte, yte = [], [], []
        for _, adv_path in top_k:
            adv = np.load(adv_path)
            if adv.ndim == 3:
                adv = adv.reshape(adv.shape[0], -1)
            idx = np.arange(len(adv))
            train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
            Xtr.append(adv[train_idx])
            Xte.append(adv[test_idx])
            yte.append(y_cnn_test[test_idx])
        X_train = np.vstack(Xtr)
        X_test  = np.vstack(Xte)
        y_test  = np.vstack(yte)

        ae = build_autoencoder()
        # use x_cnn_test for clean targets similarly
        x_clean_train = np.vstack([x_cnn_test[train_idx] for _, adv_path in top_k for train_idx, _ in [train_test_split(np.arange(len(np.load(adv_path))), test_size=0.2, random_state=42)]])
        ae.fit(X_train, x_clean_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)

        raw_acc, den_acc = eval_on_split(ae, X_test, y_test)
        eps_list = ";".join(f"{e:.2f}" for e,_ in top_k)
        print(f"[Combined {attack_name} k={k}] eps=[{eps_list}] raw={raw_acc:.4f} den={den_acc:.4f}")
        comb_res.append({"num_files": k, "eps_values": eps_list, "raw_acc": raw_acc, "denoised_acc": den_acc})

    pd.DataFrame(comb_res) \
        .to_csv(f"{attack_name.lower()}_combined_results.csv", index=False)
    print(f"Saved {attack_name.lower()}_combined_results.csv")

# ------------------------------------------------
# Main entry
# ------------------------------------------------
if __name__ == "__main__":
    for atk in ATTACKS:
        process_attack(atk["name"], atk["adv_dir"])


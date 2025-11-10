import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses, Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces



# ------------------------------------------------
# Paths & Hyperparameters
# ------------------------------------------------

# LESLIE - changed the CNN_MODEL_PATH to my local one
CNN_MODEL_PATH  = "/local/kat/LESLIE/Topic_Dimshield/CNN/OLIVETTI/cnn_model_80_20_split_again.h5"
LATENT_DIM      = 50
EPOCHS          = 50
BATCH_SIZE      = 32
MAX_TOP_FILES   = None   # None -> use all files

DATASET_FEATURE = 4096
DATASET_TARGET = 40

# Attack directories and output prefixes
ATTACKS = [
    {"name": "FGSM", "adv_dir": "/local/kat/LESLIE/Topic_Dimshield/FGSM/OLIVETTI/Datasets"},
    {"name": "BIM",  "adv_dir": "/local/kat/LESLIE/Topic_Dimshield/BIM/OLIVETTI/Datasets"},
    {"name": "JSMA",  "adv_dir": "/local/kat/LESLIE/Topic_Dimshield/JSMA/OLIVETTI/Datasets"},
]

# ------------------------------------------------
# Load clean OLIVETTI test set once
# ------------------------------------------------
data = fetch_olivetti_faces(shuffle=True, random_state=42)
x_all, y_all = data.images, data.target

def get_jsma_data(x_train,y_train,x_val,y_val,x_test,y_test):  
    """Prepare OLIVETTI data for Conv1D FGSM."""
    X_train = x_train.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_test = x_test.reshape(-1, DATASET_FEATURE, 1).astype('float32') 
    X_val = X_test  # Assuming validation data is the same as test data

    # One-hot encode the labels
    Y_train = tf.keras.utils.to_categorical(y_train, DATASET_TARGET)
    Y_test = tf.keras.utils.to_categorical(y_test, DATASET_TARGET)
    Y_val = Y_test

    X_test = X_test[0:1000,:,:]
    Y_test = Y_test[0:1000, :]

    print("X_train:", X_train.shape)  # X_train: (240, 4096, 1)
    print("Y_train:", Y_train.shape)  # Y_train: (240, 40)
    print("X_val:", X_val.shape)      # X_val: (80, 4096, 1)
    print("Y_val:", Y_val.shape)      # Y_val: (80, 40)
    print("X_test:", X_test.shape)    # X_test: (80, 4096, 1)
    print("Y_test:", Y_test.shape)    # Y_test: (80, 40)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def get_fgsm_bim_data(x_train,y_train, x_test,y_test):  
    """Prepare OLIVETTI data for Conv1D FGSM."""
    X_train = x_train.reshape(-1, DATASET_FEATURE, 1).astype('float32')
    X_test  = x_test .reshape(-1, DATASET_FEATURE, 1).astype('float32')
    Y_train = tf.keras.utils.to_categorical(y_train, DATASET_TARGET)
    Y_test  = tf.keras.utils.to_categorical(y_test, DATASET_TARGET)
    return X_train, Y_train, X_test, Y_test

# ------------------------------------------------
# Load the pretrained CNN
# ------------------------------------------------
cnn_model = load_model(CNN_MODEL_PATH)
print("Loaded CNN model from", CNN_MODEL_PATH)

# ------------------------------------------------
# Helper: build a fresh autoencoder
# ------------------------------------------------
def build_autoencoder():
    encoder_in = layers.Input((DATASET_FEATURE,))
    encoded    = layers.Dense(LATENT_DIM, activation="relu")(encoder_in)
    encoder    = Model(encoder_in, encoded, name="encoder")

    decoder_in = layers.Input((LATENT_DIM,))
    decoded    = layers.Dense(DATASET_FEATURE, activation="sigmoid")(decoder_in)
    decoder    = Model(decoder_in, decoded, name="decoder")

    ae_in  = layers.Input((DATASET_FEATURE,))
    ae_out = decoder(encoder(ae_in))
    ae     = Model(ae_in, ae_out, name="autoencoder")
    ae.compile(optimizer="adam", loss=losses.MeanSquaredError())
    return ae

# ------------------------------------------------
# Helper: evaluate raw & denoised accuracy on a test split
# ------------------------------------------------
def eval_on_split(ae, X_test, y_test):
    raw_inputs = X_test.reshape(-1, DATASET_FEATURE, 1)
    _, raw_acc = cnn_model.evaluate(raw_inputs, y_test, verbose=0)

    recon        = ae.predict(X_test, verbose=0)
    recon_inputs = recon.reshape(-1, DATASET_FEATURE, 1)
    _, den_acc   = cnn_model.evaluate(recon_inputs, y_test, verbose=0)

    return raw_acc, den_acc

# ------------------------------------------------
# Process one attack directory
# ------------------------------------------------
def process_attack(attack_name, ADV_DIR):
    print(f"\n=== Processing {attack_name} at {ADV_DIR} ===")
    attack_parameter_files = []
    gamma_or_eps = ""
    
    # if attack is JSMA, set the indicator variables to the correct gamma or eps
    if attack_name == 'JSMA':
        gamma_or_eps = 'gamma'
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,test_size=0.2, shuffle = True, random_state = 42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.25, shuffle = True, random_state = 42)
        
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_jsma_data(x_train,y_train,x_val,y_val,x_test,y_test)

    else: 
        gamma_or_eps = "eps"
        X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, shuffle=True)
        X_train, Y_train, X_test, Y_test = get_fgsm_bim_data(X_train, y_train, X_test, y_test)

    x_cnn_test = X_test
    y_cnn_test = Y_test

    print(f"x test shape: {x_cnn_test.shape}")
    print(f"y test shape: {y_cnn_test.shape}")

    # find & sort all adv files by eps descending
    all_files = glob.glob(os.path.join(ADV_DIR, f"Adv_{attack_name.lower()}_{gamma_or_eps}_*.npy"))

    for path in all_files:
        parameter_str = os.path.basename(path).split(f"_{gamma_or_eps}_")[1].replace(".npy", "")
        parameter_val = float(parameter_str)
        attack_parameter_files.append((parameter_val, path))
    
    # sort the files from largest to smallest
    attack_parameter_files.sort(key=lambda x: x[0], reverse=True)

    print(f"All attack files {attack_parameter_files}")

    # individual results
    ind_res = []
    for parameter_val, adv_path in attack_parameter_files:
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

        print(f"X_TEST shape: {X_test.shape}")
        print(f"Y_TEST shape: {y_test.shape}")

        raw_acc, den_acc = eval_on_split(ae, X_test, y_test)

        print(f"Gamma values: {parameter_val}")

        print(f"[Individual {attack_name}] {gamma_or_eps}={parameter_val:.2f} raw={raw_acc:.4f} den={den_acc:.4f}")
        ind_res.append({gamma_or_eps: parameter_val, "raw_acc": raw_acc, "denoised_acc": den_acc})

    pd.DataFrame(ind_res).sort_values(gamma_or_eps, ascending=False) \
        .to_csv(f"{attack_name.lower()}_individual_results.csv", index=False)
    print(f"Saved {attack_name.lower()}_individual_results.csv")

    # aggregated top-k
    max_k = len(attack_parameter_files) if MAX_TOP_FILES is None else min(MAX_TOP_FILES, len(attack_parameter_files))
    comb_res = []
    for k in range(1, max_k+1):
        top_k = attack_parameter_files[:k]
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
        attack_parameter_list = ";".join(f"{e:.2f}" for e,_ in top_k)

        print(f"attack_parameter_list: {attack_parameter_list}")
    
        print(f"[Combined {attack_name} k={k}] {gamma_or_eps}=[{attack_parameter_list}] raw={raw_acc:.4f} den={den_acc:.4f}")
        comb_res.append({"num_files": k, f"{gamma_or_eps}_values": attack_parameter_list, "raw_acc": raw_acc, "denoised_acc": den_acc})
        
    pd.DataFrame(comb_res) \
        .to_csv(f"{attack_name.lower()}_combined_results.csv", index=False)
    print(f"Saved {attack_name.lower()}_combined_results.csv")

# ------------------------------------------------
# Main entry
# ------------------------------------------------
if __name__ == "__main__":
    for atk in ATTACKS:
        process_attack(atk["name"], atk["adv_dir"])


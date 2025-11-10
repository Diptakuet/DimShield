##############################################
# For Now: Just use it for C&W attack on MNIST 
# For other Attacks go ahead with dimshield_multiple_final.py
#  "Need to come back and clean up the code"
##############################################
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("-------------------")
print("####  Built with CUDA:", tf.test.is_built_with_cuda())
print("####  GPUs found    :", tf.config.list_physical_devices('GPU'))
print("-------------------")

# ------------------------------------------------
# Paths & Hyperparameters
# ------------------------------------------------

DATASET = "MNIST"
root_path = "/local/kat/LESLIE/Topic_Dimshield"

MODEL_TYPE = "CNN"
MODEL_PATH = f"{root_path}/{MODEL_TYPE}/{DATASET}"

model_file = f"{MODEL_PATH}/CNN_model_{DATASET}.h5"


# Read latent dimension from CSV
id_csv_path = f"{root_path}/ID_Estimation/{DATASET}/AE_estimated_id_{DATASET}.csv"
LATENT_DIM = int(pd.read_csv(id_csv_path)["estimated_id"].iloc[0])
print("-------------------")
print(f"Using latent dimension: {LATENT_DIM} from {id_csv_path}")
print("-------------------")

# Autoencoder model file path
ae_model_file = f"{root_path}/ID_Estimation/{DATASET}/AE_model_{DATASET}_id_{LATENT_DIM}.h5"

# Attack directories and output prefixes
ATTACKS = [
    # {"name": "FGSM", "adv_dir": f"{root_path}/FGSM/{DATASET}/Datasets"},
    # {"name": "BIM",  "adv_dir": f"{root_path}/BIM/{DATASET}/Datasets"},
    {"name": "JSMA",  "adv_dir": f"{root_path}/JSMA/{DATASET}/Datasets"},
    # {"name": "CW",   "adv_dir": f"{root_path}/CW/{DATASET}/Datasets"},
]


# ------------------------------------------------

def fetch_data():
    print(f"Loading {DATASET} dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    DATASET_FEATURE = x_all.shape[1] * x_all.shape[2]  # 28*28=784
    num_classes = 10
    input_shape = x_all.shape[1:]  # (28, 28)
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

def build_simple_ae(latent_dim, DATASET_FEATURE):
    inp  = layers.Input(shape=(DATASET_FEATURE,))
    bott = layers.Dense(latent_dim, activation='relu')(inp)
    up   = layers.Dense(DATASET_FEATURE, activation='sigmoid')(bott)
    ae   = models.Model(inp, up, name=f"AE_1D_{latent_dim}")
    ae.compile(optimizer='adam', loss='mse')
    return ae

def evaluate_dimshield(X_adv, Y_adv, autoencoder, pretrained_model, DATASET_FEATURE):
    X_recon_1d = autoencoder.predict(X_adv, verbose=0)
    X_adv_recon = X_recon_1d.reshape(-1, DATASET_FEATURE, 1)
    loss_adv, acc_adv = pretrained_model.evaluate(X_adv, Y_adv, verbose=0)
    print("-------------------")
    print(f"[- Dimshield] Adversarial data:→ Loss={loss_adv:.4f}, Accuracy={acc_adv:.4f}")
    loss_recon, acc_recon = pretrained_model.evaluate(X_adv_recon, Y_adv, verbose=0)
    print(f"[+ Dimshield] Adversarial data:→ Loss={loss_recon:.4f}, Accuracy={acc_recon:.4f}")
    print("-------------------")
    return acc_adv, acc_recon, loss_adv, loss_recon

def process_attack(attack_name, ADV_DIR, autoencoder, pretrained_model, DATASET_FEATURE, Y_test):
    print(f"\n=== Processing {attack_name} at {ADV_DIR} ===")
    results = []

    if attack_name == "CW":
        all_files = glob.glob(os.path.join(ADV_DIR, f"Adv_{attack_name}_iters_*.npy"))
        param_files = []
        for path in all_files:
            iters_str = os.path.basename(path).split("_iters_")[1].replace(".npy", "")
            iters_val = int(iters_str)
            param_files.append((iters_val, path))
        param_files.sort(key=lambda x: x[0])
        param_name = "Iterations"

    elif attack_name == "JSMA":
        all_files = glob.glob(os.path.join(ADV_DIR, f"Adv_{attack_name}_gamma_*.npy"))
        param_files = []
        for path in all_files:
            gamma_str = os.path.basename(path).split("_gamma_")[1].replace(".npy", "")
            gamma_val = float(gamma_str)
            param_files.append((gamma_val, path))
        param_files.sort(key=lambda x: x[0])
        param_name = "Gamma"

    else:
        all_files = glob.glob(os.path.join(ADV_DIR, f"Adv_{attack_name}_eps_*.npy"))
        param_files = []
        for path in all_files:
            eps_str = os.path.basename(path).split("_eps_")[1].replace(".npy", "")
            eps_val = float(eps_str)
            param_files.append((eps_val, path))
        param_files.sort(key=lambda x: x[0])
        param_name = "Epsilon"

    for param_val, adv_path in param_files:
        X_adv = np.load(adv_path)
        if X_adv.ndim == 2:
            X_adv = X_adv.reshape(-1, DATASET_FEATURE, 1)
        acc_adv, acc_recon, loss_adv, loss_recon = evaluate_dimshield(
            X_adv, Y_test, autoencoder, pretrained_model, DATASET_FEATURE
        )
        print(f"[{attack_name}] {param_name}={param_val} w/o DimShield (Acc.)={acc_adv:.4f} w DimShield (Acc.)={acc_recon:.4f} w/o DimShield (Loss.)={loss_adv:.4f} w DimShield (Loss.)={loss_recon:.4f}")
        results.append({
            param_name: f"{param_val}",
            "w/o DimShield (Acc.)": f"{acc_adv:.4f}",
            "w DimShield (Acc.)": f"{acc_recon:.4f}",
            "w/o DimShield (Loss.)": f"{loss_adv:.4f}",
            "w DimShield (Loss.)": f"{loss_recon:.4f}"
        })

    out_csv = f"{root_path}/DimShield/{DATASET}/Result/DimShield_Result_{attack_name}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

def main():
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=1
    )

    for attack in ATTACKS:
        if attack["name"] == "JSMA":
            # Reduce test set to first 1000 images for attack generation (Due to time constraints)
            X_test = X_test[:1000]
            Y_test = Y_test[:1000]

    pretrained_model = models.load_model(model_file)
    print(f"Loaded pretrained model from {model_file}")
    ae_model_file = f"{root_path}/ID_Estimation/{DATASET}/AE_model_{DATASET}_id_{LATENT_DIM}.h5"
    #####  Comment out Train Autoencoder if AE is already trained  #####
    #####          Train Autoencoder  ###########
    # batch_size = 64
    # ae = build_simple_ae(LATENT_DIM, DATASET_FEATURE)
    # es = EarlyStopping(
    #     monitor='val_loss',
    #     patience=10,
    #     restore_best_weights=True,
    #     verbose=1
    # )
    # print(f"Training AE model with latent dimension {LATENT_DIM}...")
    # ae.fit(
    #     X_train, X_train,
    #     validation_data=(X_val, X_val),
    #     epochs=100,
    #     batch_size=batch_size,
    #     callbacks=[es],
    #     verbose=2
    # )
    # ae.save(ae_model_file)
    # print(f"Saved trained AE model to {ae_model_file}")
    # print("-------------------")
    ################ End Of Train Autoencoder ##################################
    
    autoencoder = models.load_model(ae_model_file)
    print(f"Loaded AE model from {ae_model_file}")
    print("-------------------")

    for attack in ATTACKS:
        process_attack(
            attack["name"], attack["adv_dir"],
            autoencoder, pretrained_model,
            DATASET_FEATURE, Y_test
        )

if __name__ == "__main__":
    main()
# # ------------------------------------------------
# # Process one attack directory
# # ------------------------------------------------
# def process_attack(attack_name, ADV_DIR):
#     print(f"\n=== Processing {attack_name} at {ADV_DIR} ===")
#     # find & sort all adv files by eps descending
#     all_files = glob.glob(os.path.join(ADV_DIR, f"Adv_{attack_name.lower()}_eps_*.npy"))
#     eps_files = []
#     for path in all_files:
#         eps_str = os.path.basename(path).split("_eps_")[1].replace(".npy", "")
#         eps_val = float(eps_str)
#         eps_files.append((eps_val, path))
#     eps_files.sort(key=lambda x: x[0], reverse=True)

#     # individual results
#     ind_res = []
#     for eps_val, adv_path in eps_files:
#         adv = np.load(adv_path)
#         adv_imgs = adv.reshape(-1,64,64,1)
#         # if adv.ndim == 3:
#         #     adv = adv.reshape(adv.shape[0], -1)

#         idx = np.arange(len(adv))
#         train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

#         X_test = adv_imgs[test_idx]
#         y_test = y_cnn_test[test_idx]

#         X_train_noisy = adv_imgs[train_idx]
#         X_train_clean = x_cnn_test_imgs[train_idx]

#         ae = build_autoencoder()
#         ae.fit(
#             X_train_noisy,  # train on the other 80%
#             X_train_clean,
#             epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0
#         )

#         raw_acc, den_acc = eval_on_split(ae, X_test, y_test)
#         print(f"[Individual {attack_name}] eps={eps_val:.2f} raw={raw_acc:.4f} den={den_acc:.4f}")
#         ind_res.append({"eps": eps_val, "raw_acc": raw_acc, "denoised_acc": den_acc})

#     pd.DataFrame(ind_res).sort_values("eps", ascending=False) \
#         .to_csv(f"{attack_name.lower()}_individual_results.csv", index=False)
#     print(f"Saved {attack_name.lower()}_individual_results.csv")

#     # aggregated top-k
#     max_k = len(eps_files) if MAX_TOP_FILES is None else min(MAX_TOP_FILES, len(eps_files))
#     comb_res = []
#     for k in range(1, max_k+1):
#         top_k = eps_files[:k]
#         Xtr, Xte, yte = [], [], []
#         for _, adv_path in top_k:
#             adv = np.load(adv_path)
#             if adv.ndim == 3:
#                 adv = adv.reshape(adv.shape[0], -1)
#             idx = np.arange(len(adv))
#             train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
#             Xtr.append(adv[train_idx])
#             Xte.append(adv[test_idx])
#             yte.append(y_cnn_test[test_idx])
#         X_train = np.vstack(Xtr)
#         X_test  = np.vstack(Xte)
#         y_test  = np.vstack(yte)

#         ae = build_autoencoder()
#         # use x_cnn_test for clean targets similarly
#         x_clean_train = np.vstack([x_cnn_test[train_idx] for _, adv_path in top_k for train_idx, _ in [train_test_split(np.arange(len(np.load(adv_path))), test_size=0.2, random_state=42)]])
#         ae.fit(X_train, x_clean_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)

#         raw_acc, den_acc = eval_on_split(ae, X_test, y_test)
#         eps_list = ";".join(f"{e:.2f}" for e,_ in top_k)
#         print(f"[Combined {attack_name} k={k}] eps=[{eps_list}] raw={raw_acc:.4f} den={den_acc:.4f}")
#         comb_res.append({"num_files": k, "eps_values": eps_list, "raw_acc": raw_acc, "denoised_acc": den_acc})

#     pd.DataFrame(comb_res) \
#         .to_csv(f"{attack_name.lower()}_combined_results.csv", index=False)
#     print(f"Saved {attack_name.lower()}_combined_results.csv")

# ------------------------------------------------
# Main entry
# ------------------------------------------------
# if __name__ == "__main__":
#     for atk in ATTACKS:
#         process_attack(atk["name"], atk["adv_dir"])


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import sys

# --- Config ---
root_path = "/local/kat/LESLIE/Topic_Dimshield/"
DATASET = "OLIVETTI"
ATTACK_NAME = "JSMA"
MODEL_TYPE = "CNN"
MODEL_PATH = f"{root_path}/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"
PATH_ADV_DATA = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets"

# --- Parameters ---
if ATTACK_NAME == "FGSM" or ATTACK_NAME == "BIM":
    eps = 0.007  # <-- Set your epsilon value here
    adv_file = f"Adv_{ATTACK_NAME}_eps_{eps:.3f}.npy"
    plot_path = f"{root_path}/{ATTACK_NAME}/verify/{ATTACK_NAME}_plot_eps_{eps:.3f}.png"

if ATTACK_NAME == "JSMA":
    gamma = 0.000  # <-- Set your gamma value here
    adv_file = f"Adv_{ATTACK_NAME}_gamma_{gamma:.3f}.npy"
    plot_path = f"{root_path}/{ATTACK_NAME}/verify/{ATTACK_NAME}_plot_gamma_{gamma:.3f}.png"


# --- Load and preprocess original data ---
# Add parent FGSM directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{DATASET}/")))
if ATTACK_NAME == "FGSM":
    from FGSM import fetch_data, split_data, preprocess_data  # Import functions for consistency
if ATTACK_NAME == "BIM":
    from BIM import fetch_data, split_data, preprocess_data  # Import functions for consistency
if ATTACK_NAME == "JSMA":
    from JSMA import fetch_data, split_data, preprocess_data  # Import functions for consistency

    
x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
    x_train, y_train, x_val, y_val, x_test, y_test,
    DATASET_FEATURE, num_classes, normalize=0
)

# --- Load adversarial data ---
adv_path = os.path.join(PATH_ADV_DATA, adv_file)
X_adv = np.load(adv_path)  # shape: (N, 4096, 1)
X_adv_img = X_adv.reshape(-1, *input_shape)  # shape: (N, 64, 64)

# --- Get first 10 images and labels ---
orig_imgs = x_test[:10]
orig_labels = y_test[:10]
adv_imgs = X_adv_img[:10]

# --- Load model and predict categories for adversarial images ---
model = load_model(model_file)
adv_imgs_1d = X_adv[:10]  # shape: (10, 4096, 1)
preds = model.predict(adv_imgs_1d, batch_size=10, verbose=0)
pred_labels = np.argmax(preds, axis=1)

# --- Plot ---
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

for i in range(10):

    # First row: original images
    axes[0, i].imshow(orig_imgs[i], cmap='gray')
    axes[0, i].set_title(f"True: {orig_labels[i]}", fontsize=8)
    axes[0, i].axis('off')
    # Second row: adversarial images
    axes[1, i].imshow(adv_imgs[i], cmap='gray')
    axes[1, i].set_title(f"Pred: {pred_labels[i]}", fontsize=8)
    axes[1, i].axis('off')

if ATTACK_NAME in ["FGSM", "BIM"]:
    plt.suptitle(f"{ATTACK_NAME} Adversarial vs Original Images (epsilon={eps})", fontsize=14)
if ATTACK_NAME == "JSMA":
    plt.suptitle(f"{ATTACK_NAME} Adversarial vs Original Images (gamma={gamma})", fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(plot_path)
plt.close()
print(f"Saved plot to {plot_path}")

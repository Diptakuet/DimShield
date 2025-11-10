# CW.py
from __future__ import absolute_import, print_function

import os
import csv
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Quiet TF logs early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF1 graph semantics (CleverHans expects this)
tf.compat.v1.disable_eager_execution()

# Keras/tf.keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# ---- TF1→TF2 shims (MUST be before CleverHans imports) ----
if not hasattr(tf, "Session"):
    tf.Session = tf.compat.v1.Session
if not hasattr(tf, "placeholder"):
    tf.placeholder = tf.compat.v1.placeholder
if not hasattr(tf, "get_default_graph"):
    tf.get_default_graph = tf.compat.v1.get_default_graph
if not hasattr(tf, "global_variables_initializer"):
    tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
if not hasattr(tf, "reset_default_graph"):
    tf.reset_default_graph = tf.compat.v1.reset_default_graph
if not hasattr(tf, "global_variables"):
    tf.global_variables = tf.compat.v1.global_variables
if not hasattr(tf, "variables_initializer"):
    tf.variables_initializer = tf.compat.v1.variables_initializer
# Make sure TF1 optimizers exist where CleverHans looks
if not hasattr(tf.train, "AdamOptimizer"):
    tf.train = tf.compat.v1.train
# tf.py_func shim
if not hasattr(tf, "py_func"):
    try:
        tf.py_func = tf.compat.v1.py_func
    except AttributeError:
        def _py_func(func, inp, Tout, stateful=True, name=None):
            return tf.numpy_function(func, inp, Tout, name=name)
        tf.py_func = _py_func

# CleverHans
try:
    from cleverhans.attacks import CarliniWagnerL2
except Exception:
    from cleverhans.attacks_tf import CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper

# Data
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Our helpers
from logits_utils import prepare_for_cleverhans

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
root_path = "/local/kat/LESLIE/Topic_Dimshield/"
DATASET = "OLIVETTI"
ATTACK_NAME = "CW"

PATH_ADV_DATA = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets"

MODEL_TYPE = "CNN"
MODEL_PATH = f"/local/kat/LESLIE/Topic_Dimshield/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"

BATCH_SIZE = 32
CLIP_MIN, CLIP_MAX = 0.0, 1.0

# Keep these constant
KAPPA = 0                # margin (confidence)
INITIAL_CONST = 1        # trade-off c
BINARY_SEARCH_STEPS = 3
LEARNING_RATE = 5e-3

# === You only specify these three; sweep is generated automatically (END is inclusive) ===
MAX_ITERS_START = 1        # e.g., 5
MAX_ITERS_END   = 22       # e.g., 50
MAX_ITERS_STEP  = 1        # e.g., 5

def build_iters_sweep(start: int, end: int, step: int):
    """Inclusive range builder; supports positive or negative step."""
    start, end, step = int(start), int(end), int(step)
    if step == 0:
        raise ValueError("MAX_ITERS_STEP must be non-zero.")
    vals = []
    i = start
    if step > 0:
        while i <= end:
            vals.append(int(i))
            i += step
    else:
        while i >= end:
            vals.append(int(i))
            i += step
    if not vals:
        raise ValueError(f"No values generated for max_iterations: start={start}, end={end}, step={step}")
    return vals

MAX_ITERS_SWEEP = build_iters_sweep(MAX_ITERS_START, MAX_ITERS_END, MAX_ITERS_STEP)

# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------
def fetch_data():
    print(f"Loading {DATASET} dataset...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    x_all, y_all = data.images, data.target
    DATASET_FEATURE = x_all.shape[1] * x_all.shape[2]
    num_classes = int(y_all.max()) + 1
    input_shape = x_all.shape[1:]  # (64, 64)
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
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, shuffle=True
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42, shuffle=True
    )
    print("Splitted Dataset (60%/20%/20%):")
    print(f"  x_train : {x_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  x_val   : {x_val.shape}")
    print(f"  y_val   : {y_val.shape}")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print("-------------------")
    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test,
                    DATASET_FEATURE, num_classes, normalize=0):
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

# -----------------------------------------------------------------------------
# C&W attack (sweeping max_iterations)
# -----------------------------------------------------------------------------
def craft_cw(sess, keras_model, X, Y, batch_size, kappa, initial_const,
             clip_min, clip_max, path_data,
             binary_search_steps, max_iterations, learning_rate):
    """
    Build CW-L2 on a Keras Input, ensure explicit softmax/logits, generate adv.
    """
    print(f"Crafting {ATTACK_NAME}: kappa={kappa:.3f}, c(init)={initial_const:g}, "
          f"iters={max_iterations}, bs_steps={binary_search_steps}, lr={learning_rate:g}")

    tf.compat.v1.keras.backend.set_session(sess)
    tf.keras.backend.set_learning_phase(0)  # inference

    input_shape = X.shape[1:]
    num_classes = Y.shape[1]

    # Labels placeholder
    y_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name="cw_y")

    # Model on a plain Keras Input
    x_in = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="cw_input")
    y_out = keras_model(x_in)
    model_on_in = tf.keras.Model(inputs=x_in, outputs=y_out, name="model_on_in")

    # Ensure explicit softmax/logits
    probs_model = prepare_for_cleverhans(model_on_in)
    _ = probs_model(x_in)  # create inbound node
    x_ph = probs_model.inputs[0]

    # Capture variables before building the attack
    before = {v.name for v in tf.global_variables()}

    # Attack graph
    ch_model = KerasModelWrapper(probs_model)
    attack = CarliniWagnerL2(ch_model, sess=sess)
    cw_params = {
        "y": y_ph,
        "confidence": float(kappa),
        "learning_rate": float(learning_rate),
        "binary_search_steps": int(binary_search_steps),
        "max_iterations": int(max_iterations),
        "batch_size": int(batch_size),
        "initial_const": float(initial_const),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
    }
    adv_op = attack.generate(x_ph, **cw_params)

    # Initialize ONLY the new variables created by the attack (do NOT re-init model!)
    after = {v.name for v in tf.global_variables()}
    new_var_names = after - before
    if new_var_names:
        new_vars = [v for v in tf.global_variables() if v.name in new_var_names]
        sess.run(tf.variables_initializer(new_vars))

    # learning_phase feed only if it's a Tensor
    lp = tf.keras.backend.learning_phase()
    lp_is_tensor = hasattr(lp, "op") or getattr(lp, "dtype", None) is not None

    # Generate (pad last batch to fixed size)
    n = X.shape[0]
    X_adv = np.zeros_like(X, dtype=np.float32)

    for i in tqdm(range(0, n, batch_size)):
        j = min(i + batch_size, n)
        m = j - i

        if m == batch_size:
            X_feed = X[i:j]
            Y_feed = Y[i:j]
        else:
            X_feed = np.zeros((batch_size,) + input_shape, dtype=np.float32)
            Y_feed = np.zeros((batch_size, num_classes), dtype=np.float32)
            X_feed[:m] = X[i:j]
            Y_feed[:m] = Y[i:j]

        if lp_is_tensor:
            feed = {x_ph: X_feed, y_ph: Y_feed, lp: 0}
        else:
            feed = {x_ph: X_feed, y_ph: Y_feed}

        adv_batch_full = sess.run(adv_op, feed_dict=feed)
        X_adv[i:j] = adv_batch_full[:m]

    # Evaluate on adversarial set with the ORIGINAL model (unchanged weights)
    _, acc = keras_model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print(f"  → Adversarial accuracy: {100*acc:.3f}%")

    # High-precision L2 stats
    # diff = (X_adv.astype(np.float64) - X.astype(np.float64)).reshape((n, -1))
    # l2_per = np.sqrt((diff * diff).sum(axis=1))
    # mean_l2 = float(l2_per.mean())
    # max_abs = float(np.max(np.abs(diff)))
    # print(f"  → L2 stats: mean={mean_l2:.8f}, max|Δ|={max_abs:.6f}")

    # compute average L2
    avg_l2 = np.linalg.norm(
        X_adv.reshape((len(X), -1)) - X.reshape((len(X), -1)),
        axis=1
    ).mean()
    print(f"  → Avg L2 perturbation: {avg_l2:.4f}\n")

    # Save the adversarial examples for this iters setting
    os.makedirs(path_data, exist_ok=True)
    fname = f"Adv_{ATTACK_NAME}_iters_{max_iterations}.npy"
    np.save(os.path.join(path_data, fname), X_adv)
    print("  → Saved", fname)

    return avg_l2, acc


def run_once(model_file, X, Y, batch_size, kappa, initial_const, clip_min, clip_max, path_data, max_iterations):
    """Load model, evaluate base acc, run CW with given max_iterations, return metrics."""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    # Load model
    try:
        model = load_model(model_file)
    except Exception:
        from keras.models import load_model as k_load_model
        model = k_load_model(model_file)

    # Base accuracy (for reference)
    _, base_acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    print(f"Base test accuracy (reference): {100*base_acc:.3f}%")

    # Craft adversarial set
    avg_l2, adv_acc = craft_cw(
        sess, model, X, Y, batch_size,
        kappa, initial_const, clip_min, clip_max, path_data,
        binary_search_steps=BINARY_SEARCH_STEPS,
        max_iterations=max_iterations,
        learning_rate=LEARNING_RATE
    )
    sess.close()
    return avg_l2, adv_acc, base_acc

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Data
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=0
    )

    print(f"max_iterations sweep (inclusive): {MAX_ITERS_SWEEP}")

    # Sweep max_iterations while keeping others constant
    rows = []
    for iters in MAX_ITERS_SWEEP:
        print("\n================ SWEEP STEP ================")
        print(f"Running C&W with max_iterations = {iters}")
        avg_l2, adv_acc, base_acc = run_once(
            model_file, X_test, Y_test, BATCH_SIZE,
            KAPPA, INITIAL_CONST, CLIP_MIN, CLIP_MAX, PATH_ADV_DATA,
            max_iterations=iters
        )
        rows.append([int(iters), avg_l2, float(adv_acc)])

    # Save CSV summary
    out_dir = f"{root_path}/{ATTACK_NAME}/{DATASET}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/CW_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['MaxIterations', 'Avg. L2', 'Adv Acc'])
        writer.writerows(rows)

    print(f"\nDone! Results in {csv_path}")

if __name__ == "__main__":
    main()

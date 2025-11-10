#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, print_function

import os
import csv
import numpy as np
import tensorflow as tf
# Disable eager exec so tf.compat.v1.placeholder works
tf.compat.v1.disable_eager_execution()

from cleverhans.utils_tf import batch_eval
import keras.backend as K
from keras.models import load_model
from keras.datasets import mnist

def fgsm(x, predictions, eps, clip_min=None, clip_max=None, y=None):
    """Compute the FGSM perturbation."""
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

def fast_gradient_sign_method(sess, model, X, Y, eps, clip_min=None, clip_max=None, batch_size=256):
    """Generate adversarial examples with one FGSM step."""
    x_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    y_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + Y.shape[1:])
    adv_tensor = fgsm(x_ph, model(x_ph), eps=eps, clip_min=clip_min, clip_max=clip_max, y=y_ph)
    X_adv, = batch_eval(
        sess, [x_ph, y_ph], [adv_tensor],
        [X, Y],
        feed={K.learning_phase(): 0},
        args={'batch_size': batch_size}
    )
    return X_adv

def craft_fgsm(sess, model, X, Y, batch_size, eps, clip_min, clip_max, path_data):
    """Craft FGSM adversarial samples, save them, and compute avg L2 + accuracy."""
    attack = 'fgsm'
    print(f"\nCrafting {attack} samples for ε={eps:.2f}…")
    X_adv = fast_gradient_sign_method(sess, model, X, Y, eps, clip_min, clip_max, batch_size)

    # evaluate
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print(f"  → Adversarial accuracy: {100*acc:.2f}%")

    # save the .npy
    fname = f"Adv_{attack}_eps_{eps:.2f}.npy"
    np.save(os.path.join(path_data, fname), X_adv)
    print("  → Saved", fname)

    # compute average L2
    avg_l2 = np.linalg.norm(
        X_adv.reshape((len(X), -1)) - X.reshape((len(X), -1)),
        axis=1
    ).mean()
    print(f"  → Avg L2 perturbation: {avg_l2:.4f}\n")

    return avg_l2, acc

def get_data(x_train, y_train, x_test, y_test):
    """Prepare MNIST data for Conv1D FGSM."""
    X_train = x_train.reshape(-1, 784, 1).astype('float32') / 255.0
    X_test  = x_test .reshape(-1, 784, 1).astype('float32') / 255.0
    Y_train = tf.keras.utils.to_categorical(y_train, 10)
    Y_test  = tf.keras.utils.to_categorical(y_test, 10)
    return X_train, Y_train, X_test, Y_test

def gen_adv_instance(model_file, X, Y, batch_size,
                     eps, clip_min, clip_max, path_data):
    """Load model, evaluate base acc, craft FGSM, and return metrics."""
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    model = load_model(model_file)
    _, base_acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    print(f"Base test accuracy: {100*base_acc:.2f}%")

    avg_l2, adv_acc = craft_fgsm(
        sess, model, X, Y, batch_size,
        eps, clip_min, clip_max, path_data
    )
    sess.close()
    return avg_l2, adv_acc

if __name__ == "__main__":
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train, Y_train, X_test, Y_test = get_data(x_train, y_train, x_test, y_test)

    # FGSM parameters
    CLIP_MIN   = 0.0
    CLIP_MAX   = 1.0
    batch_size = 128

    model_file = '/home/seonghun/Research/Research-son/DETool_MNIST/cnn_model_80_20_split.h5'
    PATH_DATA  = '/home/seonghun/Research/Research-son/FGSM'

    # Sweep ε from 0.20 down to 0.01 in steps of 0.01 → 20 values
    eps_values = np.round(np.arange(0.20, 0.009, -0.01), 2)
    results    = []

    for eps in eps_values:
        l2, acc = gen_adv_instance(
            model_file,
            X_test, Y_test,
            batch_size,
            eps,
            CLIP_MIN, CLIP_MAX,
            PATH_DATA
        )
        results.append([eps, l2, acc])

    # Save summary CSV
    with open('fgsm_results_0.20-0.01.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epsilon', 'Average L2', 'Accuracy'])
        writer.writerows(results)

    print("Done! FGSM results in fgsm_results_0.20-0.01.csv")


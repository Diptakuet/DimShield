from __future__ import absolute_import
from __future__ import print_function

import copy
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Disable eager exec so tf.compat.v1.placeholder works
tf.compat.v1.disable_eager_execution()

from tqdm import tqdm
# from six.moves import xrange

from cleverhans.utils import other_classes
from cleverhans.utils_tf import batch_eval, model_argmax
from cleverhans.attacks_tf import (jacobian_graph, jacobian,
                                   apply_perturbations, saliency_map)
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

print("-------------------")
print("####  Built with CUDA:", tf.test.is_built_with_cuda())
print("####  GPUs found    :", tf.config.list_physical_devices('GPU'))
print("-------------------")


# Root path for the project
root_path = "/local/kat/LESLIE/Topic_Dimshield/"

# Dataset name
'''OLIVETTI
   - 400 grayscale images of faces, each 64x64 pixels
   - 10 classes (0-9) representing different individuals
   - Each image is already normalized to [0,1]
'''
DATASET = "OLIVETTI"

# Attack Type
ATTACK_NAME = "JSMA"

# Path to the dataset for the attack
PATH_ADV_DATA  = f"{root_path}/{ATTACK_NAME}/{DATASET}/Datasets"

# Model type
'''CNN
   - Convolutional Neural Network for image classification
   - Uses 1D convolutional layers since images are flattened to 1D
'''
MODEL_TYPE = "CNN"
MODEL_PATH = f"/local/kat/LESLIE/Topic_Dimshield/{MODEL_TYPE}/{DATASET}"

# Pretrained model name
model_file = f"{MODEL_PATH}/{MODEL_TYPE}_model_{DATASET}.h5"

# Batch Size
BATCH_SIZE = 32
        
# Load Dataset
def fetch_data():
    print(f"Loading {DATASET} dataset...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    x_all, y_all = data.images, data.target

    # Dataset feature size
    DATASET_FEATURE = x_all.shape[1]*x_all.shape[2]
    # Number of classes
    num_classes = y_all.max() + 1
    # Input shape for reshaping
    input_shape = x_all.shape[1:]  # (height, width) tuple


    # There are 400 examples in the dataset and 4096 dimensions
    print("Original dataset:")
    print(f"x_all: {x_all.shape} images")
    print(f"y_all: {y_all.shape} images")
    print(f"Dataset Features: {DATASET_FEATURE}")
    print(f"Number of Classes: {num_classes}")
    print(f"Input Shape: {input_shape}")
    print("-------------------")
    return x_all, y_all, DATASET_FEATURE, num_classes, input_shape


# Split Dataset
def split_data(x_all, y_all):
    ''' First split: Org_img --> 80% (train+val) / 20% (test) 
        Second split: 80% (train+val) --> 75%(train) / 25%(val)
        Resulting in 60% of Org_Img for training, 20% for validation, and 20% for testing
    '''
    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2,random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,random_state=42, shuffle=True)

    print("Splitted Dataset (60%/20%/20%):")
    print(f"  x_train : {x_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  x_val  : {x_val.shape}")
    print(f"  y_val  : {y_val.shape}")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print("-------------------")
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test, DATASET_FEATURE, num_classes, normalize=0):
    ''' 
    Reshape data to 1D for CNN input.
    If normalize=1, divide by 255 to scale to [0,1].
    '''
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

def plot_img(X, Y, plot_path=f"{root_path}/{DATASET}/AE_Reconstructed_{DATASET}.png"):
    '''
    Plot images in a grid format.
    Displays the first 16 images with their corresponding labels.
    '''
    W_grid = 4
    L_grid = 4

    _, axes = plt.subplots(L_grid, W_grid, figsize=(10, 10))
    axes = axes.ravel()  # flatten the grid into a 1D array

    # Loop through the first 16 images and plot them
    for i in range(0, W_grid * L_grid):
        axes[i].imshow(X[i], cmap='gray')  # preserve original style/colors
        label_index = int(Y[i])
        axes[i].set_title(f"{label_index}", fontsize=8)
        axes[i].axis('off')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def jsma(sess, x, predictions, grads, sample, target, theta, gamma,
         increase, nb_classes, clip_min, clip_max, verbose=False):
    """
    TensorFlow implementation of the jacobian-based saliency map method (JSMA).
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param sample: numpy array with sample input
    :param target: target class for sample input
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param nb_classes: integer indicating the number of classes in the model
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param verbose: boolean; whether to print status updates or not
    :return: an adversarial sample
    """

    # Copy the source sample and define the maximum number of features
    # (i.e. the maximum number of iterations) that we may perturb
    adv_x = copy.copy(sample)
    # count the number of features. For MNIST, 1x28x28 = 784; for
    # CIFAR, 3x32x32 = 3072; etc.
    nb_features = np.product(adv_x.shape[1:])
    # reshape sample for sake of standardization
    original_shape = adv_x.shape
    adv_x = np.reshape(adv_x, (1, nb_features))
    # compute maximum number of iterations
    max_iters = np.floor(nb_features * gamma / 2)
    if verbose:
        print('Maximum number of iterations: {0}'.format(max_iters))
    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = set([i for i in range(nb_features)
                             if adv_x[0, i] < clip_max])
    else:
        search_domain = set([i for i in range(nb_features)
                             if adv_x[0, i] > clip_min])

    # Initialize the loop variables
    iteration = 0
    adv_x_original_shape = np.reshape(adv_x, original_shape)
    current = model_argmax(sess, x, predictions, adv_x_original_shape, feed={K.learning_phase(): 0})

    # Repeat this main loop until we have achieved misclassification
    while (current != target and iteration < max_iters and
           len(search_domain) > 1):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_original_shape,
                                              nb_features, nb_classes,
                                              feed={K.learning_phase(): 0})

        # Compute the saliency map for each of our target classes
        # and return the two best candidate features for perturbation
        i, j, search_domain = saliency_map(
            grads_target, grads_others, search_domain, increase)

        # Apply the perturbation to the two input features selected previously
        adv_x = apply_perturbations(
            i, j, adv_x, increase, theta, clip_min, clip_max)

        # Update our current prediction by querying the model
        current = model_argmax(sess, x, predictions, adv_x_original_shape, feed={K.learning_phase(): 0})

        # Update loop variables
        iteration += 1

        # This process may take a while, so outputting progress regularly
        if iteration % 5 == 0 and verbose:
            msg = 'Current iteration: {0} - Current Prediction: {1}'
            print(msg.format(iteration, current))

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        if verbose:
            print('Successful')
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        if verbose:
            print('Unsuccesful')
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


def saliency_map_method(sess, model, X, Y, theta, gamma, clip_min,
                        clip_max):
    """"
    TODO
    :param sess:
    :param model: predictions or after-softmax
    :param X:
    :param Y:
    :param theta:
    :param gamma:
    :param clip_min:
    :param clip_max:
    :return:
    """
    nb_classes = Y.shape[1]
    # Define TF placeholder for the input
    # tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    # Define model gradients
    grads = jacobian_graph(model(x), x, nb_classes)
    X_adv = np.zeros_like(X)
    for i in tqdm(range(len(X))):
        # print("i =", i, "X.shape =", X.shape)
        current_class = int(np.argmax(Y[i]))
        target_class = np.random.choice(other_classes(nb_classes, current_class))
        X_adv[i], _, _ = jsma(
            sess, x, model(x), grads, X[i:(i+1)], target_class, theta=theta,
            gamma=gamma, increase=True, nb_classes=nb_classes,
            clip_min=clip_min, clip_max=clip_max
        )

    return X_adv

def craft_jsma(sess, model, X, Y, batch_size,gamma, clip_min, clip_max, path_data):
    """Craft JSMA adversarial samples, save them, and compute avg L2 + accuracy."""

    print(f"Crafting {ATTACK_NAME} adversarial samples for gamma={gamma:.3f}...")
    X_adv  = saliency_map_method(sess, model, X, Y, theta=1, gamma=gamma, clip_min=clip_min, clip_max=clip_max)
    # X_adv = np.asarray([results[its[i], i] for i in range(len(Y))]) #v1
    #X_adv = results[-1] #v2       

    # evaluate
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print(f"  → Adversarial accuracy: {100*acc:.3f}%")

    # save the .npy
    fname = f"Adv_{ATTACK_NAME}_gamma_{gamma:.3f}.npy"
    np.save(os.path.join(path_data, fname), X_adv)
    print("  → Saved", fname)

    # compute average L2
    avg_l2 = np.linalg.norm(
        X_adv.reshape((len(X), -1)) - X.reshape((len(X), -1)),
        axis=1
    ).mean()
    print(f"  → Avg L2 perturbation: {avg_l2:.4f}\n")

    return avg_l2, acc

def gen_adv_instance(model_file, X, Y, batch_size,
                     eps, clip_min, clip_max, path_data):
    """Load model, evaluate base acc, craft FGSM, and return metrics."""
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    model = load_model(model_file)
    _, base_acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    print(f"Base test accuracy: {100*base_acc:.3f}%")

    avg_l2, adv_acc = craft_jsma(
        sess, model, X, Y, batch_size,
        eps, clip_min, clip_max, path_data
    )
    sess.close()
    return avg_l2, adv_acc


def main():
    # Fetch and split data
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data() 
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)

    # Preprocess data (1D reshape and normalization)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=0
    )

    # FGSM parameters
    CLIP_MIN   = 0.0
    CLIP_MAX   = 1.0

    # Sweep gamma
    gamma_start = 0.050
    gamma_end = -0.005
    gamma_step = 0.005
    gamma_values = np.round(np.arange(gamma_start, gamma_end, -gamma_step), 3)  # 0.050, 0.045, ..., 0.000

    results = []

    for gamma in gamma_values:
        l2, acc = gen_adv_instance(
            model_file, X_test, Y_test, BATCH_SIZE,
            gamma, CLIP_MIN, CLIP_MAX, PATH_ADV_DATA
        )
        results.append([gamma, l2, acc])

    # Save summary CSV
    with open(f"{root_path}/{ATTACK_NAME}/{DATASET}/{ATTACK_NAME}_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Gamma', 'Average L2', 'Accuracy'])
        writer.writerows(results)

    print(f"Done! Results in {ATTACK_NAME}_results.csv")


if __name__ == "__main__":
     main()    # Run the main function
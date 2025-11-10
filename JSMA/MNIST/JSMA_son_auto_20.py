# %%
from __future__ import absolute_import
from __future__ import print_function

import copy
import pandas as pd
from collections import defaultdict
import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf
from tqdm import tqdm
from six.moves import xrange

from cleverhans.utils import other_classes
from cleverhans.utils_tf import batch_eval, model_argmax
from cleverhans.attacks_tf import (jacobian_graph, jacobian,
                                   apply_perturbations, saliency_map)
import keras.backend as K
# from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from keras import utils
import os
from sklearn.model_selection import train_test_split
import pickle
from keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import csv


# Clean up old sessions and 
# %%
tf.compat.v1.disable_eager_execution()

K.clear_session()
tf.reset_default_graph()

import gc
gc.collect()

curr_sess = tf.Session()
tf.keras.backend.set_session(curr_sess)

# %%
pre_trained_model = load_model('/local/kat/LESLIE/Research/JSMA/cnn_model_80_20_split.h5')
print(pre_trained_model.input_shape)

# %%
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
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] < clip_max])
    else:
        search_domain = set([i for i in xrange(nb_features)
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

# %%
def saliency_map_method(sess, model, X, Y, theta, gamma, clip_min=None,
                        clip_max=None):
    """
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
    x = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
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

# %%
def craft_jsma(sess, model, X, Y, batch_size,gamma):
    attack='jsma'
    print('Crafting jsma adversarial samples...')
    X_adv  = saliency_map_method(sess, model, X, Y, theta=1, gamma=gamma, clip_min=CLIP_MIN, clip_max=CLIP_MAX)
    # X_adv = np.asarray([results[its[i], i] for i in range(len(Y))]) #v1
    #X_adv = results[-1] #v2       
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save(os.path.join(PATH_DATA, 'Adv_%s_gamma_%f.npy' % (attack,gamma)), X_adv)
    l2_diff = np.linalg.norm(X_adv.reshape((len(X), -1))-X.reshape((len(X), -1)),axis=1).mean()
    print("Average L-2 perturbation size of the %s attack: %f" %(attack, l2_diff))

    # visualize 
    num_show = min(5, X.shape[0])
    fig, axes = plt.subplots(2, num_show, figsize=(num_show*2.5, 5))
    for i in range(num_show):
        # original
        axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        # adversarial
        axes[1, i].imshow(X_adv[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Adversarial")

    plt.tight_layout()
    fig.savefig(f'{PATH_DATA}jsma_gamma_{gamma:.2f}.png', dpi=150, bbox_inches='tight')

    return l2_diff,acc

# %%
def gen_adv_instance(sess, pre_trained_model, attack,eps,gamma):
    print('Attack: %s' % (attack))
    _, _, _, _, X_test, Y_test = get_data(x_train, y_train, x_val, y_val, x_test, y_test)
    # _, _, _, _, X_test_jsma, Y_test_jsma = get_data(x_train, y_train, x_val, y_val, x_test_jsma_2, y_test_jsma_2)
    _, acc = pre_trained_model.evaluate(X_test, Y_test, batch_size=batch_size,verbose=0)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    attack=='jsma'
    l2_diff,acc=craft_jsma(sess, pre_trained_model, X_test, Y_test,batch_size=batch_size, gamma=gamma)
    print('Adversarial samples crafted and saved to %s ' % PATH_DATA)
    return l2_diff,acc

# %%
(x_all, y_all), (_, _) = mnist.load_data()
# Create data tuples
# train_data = (x_train, y_train)  # Tuple for training data
# test_data = (x_test, y_test)
# x_val = x_test
# y_val = y_test


# %%
def get_data(x_train,y_train,x_val,y_val,x_test,y_test):  
 # Reshape to (n_samples, 784, 1) for Conv1D
    X_train = x_train.reshape(-1, 784, 1).astype('float32') / 255.0
    X_test = x_test.reshape(-1, 784, 1).astype('float32') / 255.0
    X_val = X_test  # Assuming validation data is the same as test data

    # One-hot encode the labels
    Y_train = tf.keras.utils.to_categorical(y_train, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)
    Y_val = Y_test


    X_test = X_test[0:1000,:,:]
    Y_test = Y_test[0:1000, :]

    print("X_train:", X_train.shape)  # Should print (60000, 784, 1)
    print("Y_train:", Y_train.shape)  # Should print (60000, 10)
    print("X_val:", X_val.shape)      # Should print (10000, 784, 1)
    print("Y_val:", Y_val.shape)      # Should print (10000, 10)
    print("X_test:", X_test.shape)    # Should print (10000, 784, 1)
    print("Y_test:", Y_test.shape)    # Should print (10000, 10)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
    

# %%
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,test_size=0.2, shuffle = True, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.25, shuffle = True, random_state = 42)

# X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(x_train,y_train,x_val,y_val,x_test,y_test)
# X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(x_train, y_train, x_val, y_val, x_test, y_test)

# %%
# FGSM/BIM parameters that were chosen
eps_iter=0.1
CLIP_MIN = 0
CLIP_MAX = 1
PATH_DATA ='/local/kat/LESLIE/Research/JSMA/results/'
#num_classes=6
batch_size=32
model_file='/local/kat/LESLIE/Research/JSMA/cnn_model_80_20_split.h5'     # The name of the Pretrained Model 

# %% [markdown]
# gamma | accuracy \n
# 0.5     2% <br>
# 0.35    7% <br>
# 0.3     8% <br>
# 0.25    6% <br>
# 0.22    8% <br>
# **0.215   12%**  <br>
# 0.21    11% -> 18% <br>
# 0.2     13% -> 14% <br>
# **0.15    19.00%** <br>
# 0.12    25% <br>
# **0.01    30%**<br>
# **0.09     40.00%** <br>
# 0.08    43.00% <br>
# 0.07    46.00% <br>
# **0.06    48.00%** <br>
# **0.05    62.00%** <br>
# 0.013   88% <br>
# **0.012   91%**<br>
# 0.01    94% <br>

# %%

# Run gen_adv_instance on 10 different gamma values
result_jsma=[]
result_jsma.append([' gamma ', " l2_diff ", " accuracy "])
eps =0.1
attack='jsma'
# gamma_values = [0.25, 0.15, 0.1]
gamma_values = [0.1]

for gamma in gamma_values:
    l2_diff,acc=gen_adv_instance(sess=curr_sess, pre_trained_model=pre_trained_model, attack=attack, eps=eps, gamma=gamma)
    result_jsma.append([gamma, l2_diff, acc])
    

# %%
# DONE: save result_jsma as csv file
SAVE_PATH_DATA = '/local/kat/LESLIE/Research/JSMA/results/gamma_results.csv'
gamma_df = pd.DataFrame(result_jsma)
gamma_df.to_csv(SAVE_PATH_DATA)


# clean up
curr_sess.close()
K.clear_session()
gc.collect()


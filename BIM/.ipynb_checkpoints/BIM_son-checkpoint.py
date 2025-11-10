from __future__ import absolute_import
from __future__ import print_function

import copy
import pandas as pd
from collections import defaultdict
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from tqdm import tqdm
# from six.moves import xrange
import cleverhans
from cleverhans.utils import other_classes
from cleverhans.utils_tf import batch_eval, model_argmax
from cleverhans.attacks_tf import (jacobian_graph, jacobian,
                                   apply_perturbations, saliency_map)
import keras.backend as K
from keras.models import load_model
from keras.datasets import mnist
from keras import utils
import os
from sklearn.model_selection import train_test_split
import pickle
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import csv

print(cleverhans.__file__)
print(cleverhans.__version__)

def fgsm(x, predictions, eps, clip_min=None, clip_max=None, y=None):
    # Compute loss
    if y is None:
        # In this case, use model predictions as ground truth
        y = tf.cast(tf.equal(predictions, tf.reduce_max(predictions, 1, keepdims=True)), tf.float32)
    y = y / tf.reduce_sum(y, 1, keepdims=True)  # Normalize to ensure y is a valid probability distribution
    logits = predictions  # Assuming predictions are raw logits here
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # Define gradient of loss wrt input
    grad = tf.gradients(loss, x)[0]  # Get the gradient of the loss w.r.t. the input


    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if clip_min is not None and clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

def basic_iterative_method(sess, model, X, Y, eps, eps_iter, nb_iter=10,
                           clip_min=None, clip_max=None, batch_size=256):
    # Define TF placeholders for the input and output
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None,)+X.shape[1:])
    y = tf.compat.v1.placeholder(tf.float32, shape=(None,)+Y.shape[1:])
    # results will hold the adversarial inputs at each iteration of BIM;
    # thus it will have shape (nb_iter, n_samples, n_rows, n_cols, n_channels)
    results = np.zeros((nb_iter, X.shape[0],) + X.shape[1:])
    # Initialize adversarial samples as the original samples, set upper and
    # lower bounds
    X_adv = X
    X_min = X_adv - eps
    X_max = X_adv + eps
    print('Running BIM iterations...')
    # "its" is a dictionary that keeps track of the iteration at which each
    # sample becomes misclassified. The default value will be (nb_iter-1), the
    # very last iteration.
    def f(val):
        return lambda: val
    its = defaultdict(f(nb_iter-1))
    # Out keeps track of which samples have already been misclassified
    out = set()
    for i in tqdm(range(nb_iter)):
        adv_x = fgsm(x, model(x), eps=eps_iter,clip_min=clip_min, clip_max=clip_max, y=y)
        X_adv, = batch_eval(sess, [x, y], [adv_x],[X_adv, Y], feed={K.learning_phase(): 0},args={'batch_size': batch_size}
        )
        X_adv = np.maximum(np.minimum(X_adv, X_max), X_min)
        results[i] = X_adv
        # check misclassifiedsThis
        predictions = np.argmax(model.predict(X_adv, batch_size=512, verbose=0),axis=1)
        misclassifieds = np.where(predictions != Y.argmax(axis=1))[0]
        for elt in misclassifieds:
            if elt not in out:
                its[elt] = i
                out.add(elt)

    return its, results

def craft_bim(sess, model, X, Y, batch_size,eps):
    attack='bim'
    print('Crafting bim adversarial samples...')
    its, results = basic_iterative_method(sess, model, X, Y, eps=eps,eps_iter=eps_iter, clip_min=CLIP_MIN,
            clip_max=CLIP_MAX, batch_size=batch_size)
    X_adv = np.asarray([results[its[i], i] for i in range(len(Y))]) #v1
    #X_adv = results[-1] #v2
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save(os.path.join(PATH_DATA, 'Adv_%s_eps_%f.npy' % (attack,eps)), X_adv)  ## Saving the adversarial data
    l2_diff = np.linalg.norm(X_adv.reshape((len(X), -1))-X.reshape((len(X), -1)),axis=1).mean()
    print("Average L-2 perturbation size of the %s attack: %f" %(attack, l2_diff))
    return l2_diff,acc

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Create data tuples
train_data = (x_train, y_train)  # Tuple for training data
test_data = (x_test, y_test)
x_val = x_test
y_val = y_test

print("Training data tuple:")
print("Images shape:", train_data[0].shape)  # (60000, 28, 28)


def get_data(x_train,y_train,x_val,y_val,x_test,y_test):
 # Reshape to (n_samples, 784, 1) for Conv1D
    X_train = train_data[0].reshape(-1, 784, 1).astype('float32') / 255.0
    X_test = test_data[0].reshape(-1, 784, 1).astype('float32') / 255.0
    X_val = X_test  # Assuming validation data is the same as test data

    # One-hot encode the labels
    Y_train = tf.keras.utils.to_categorical(y_train, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)
    Y_val = Y_test

    print("X_train:", X_train.shape)  # Should print (60000, 784, 1)
    print("Y_train:", Y_train.shape)  # Should print (60000, 10)
    print("X_val:", X_val.shape)      # Should print (10000, 784, 1)
    print("Y_val:", Y_val.shape)      # Should print (10000, 10)
    print("X_test:", X_test.shape)    # Should print (10000, 784, 1)
    print("Y_test:", Y_test.shape)    # Should print (10000, 10)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# FGSM/BIM parameters that were chosen
eps_iter=1
CLIP_MIN = 0
CLIP_MAX = 1
#PATH_DATA ='/local/kat/BIM'
PATH_DATA ='/home/leslie/Research-son/BIM'


#num_classes=6
batch_size=100
# model_file='/local/kat/MNIST_research/SavedModel/CNN_Sigmoid_MNIST'     # The name of the Pretrained Model
#model_file='/local/kat/MNIST_research/SavedModel/cnn_model_80_20_split.h5'
model_file='/home/leslie/Research-son/DETool_MNIST/cnn_model_80_20_split.h5'


def gen_adv_instance(attack,eps):
    print('Attack: %s' % (attack))
    # Create TF session, set it as Keras backend
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    model = load_model(model_file)
    _, _, _, _, X_test, Y_test = get_data(x_train, y_train, x_val, y_val, x_test, y_test)
    _, acc = model.evaluate(X_test, Y_test, batch_size=batch_size,verbose=0)
    #print pre traind modle
    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    if attack=='bim':
        l2_diff,acc=craft_bim(sess, model, X_test, Y_test,batch_size=batch_size, eps=eps)
        print('Adversarial samples crafted and saved to %s ' % PATH_DATA)
        sess.close()
    return l2_diff,acc

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(x_train, y_train, x_val, y_val, x_test, y_test)

tf.compat.v1.disable_eager_execution()


result_bim=[]

for i in range(1,2,1): #50
    eps=i/1000+0.1-0.001
    print(eps)
    l2_diff,acc=gen_adv_instance(attack='bim', eps=eps)
    result_bim.append([eps, l2_diff, acc])

# save results
fields = ['Epsilon', 'Noise', 'Accuracy']
with open('bim8roy.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(result_bim)

attack='bim'
print('Adv_%s_eps_%f.npy' % (attack,eps))

sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

# Model Load
#pre_trained_model = load_model('/local/kat/MNIST_research/SavedModel/cnn_model_80_20_split.h5')
pre_trained_model = load_model('/home/leslie/Research-son/DETool_MNIST/cnn_model_80_20_split.h5')


print(pre_trained_model.input_shape)

# Adv Data load
attack='bim'
EPS = 0.10
adv_filename = f"Adv_{attack}_eps_{EPS:.6f}.npy"
adv_path=os.path.join(PATH_DATA, adv_filename)
adv_data = np.load(adv_path)
print("Loaded adversarial data shape (raw):", adv_data.shape)


_, acc = pre_trained_model.evaluate(adv_data, Y_test, batch_size=batch_size, verbose=0)
print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))

sess.close()



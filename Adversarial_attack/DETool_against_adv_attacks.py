#!/usr/bin/env python
# coding: utf-8

# In[16]:


#----------------------------------Import modules------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from keras import optimizers, regularizers
from keras import utils 

np.random.seed(697)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import keras
from keras.datasets import mnist
from keras.layers import Input, Reshape, Conv1D,Conv2D, Conv1DTranspose,  Flatten, Dense, Lambda, MaxPooling1D, MaxPooling2D, BatchNormalization, Dropout, Activation, UpSampling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

from tensorflow import keras
from tensorflow.keras import layers,losses

from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

import pickle
import sys
import os


# In[17]:


filename = '../dataSet/BigDataset/dataset_IM_1B.pkl'

with open(filename, 'rb') as f:
    data_tuples = pickle.load(f)
f.close()

class_labels = []
for i in range(len(data_tuples)):
    class_labels.append(data_tuples[i][1])
    
num_class = int(len(data_tuples)/1000)

data = []
for i in range(len(data_tuples)):
    data.append(data_tuples[i][0])
    
data_x=np.array(data)
data_y=np.array(class_labels)


# print('Number of images:', len(data_tuples))
# print('Number of classes: ', num_class)
# print('data_x:', data_x.shape)
# print('data_y:', data_y.shape)


# In[18]:


# Random split
# x_train, x_val, y_train, y_val = train_test_split(data_x, data_y,test_size=0.2, shuffle = True, random_state = 8)
# #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size=0.25, shuffle = True, random_state = 8)

# print('x_train:', x_train.shape)
# print('x_val:', x_val.shape)
# #print('x_test:', x_test.shape)
# print('y_train:', y_train.shape)
# print('y_train:', y_val.shape)
# #print('y_test:', y_test.shape)




## Manual split
# Train data
N = int(data_x.shape[0] * 0.6)
num_class = 6
num_measure=1000
interval = int(N / num_class)
k = 0
x_train = np.zeros((N,data_x.shape[1],data_x.shape[2]))
y_train = np.zeros(N)
 
for i in range(0, N, interval):
    x_train[i:i + interval] = data_x[k:k + interval]
    y_train[i:i + interval] = data_y[k:k + interval]
    k += num_measure

#Validation data

N = int(data_x.shape[0] * 0.2)
k = interval
interval = int(N / num_class)
 
x_val = np.zeros((N,data_x.shape[1],data_x.shape[2]))
y_val = np.zeros(N)
 
for i in range(0, N, interval):
    x_val[i:i + interval] = data_x[k:k + interval]
    y_val[i:i + interval] = data_y[k:k + interval]
    k += num_measure

#Test data
N = int(data_x.shape[0] * 0.2)
k = num_measure - interval
interval = int(N / num_class)
x_test = np.zeros((N,data_x.shape[1],data_x.shape[2]))
y_test = np.zeros(N)
 
for i in range(0, N, interval):
    x_test[i:i + interval] = data_x[k:k + interval]
    y_test[i:i + interval] = data_y[k:k + interval]
    k += num_measure
 
    
# print('x_train:', x_train.shape)
# print('x_val:', x_val.shape)
# print('x_test:', x_test.shape)
# print('y_train:', y_train.shape)
# print('y_val:', y_val.shape)
# print('y_test:', y_test.shape)


# In[24]:


CLIP_MIN = 0
CLIP_MAX = 1

X_train = x_train.reshape(-1, 28, 28, 1)
X_test = x_test.reshape(-1, 28, 28, 1)        
    # cast pixels to floats, normalize to [0, 1] range
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = (X_train/255.0) #- (1.0 - CLIP_MAX)
X_test = (X_test/255.0) #- (1.0 - CLIP_MAX)

    # one-hot-encode the labels
Y_train = utils.to_categorical(y_train, num_class)
Y_test = utils.to_categorical(y_test, num_class)

# print("X_train:", X_train.shape)
# print("Y_train:", Y_train.shape)
# print("X_test:", X_test.shape)
# print("Y_test", Y_test.shape)


# In[5]:


class_1 = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
class_1.shape


# In[6]:


#------------------------------------Build the AutoEncoder------------------------------------
latent_dim = 352 
encoding_dim = latent_dim


# Define input layer
input_data = Input(shape=(class_1.shape[1],))
# Define encoding layer
encoded = Dense(encoding_dim, activation='relu')(input_data)

encoder_model = Model(inputs=input_data, outputs=encoded)

# Define decoding layer
#decoded = Dense(class_1.shape[1], activation='sigmoid')(encoded)
decoded = Dense(class_1.shape[1], activation='linear')(encoded)

decoder_model = Model(inputs=input_data, outputs=decoded)


# In[7]:


# Create the autoencoder model
autoencoder = Model(input_data, decoded)

#Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
                  
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=5, mode='auto')


hist_auto = autoencoder.fit(class_1, class_1,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                validation_split=0.2,
                #validation_data=(X_val, X_val),
                callbacks=[early_stopping])


# In[44]:


PATH_DATA = "Toy_data/"
model_file='Toy_data/model.h5'

classification_model = tf.keras.models.load_model(model_file,compile=False)
classification_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


# In[45]:


attack='fgsm'
eps=0.0007
adv_data_fgsm=np.load(os.path.join(PATH_DATA, 'Adv_%s_eps_%0.4f.npy' % (attack,eps)))

# attack='bmi'
# eps=0.0005
# adv_data_bim=np.load(os.path.join(PATH_DATA, 'Adv_%s_eps_%f.npy' % (attack,eps)))


# In[46]:


_, acc  = classification_model.evaluate(X_test, Y_test, batch_size=100, verbose=0)
print("Accuracy on the test set: %0.2f%%" % (100*acc))


_, acc = classification_model.evaluate(adv_data_fgsm, Y_test, batch_size=100,verbose=0)
print("Accuracy on the adversarial test set (FGSM): %0.2f%%" % (100*acc))


_, acc = classification_model.evaluate(adv_data_bim, Y_test, batch_size=100,verbose=0)
print("Accuracy on the adversarial test set (BIM): %0.2f%%" % (100*acc))


# In[47]:


X_test_flat=X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
#print("Flattened original test data: ", X_test_flat.shape)
re_gen_test=autoencoder(X_test_flat)
re_gen_test=np.array(re_gen_test)
#print("Flattened regerated test data: ", re_gen_test.shape)


adv_data_fgsm_flat=adv_data_fgsm.reshape((len(adv_data_fgsm), np.prod(adv_data_fgsm.shape[1:])))
#print("Flattened adv (FGSM) test data: ", adv_data_fgsm_flat.shape)
re_gen_adv_fgsm_test=autoencoder(adv_data_fgsm_flat)
re_gen_adv_fgsm_test=np.array(re_gen_adv_fgsm_test)
#print("Flattened regerated adv (FGSM) test data: ", re_gen_adv_fgsm_test.shape)


adv_data_bim_flat=adv_data_bim.reshape((len(adv_data_bim), np.prod(adv_data_bim.shape[1:])))
#print("Flattened adv (BIM) test data: ", adv_data_bim_flat.shape)
re_gen_adv_bim_test=autoencoder(adv_data_bim_flat)
re_gen_adv_bim_test=np.array(re_gen_adv_bim_test)
#print("Flattened regerated adv (BIM) test data: ", re_gen_adv_bim_test.shape)



# Reshape back: Deflattening
X_test_reconstructed = re_gen_test.reshape(-1, 28, 28, 1)   
adv_fgsm_reconstructed = re_gen_adv_fgsm_test.reshape(-1, 28, 28, 1)  
adv_bim_reconstructed = re_gen_adv_bim_test.reshape(-1, 28, 28, 1)

# print("Deflattened regerated test data: ", X_test_reconstructed.shape)
# print("Deflattened regerated adv (FGSM) test data: ", adv_fgsm_reconstructed.shape)
# print("Deflattened regerated adv (BIM) test data: ", adv_bim_reconstructed.shape)


# In[48]:


_, acc  = classification_model.evaluate(X_test, Y_test, batch_size=100, verbose=0)
print("Accuracy on the original test set: %0.2f%%" % (100*acc))

_, re_acc = classification_model.evaluate(X_test_reconstructed, Y_test, batch_size=128,verbose=0)
print("Accuracy on the reconstructed test set: %0.2f%%" % (100*re_acc))

print("----------------------------------------------------------")

_, acc = classification_model.evaluate(adv_data_fgsm, Y_test, batch_size=100,verbose=0)
print("Accuracy on the adversarial test set (FGSM): %0.2f%%" % (100*acc))

_, re_acc = classification_model.evaluate(adv_fgsm_reconstructed, Y_test, batch_size=128,verbose=0)
print("Accuracy on the reconstructed adv (FGSM) test set: %0.2f%%" % (100*re_acc))

print("----------------------------------------------------------")

_, acc = classification_model.evaluate(adv_data_bim, Y_test, batch_size=100,verbose=0)
print("Accuracy on the adversarial test set (BIM): %0.2f%%" % (100*acc))

_, re_acc = classification_model.evaluate(adv_bim_reconstructed, Y_test, batch_size=128,verbose=0)
print("Accuracy on the reconstructed adv (BIM) test set: %0.2f%%" % (100*re_acc))


# # In[15]:


# n = 6 
# start=2
# end=1200
# interval=200
# plt.figure(figsize=(20,7))
 
# k=1

# for i in range(start,end,interval):
#     # Display original
#     ax = plt.subplot(4, n, k)
#     plt.imshow(X_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(int(y_test[i]))
#     ax.set_title('Original')
    
#     # Display reconstructed from original
#     ax = plt.subplot(4, n, k+n)
#     plt.imshow(X_test_reconstructed[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(int(y_test[i]))
#     ax.set_title('Reconstructed from Original')
    
#     # Adversarial FGSM
#     ax = plt.subplot(4, n, k+2*n)
#     plt.imshow(adv_data_bim[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(adv_data_fgsm_pred_classes[i])
#     ax.set_title('BIM')
    
#     # Display reconstructed from adversarial
#     ax = plt.subplot(4, n, k+3*n)
#     plt.imshow(adv_bim_reconstructed[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(int(y_test[i]))
#     ax.set_title('Reconstructed from Adv')  
#     k += 1
 
 

# plt.show()


# # In[16]:


# encoder_file='Toy_data/encoder_352_new.h5'
# decoder_file='Toy_data/decoder_352_new.h5'

# encoder_model = tf.keras.models.load_model(encoder_file)
# decoder_model = tf.keras.models.load_model(decoder_file)


# # In[17]:


# encoder_model.summary()


# # In[18]:


# decoder_model.summary()


# # In[19]:


# # Reconstruct the images

# X_test=X_test*255.0
# X_test_tensor=tf.convert_to_tensor(X_test)


# encoded_test=encoder_model.predict(X_test_tensor)
# print(encoded_test.shape)
# decoded_test=decoder_model.predict(encoded_test)
# re_gen_test=np.array(decoded_test)
# print("Regenarated test data: ", re_gen_test.shape)


# adv_data_fgsm=adv_data_fgsm*255.0
# adv_data_fgsm_tensor=tf.convert_to_tensor(adv_data_fgsm)

# encoded_adv_fgsm=encoder_model.predict(adv_data_fgsm_tensor)
# decoded_adv_fgsm=decoder_model.predict(encoded_adv_fgsm)
# re_gen_adv_fgsm=np.array(decoded_adv_fgsm)
# print("Regenarated adv (FGSM) data: ", re_gen_adv_fgsm.shape)



# adv_data_bim=adv_data_bim*255.0
# adv_data_bim_tensor=tf.convert_to_tensor(adv_data_bim)

# encoded_adv_bim=encoder_model.predict(adv_data_bim_tensor)
# decoded_adv_bim=decoder_model.predict(encoded_adv_bim)
# re_gen_adv_bim=np.array(decoded_adv_bim)
# print("Regenarated adv (BIM) data: ", re_gen_adv_bim.shape)


# # In[20]:


# n = 6 
# start=0
# end=1200
# interval=200
# plt.figure(figsize=(20,7))
 
# k=1

# for i in range(start,end,interval):
#     # Display original
#     ax = plt.subplot(4, n, k)
#     plt.imshow(X_test_tensor[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(int(y_test[i]))
#     ax.set_title('Original')
    
#     # Display reconstructed from original
#     ax = plt.subplot(4, n, k+n)
#     plt.imshow(re_gen_test[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(int(y_test[i]))
#     ax.set_title('Reconstructed from Original')
    
#     # Adversarial FGSM
#     ax = plt.subplot(4, n, k+2*n)
#     plt.imshow(adv_data_bim_tensor[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(adv_data_fgsm_pred_classes[i])
#     ax.set_title('BIM')
    
#     # Display reconstructed from adversarial
#     ax = plt.subplot(4, n, k+3*n)
#     plt.imshow(re_gen_adv_bim[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #ax.set_title(int(y_test[i]))
#     ax.set_title('Reconstructed from Adv')  
#     k += 1
 
 
# #plt.savefig('FGSM.png',dpi=600)
# plt.show()


# # In[21]:


# _, acc  = classification_model.evaluate(X_test/255.0, Y_test, batch_size=100, verbose=0)
# print("Accuracy on the original test set: %0.2f%%" % (100*acc))

# _, re_acc = classification_model.evaluate(re_gen_test/255.0, Y_test, batch_size=128,verbose=0)
# print("Accuracy on the reconstructed test set: %0.2f%%" % (100*re_acc))

# print("----------------------------------------------------------")

# _, acc = classification_model.evaluate(adv_data_fgsm/255.0, Y_test, batch_size=100,verbose=0)

# print("Accuracy on the adversarial test set (FGSM): %0.2f%%" % (100*acc))

# _, re_acc = classification_model.evaluate(re_gen_adv_fgsm/255.0, Y_test, batch_size=128,verbose=0)

# print("Accuracy on the reconstructed adv (FGSM) test set: %0.2f%%" % (100*re_acc))

# print("----------------------------------------------------------")

# _, acc = classification_model.evaluate(adv_data_bim/255.0, Y_test, batch_size=100,verbose=0)
# print("Accuracy on the adversarial test set (BIM): %0.2f%%" % (100*acc))

# _, re_acc = classification_model.evaluate(re_gen_adv_bim/255.0, Y_test, batch_size=128,verbose=0)
# print("Accuracy on the reconstructed adv (BIM) test set: %0.2f%%" % (100*re_acc))


# # In[22]:


# # import tensorflow as tf
# # import pickle
# # import pandas as pd
# # import numpy as np
 
# # # Load the saved decoder
# # saved_encoder = tf.keras.models.load_model('Toy_data/encoder_352_new.h5')
 
# # # Load the encoded data
# # f = open('../dataSet/BigDataset/dataset_IM_1B.pkl', 'rb')
# # data_tuples = pickle.load(f)
# # f.close()
# # df = pd.DataFrame(data_tuples, columns=['predictor', 'label'])
 
# # input_tensor_features  = tf.convert_to_tensor(df['predictor'].tolist())
# # input_tensor_features_noise  = input_tensor_features+np.random.normal(0, 0.1, input_tensor_features.shape)

# # # Use it to reconstruct data from encoded representations
# # reconstructed_data = saved_encoder.predict(input_tensor_features)


# # In[23]:


# # import matplotlib.pyplot as plt
 
# # saved_decoder = tf.keras.models.load_model('Toy_data/decoder_352_new.h5')
# # imageReconstructed = saved_decoder.predict(reconstructed_data)
 
# # plt.figure(figsize=(50, 10), dpi=80)
# # for i in range(10):
# #     plt.subplot(3,10,i+1)
# #     plt.imshow(X_test[100*i//2 + i%2+5], cmap='gray')
# #     plt.title(f'Original Image {1000*i//2 + i%2}')
# #     plt.axis('off')
# # #     plt.subplot(3,10,i+11)
# # #     plt.imshow(input_tensor_features_noise[1000*i//2 + i%2+5], cmap='gray')
# # #     plt.title('Input image with noise')
# # #     plt.axis('off')
# #     plt.subplot(3,10,i+21)
# #     plt.imshow(imageReconstructed[1000*i//2 + i%2+5], cmap='gray')
# #     plt.title('Reconstructed Image')
# #     plt.axis('off')
# #     plt.suptitle('Random noise from range (0, 0.2) is added to the input image')


# # In[ ]:





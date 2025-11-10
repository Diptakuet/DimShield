import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# Functions to load and preprocess MNIST data
# ------------------------------------------------

def fetch_data():
    print("Loading MNIST dataset...")
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
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, shuffle=True
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42, shuffle=True
    )
    print("Splitted Dataset (80%/20%/20%):")
    print(f"  x_train : {x_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  x_val   : {x_val.shape}")
    print(f"  y_val   : {y_val.shape}")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print("-------------------")
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test,
                    DATASET_FEATURE, num_classes, normalize=1):
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

# ------------------------------------------------
# Main
# ------------------------------------------------

print("#######Built with CUDA:", tf.test.is_built_with_cuda())
print("#######GPUs found    :", tf.config.list_physical_devices('GPU'))

# 1) Load and preprocess MNIST data
x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
    x_train, y_train, x_val, y_val, x_test, y_test,
    DATASET_FEATURE, num_classes, normalize=1
)

# 2) Build CNN Model for 1D input
model = models.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(DATASET_FEATURE, 1)))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# 3) Define EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# 4) Train the CNN
history = model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping]
)

# 5) Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# 6) Save the trained CNN model
model.save("CNN_model_MNIST.h5")
print("Model saved as CNN_model_MNIST.h5")
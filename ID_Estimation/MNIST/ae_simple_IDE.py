import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

print("-------------------")
print("####  Built with CUDA:", tf.test.is_built_with_cuda())
print("####  GPUs found    :", tf.config.list_physical_devices('GPU'))
print("-------------------")

DATASET = "MNIST"
root_path = "/local/kat/LESLIE/Topic_Dimshield/ID_Estimation"

MODEL_TYPE = "CNN"
MODEL_PATH = f"/local/kat/LESLIE/Topic_Dimshield/{MODEL_TYPE}/{DATASET}"
model_file = f"{MODEL_PATH}/CNN_model_{DATASET}.h5"

def fetch_data():
    print(f"Loading {DATASET} dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    DATASET_FEATURE = x_all.shape[1] * x_all.shape[2]
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
    ''' 60% train, 20% val, 20% test '''
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

def plot_img(X, Y, plot_path=f"{root_path}/{DATASET}/AE_Reconstructed_{DATASET}.png"):
    W_grid = 4
    L_grid = 4
    _, axes = plt.subplots(L_grid, W_grid, figsize=(10, 10))
    axes = axes.ravel()
    for i in range(0, W_grid * L_grid):
        axes[i].imshow(X[i], cmap='gray')
        label_index = int(Y[i])
        axes[i].set_title(f"{label_index}", fontsize=8)
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

def build_simple_ae(latent_dim, DATASET_FEATURE):
    inp  = layers.Input(shape=(DATASET_FEATURE,))
    bott = layers.Dense(latent_dim, activation='relu')(inp)
    up   = layers.Dense(DATASET_FEATURE, activation='sigmoid')(bott)
    ae   = models.Model(inp, up, name=f"AE_1D_{latent_dim}")
    ae.compile(optimizer='adam', loss='mse')
    return ae

def plot_results_and_save(results, plot_path=f"{root_path}/{DATASET}/ID_vs_MSE_vs_Accuracy_{DATASET}.png", csv_out=f"{root_path}/{DATASET}/AE_Result_{DATASET}.csv"):
    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)
    print(f"\nSaved results to {csv_out}")
    latent_dims = [d['latent_dim'] for d in results]
    mses = [d['mse'] for d in results]
    accuracies = [d['accuracy'] for d in results]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('MSE', color='tab:blue')
    ax1.plot(latent_dims, mses, color='tab:blue', marker='o', label='MSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:green')
    ax2.plot(latent_dims, accuracies, color='tab:green', marker='s', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    plt.title('Latent Dimension vs MSE and Accuracy')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    return df

def estimate_knee_point(df, id_csv=f"{root_path}/{DATASET}/AE_estimated_id_{DATASET}.csv"):
    mse_norm = (df['mse'].max() - df['mse']) / (df['mse'].max() - df['mse'].min())
    acc_norm = (df['accuracy'] - df['accuracy'].min()) / (df['accuracy'].max() - df['accuracy'].min())
    score    = acc_norm - mse_norm
    knee_idx = score.idxmax()
    estimated_id = int(df.loc[knee_idx, 'latent_dim'])
    print(f"Estimated intrinsic dimension (knee point): {estimated_id}")
    id_df = pd.DataFrame({'estimated_id': [estimated_id]})
    id_df.to_csv(id_csv, index=False)
    print(f"Saved estimated ID to {id_csv}")
    return estimated_id

def main():
    x_all, y_all, DATASET_FEATURE, num_classes, input_shape = fetch_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test,
        DATASET_FEATURE, num_classes, normalize=1
    )
    sweep_start = 0
    sweep_end = 100 #DATASET_FEATURE + 1
    sweep_step = 10
    batch_size = 64
    latent_dims = list(range(sweep_start, sweep_end, sweep_step))
    results = []
    pretrained_model = models.load_model(model_file)
    for ld in latent_dims:
        print(f"\n=== Training AE with latent_dim = {ld} ===")
        best_acc = -1
        best_result = None
        for run in range(3):
            print(f"[+] Run {run+1}/3")
            K.clear_session()
            ae = build_simple_ae(ld, DATASET_FEATURE)
            es = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            ae.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=100,
                batch_size=batch_size,
                callbacks=[es],
                verbose=2
            )
            X_recon_1d = ae.predict(X_test, verbose=0)
            X_recon_1d = X_recon_1d.reshape(-1, DATASET_FEATURE, 1)
            mse = mean_squared_error(X_test.squeeze(), X_recon_1d.squeeze())
            loss, acc = pretrained_model.evaluate(X_recon_1d, Y_test, verbose=0)
            print(f"Reconstructed data:\nlatent_dim={ld} → MSE={mse:.4f}, Loss={loss:.4f}, Accuracy={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_result = {
                    'latent_dim': ld,
                    'mse': mse,
                    'accuracy': acc
                }
        results.append(best_result)
    df = plot_results_and_save(results)
    estimated_id = estimate_knee_point(df)
    print(f"\nTraining AE with estimated latent_dim = {estimated_id} for final reconstruction and saving model...")
    ae_final = build_simple_ae(estimated_id, DATASET_FEATURE)
    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    ae_final.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=64,
        callbacks=[es],
        verbose=0
    )
    ae_final.save(f"{root_path}/{DATASET}/AE_model_{DATASET}_id_{estimated_id}.h5")
    print(f"Final AE model saved as AE_model_{DATASET}_id_{estimated_id}.h5")
    X_recon_final_1d = ae_final.predict(X_test, verbose=0)
    X_recon_final_img = X_recon_final_1d.reshape(-1, *input_shape)
    plot_img(X_recon_final_img, y_test)
    loss, acc = pretrained_model.evaluate(X_recon_final_1d.reshape(-1, DATASET_FEATURE, 1), Y_test, verbose=0)
    print("-------------------")
    print(f"Reconstructed data:\nlatent_dim={estimated_id} → Loss={loss:.4f}, Accuracy={acc:.4f}")
    print("-------------------")
    K.clear_session()

if __name__ == "__main__":
    main()
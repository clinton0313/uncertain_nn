# %%
#%%
import os, pathlib, pickle, sys

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from HH_NN import NNDropout, NNfuncs
import h5py

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Global Variables
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(42)
tf.random.set_seed(42)

CHECKPOINT_PATH = "checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
# %%
def sine_data(n=2048, a:int=0, b:int=10, noise:float=0.3):
    x = np.linspace(a,b,n)
    y = 3*np.sin(x)
    y = y+np.random.normal(0,noise*np.abs(y),n)
    return y,x

y_train, X_train = sine_data(n=4096, a=-10, b=10)
y_test, X_test = sine_data(n=512, a=-11, b=11)
# %%

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
callback = EarlyStopping(monitor='mean_squared_error', patience=20)
loss_fn = NNDropout.nll
reduce_lr_mse = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.9, patience=5, verbose=1, min_delta=1e-4, mode='max')

model_configs = {
    "reg_d1_2": (1,0.2,"mse"),
    "reg_d1_5": (1,0.5,"mse"),
    "reg_d2_2": (2,0.2,loss_fn),
    "reg_d2_5": (2,0.5,loss_fn)
}

checkpoints = {
    name: ModelCheckpoint(
        os.path.join(CHECKPOINT_PATH, name), 
        save_weights_only=True, 
        save_best_only=True, 
        monitor="val_mean_squared_error"
    ) for name in model_configs.keys()
}

out_models = []
# %%
for label, (output_dim, p_dropout, loss) in model_configs.items():
    model = NNDropout(output=output_dim, p_dropout=p_dropout)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])

    try:
        model.load_weights(os.path.join(CHECKPOINT_PATH, label))
        print("Weights loaded successfully for {label}")
    except Exception as e:
        print(e)
        print("No weights to load, starting to fit instead...")

    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        verbose=1, 
        validation_split=0.2, 
        callbacks=[callback, reduce_lr_mse] #checkpoints disabled, conflict with dropbox
    )

    with open(os.path.join(CHECKPOINT_PATH, f"{label}_history.pkl"), "wb") as outfile:
        pickle.dump(history.history, outfile)
    out_models.append(model)


# %% 
# Fit regressions, plot results
model_utils = NNfuncs(model=out_models[3])
model_utils.fit_regression(X_test, mc=True, T=100)
model_utils.plot_regression(X_train, y_train)

# %% WIP
# CO2 2015 example
import pandas as pd
with h5py.File('co2/co2_data.h5','r') as h5f:
    # return a h5py dataset object:
    data = h5f[("data")][:]
    labels = h5f[("label")][:]

data_train = np.concatenate((data, labels), axis=1)

model = NNDropout(output=2, p_dropout=0.1)
model.compile(optimizer=optimizer, loss="mse", metrics=["mean_squared_error"])

history = model.fit(
    data_train[:,0], 
    data_train[:,1], 
    epochs=100, 
    verbose=1, 
    validation_split=0.2
)

model_utils = NNfuncs(model=model)
model_utils.fit_regression(np.linspace(data_train[1,0],data_train[1,0]+10,100), mc=True, T=100)
model_utils.plot_regression(data_train[:,0], data_train[:,1])
# %%

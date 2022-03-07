# %%
#%%
import os, pathlib, pickle, sys
from typing import List, Dict, Tuple
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from RegNN import NNDropout, NNDropConnect, NNRegressor
import h5py
from tqdm import tqdm

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

def evaluator(model_configs:Dict=None, traindata:Tuple=None, testdata:Tuple=None, epochs:int=100, include_plot:bool=False, **kwargs):
    checkpoints = {
        name: ModelCheckpoint(
            os.path.join(CHECKPOINT_PATH, name), 
            save_weights_only=True, 
            save_best_only=True, 
            monitor="val_mean_squared_error"
        ) for name in model_configs.keys()
    }
    out_models = []
    for label, (model, loss) in tqdm(model_configs.items()):
        model.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])
        try:
            model.load_weights(os.path.join(CHECKPOINT_PATH, label))
            print("Weights loaded successfully for {label}")
        except Exception as e:
            print(e)
            print("No weights to load, starting to fit instead...")

        history = model.fit(
            traindata[0], 
            traindata[1], 
            epochs=epochs, 
            verbose=0, 
            validation_split=0.2, 
            callbacks=[checkpoints[label]] + [v for (k,v) in kwargs.items() if k in ["callback", "reduce_lr_mse"]] #callback,checkpoints disabled, conflict with dropbox
        )
        with open(os.path.join(CHECKPOINT_PATH, f"{label}_history.pkl"), "wb") as outfile:
            pickle.dump(history.history, outfile)
        out_models.append(model)

        model_utils = NNRegressor(model=model)
        model_utils.predict(testdata[0], mc=True, T=100)
        figure = model_utils.plot(Xtrain=traindata[0], Ytrain=traindata[1], combined=True, title=label)
        if include_plot:
            figure.savefig(f"figs/{label}.png", bbox_inches='tight', dpi=600)
    
    return out_models

# %%
# Sine data, evaluate dropout/dropconnect, output1/output2, dropout 0.2/0.5
y_train, X_train = sine_data(n=4096, a=-10, b=10)
y_test, X_test = sine_data(n=512, a=-11, b=11)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
callback = EarlyStopping(monitor='val_loss', patience=2000)
reduce_lr_mse = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=200, verbose=1, min_delta=1e-4, mode='max')

model_configs = {
    "nndropout_o1_p2": (NNDropout(output=1, p_dropout=0.2),"mse"),
    "nndropout_o2_p2": (NNDropout(output=2, p_dropout=0.2),NNDropout.nll),
    "nndropout_o1_p5": (NNDropout(output=1, p_dropout=0.5),"mse"),
    "nndropout_o2_p5": (NNDropout(output=2, p_dropout=0.5),NNDropout.nll),
    "nndropconn_o1_p2": (NNDropConnect(output=1, p_dropout=0.2),"mse"),
    "nndropconn_o2_p2": (NNDropConnect(output=2, p_dropout=0.2),NNDropout.nll),
    "nndropconn_o1_p5": (NNDropConnect(output=1, p_dropout=0.5),"mse"),
    "nndropconn_o2_p5": (NNDropConnect(output=2, p_dropout=0.5),NNDropout.nll),
}

models = evaluator(
    model_configs=model_configs,
    traindata=(X_train, y_train),
    testdata=(X_test, y_test),
    epochs=10000,
    include_plot=True,
    optimizer=optimizer,
    callback=callback,
    reduce_lr_mse=reduce_lr_mse
)


# %% WIP
# CO2 2015 example

with h5py.File('co2/co2_data.h5','r') as h5f:
    # return a h5py dataset object:
    data = h5f[("data")][:]
    labels = h5f[("label")][:]
data_train = np.concatenate((data, labels), axis=1)

model_configs = {
    "co2_nndropout_o1_p2": (NNDropout(output=1, p_dropout=0.2),"mse"),
    "co2_nndropout_o2_p2": (NNDropout(output=2, p_dropout=0.2),NNDropout.nll),
    "co2_nndropconn_o1_p2": (NNDropConnect(output=1, p_dropout=0.2),"mse"),
    "co2_nndropconn_o2_p2": (NNDropConnect(output=2, p_dropout=0.2),NNDropout.nll),
}
reduce_lr_mse = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2000, verbose=1, min_delta=1e-4, mode='max')

models = evaluator(
    model_configs=model_configs,
    traindata=(data_train[:,0], data_train[:,1]),
    testdata=(np.arange(-1.72, 3.51, 0.01).reshape(-1, 1), 1),
    epochs=10000,
    include_plot=True,
    optimizer=optimizer,
    reduce_lr_mse=reduce_lr_mse,
    callback=callback
)


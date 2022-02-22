#%%
from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np
import os, PIL, pathlib, itertools, random
import pandas as pd
from abc import abstractmethod

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Sequential
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Global Variables
DATA_DIR = pathlib.Path("horse_or_human")
LABELS = {0: "horse", 1: "human"}

#%%
class MCDropout(Dropout):
    def call(self, x):
        return super().call(x, training=True)
class CNNmodel(Model):
    def __init__(self, num_classes:int, p_dropout:Sequence = 0.2, mc_dropout:bool = False):
        super(CNNmodel, self).__init__()
        self.n_linear = 3
        self.mc_dropout = mc_dropout
        self.p_dropout = p_dropout
        self.num_classes = num_classes

        self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = Conv2D(64, 3, padding='same', activation='relu')
        self.lin1 = Dense(128, activation="relu")
        self.lin2 = Dense(64, activation="relu")
        self.lin3 = Dense(self.num_classes)

    @property
    def mc_dropout(self) -> bool:
        return self._mc_dropout
    
    @mc_dropout.setter
    def mc_dropout(self, mc_dropout:bool):
        assert isinstance(mc_dropout, bool), f"mc_dropout must be a boolean, instead got {mc_dropout}"
        self._mc_dropout = mc_dropout

    @property
    def p_dropout(self) -> Sequence:
        return self._p_dropout
    
    @p_dropout.setter
    def p_dropout(self, p):
        assert isinstance(p, (float, Sequence, int)), "p needs to ne be either a probability or a sequence of probabilities"
        if isinstance(p, Sequence):
            assert len(p) == self.n_linear, "This model has three linear layers and needs three dropout layers"
            assert all(np.array(p) <= 1) and all(np.array(p) >= 0), "Non valid probabilities given"
        else:
            assert 0 <= p <= 1, "p needs to be a valid probability"
            p = [p for _ in range(self.n_linear)]
        self._p_dropout = p
        self._set_dropout_layers()
    
    def _set_dropout_layers(self):
        if not self._mc_dropout:
            self._dropout_layers = [Dropout(self.p_dropout[i]) for i in range(self.n_linear)]
        elif self._mc_dropout:
            self._dropout_layers = [MCDropout(self.p_dropout[i]) for i in range(self.n_linear)]

    def featurize(self, x):
        x = Rescaling(1./255)(x)
        x = self.conv1(x)
        x = MaxPool2D()(x)
        x = self.conv2(x)
        x = MaxPool2D()(x)
        x = self.conv3(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        return x
    
    def call(self, x):
        x = self.featurize(x)
        x = self._dropout_layers[0](x)
        x = self.lin1(x)
        x = self._dropout_layers[1](x)
        x = self.lin2(x)
        x = self._dropout_layers[2](x)
        x = self.lin3(x)
        return x

    def mc_predict(self, x, T):
        #Save and set dropout layers
        original_mc_dropout = self.mc_dropout
        self.mc_dropout = True
        self._set_dropout_layers
        predictions = tf.stack([self.call(x) for _ in range(T)], axis=0)

        #Return dropout layers to original setting
        self.mc_dropout = original_mc_dropout
        self._set_dropout_layers

        predictions = tf.nn.softmax(predictions, axis=2)
        labels = tf.cast(tf.argmax(predictions, axis=2), dtype=tf.float64)
        mean = tf.math.reduce_mean(labels, axis=0)
        var = tf.math.reduce_variance(labels, axis=0) #SHOULD I BE TAKING THE VARIANCE BEFORE OR AFTER ARGMAX???
        return mean, var

class CNNPlotter():
    def __init__(self, model:CNNmodel=None, labels:dict={}):
        self.model = model
        self.labels = labels
        self.fig = None

    def _model_check(self):
        assert self._model, "This method requires a model to be set."

    def _label_check(self):
        assert self._model.num_classes == len(self._labels.keys()) or self._labels == {}, \
            f"Incorrect number of labels. Got {len(self._labels.keys())} for {self._model.num_classes} classes."
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        assert isinstance(model, (CNNmodel, type(None))), "CNNPlotter only accepts a CNNmodel model."
        self._model = model
    
    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self, labels):
        assert isinstance(labels, dict), f"Labels should be a dictionary of labels, instead got {labels}"
        self._labels = labels
        

    def classify_image(self, img, mc = False, T=10):
        self._model_check()
        self._label_check()
        img = tf.reshape(img, [-1, 300, 300, 3])
        if mc:
            class_pred, pred_proba = self.model.mc_predict(img, T=T)
            class_pred = 1 if class_pred >= 0.5 else 0
            class_pred = self._labels[class_pred] if self._labels else class_pred
        else:
            prediction = self.model.predict(img)
            score = tf.nn.softmax(prediction)
            class_pred = np.argmax(score) if self._labels == {} else self._labels[np.argmax(score)]
            pred_proba = np.max(score)
        return class_pred, pred_proba

    def plot_prediction(self, img, ax=None, mc = False, T = 10):
        class_pred, pred_proba = self.classify_image(img, mc=mc, T=T)
        if not ax:
            self.fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(tf.squeeze(img))
        if mc:
            ax.set_title(f"Class: {class_pred}\nVariance: {round(float(pred_proba), 1)}")
        elif not mc:
            ax.set_title(f"Class: {class_pred}\nSoftmax: {round(float(pred_proba), 1)}")

    def plot_batch(self, images, nrow, ncol, figsize=(16,16), predict=False, mc=False, T=10):
        assert len(images) == nrow * ncol, "Not the same number of images as grid"
        self.fig, ax = plt.subplots(nrow, ncol, figsize=figsize, tight_layout=True)
        self.fig.suptitle(lambda _: "MC Dropout" if mc else "Softmax")
        self.fig.set_facecolor("white")
        grid_pos = list(itertools.product(range(nrow), range(ncol)))
        for img, (i, j) in zip(images, grid_pos):
            if predict:
                self.plot_prediction(img, ax=ax[i][j], mc=mc, T=T)
            else: 
                ax[i][j].imshow(img)
                ax[i][j].axis("off")
#%%
#Plot some horses and some humans
humans = list(DATA_DIR.glob('*/humans/*'))
horses = list(DATA_DIR.glob('*/horses/*'))
horse_sample = [PIL.Image.open(str(horse)) for horse in random.sample(horses, 8)]
human_sample = [PIL.Image.open(str(human)) for human in random.sample(humans, 8)]

cnn_plotter = CNNPlotter()
cnn_plotter.plot_batch(horse_sample, 2, 4, figsize=(8, 16))
cnn_plotter.fig

#%%
cnn_plotter.plot_batch(human_sample, 2, 4, figsize=(8, 16))
cnn_plotter.fig
#%%
#Load the data
batch_size = 16
img_height = 300
img_width = 300

img_folder = tfds.folder_dataset.ImageFolder(
    root_dir=str(DATA_DIR),
    shape=(img_height, img_width, 3)
)

ds = img_folder.as_dataset(
    batch_size=batch_size,
    as_supervised=True,
)

train_ds = ds["train"].cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = ds["validate"].cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# %%
#Specify model
model = CNNmodel(num_classes=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="crossentropy")

#%%
#Compile and fit
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
history = model.fit(train_ds, validation_data = val_ds, epochs = 1, verbose = 1)

#%%
#Plot our losses

results = pd.DataFrame(history.history)
res_fig, res_ax = plt.subplots(figsize=(14,14))
res_ax = results.plot(ax=res_ax)
res_ax.grid(True)

#%%
#Plot some predictions!
images, labels = next(iter(val_ds))

cnn_plotter.model = model

cnn_plotter.plot_batch(images, 4, 4, predict=True)
softmax_fig = cnn_plotter.fig
#%%
cnn_plotter.plot_batch(images, 4, 4, predict=True, mc=True)
mc_dropout_fig = cnn_plotter.fig
# %%

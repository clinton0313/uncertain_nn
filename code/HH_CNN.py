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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from CustomDropout import MCDropConnect, MCDropout, DropConnect


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
        self._set_dropout_layers()
        predictions = tf.stack([self.call(x) for _ in range(T)], axis=0)

        #Return dropout layers to original setting
        self.mc_dropout = original_mc_dropout
        self._set_dropout_layers

        #Dimension T x B x C  ~~~ NEED TO TEST
        soft = tf.nn.softmax(predictions, axis=2)
        means = tf.math.reduce_mean(soft, axis=0)
        H_hat = -means * tf.math.log(means)
        H = -1 * tf.math.reduce_mean(soft * tf.math.log(soft), axis = 0)
        mutual_info = H_hat - H
        return means, mutual_info

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
        
    def classify_image(self, img, mc = False, T=100):
        self._model_check()
        self._label_check()
        img = tf.reshape(img, [-1, 300, 300, 3])
        if mc:
            class_pred, uncertainty = self.model.mc_predict(img, T=T)
            scores = tf.squeeze(class_pred).numpy()
            score = np.max(class_pred)
            label = self._labels[np.argmax(scores)] if self._labels != {} else np.argmax(scores)
            uncertainty = tf.squeeze(uncertainty).numpy().sum()
        else:
            prediction = self.model.predict(img)
            scores = tf.nn.softmax(prediction)
            score = np.max(scores)
            label = self._labels[np.argmax(scores)] if self._labels != {} else np.argmax(scores)
            uncertainty = 0
        return label, score, uncertainty

    def plot_prediction(self, img, ax=None, mc = False, T = 100):
        if not ax:
            self.fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(tf.squeeze(img))

        label, score, uncertainty = self.classify_image(img, mc=mc, T=T)
        ax.set_title(f"Predicted Class: {label}\nSoftmax Score: {score:.2f}\nUncetainty: {uncertainty:.2f}")

    def plot_batch(self, images, nrow, ncol, figsize=(16,16), predict=False, mc=False, T=10):
        assert len(images) == nrow * ncol, "Not the same number of images as grid"
        self.fig, ax = plt.subplots(nrow, ncol, figsize=figsize, tight_layout=True)
        title = ""
        if predict:
            title = "MC Dropout" if mc else "Softmax"
        self.fig.suptitle(title)
        self.fig.set_facecolor("white")
        grid_pos = list(itertools.product(range(nrow), range(ncol)))
        for img, (i, j) in zip(images, grid_pos):
            if predict:
                self.plot_prediction(img, ax=ax[i][j], mc=mc, T=T)
            else: 
                ax[i][j].imshow(img)
                ax[i][j].axis("off")
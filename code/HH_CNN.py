from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np
import itertools
from abc import abstractmethod

import seaborn as sns
from scipy import stats
import tensorflow as tf
from tensorflow.keras import Model

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from CustomDropout import MCDropout, DropConnect

class CNNDropout(Model):
    def __init__(self, num_classes:int, p_dropout:Sequence = 0.2):
        super(CNNDropout, self).__init__()
        self.n_linear = 3
        self.p_dropout = p_dropout
        self.num_classes = num_classes

        self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = Conv2D(64, 3, padding='same', activation='relu')
        self.lin1 = Dense(128, activation="relu")
        self.lin2 = Dense(64, activation="relu")
        self.lin3 = Dense(self.num_classes)

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
    
    def call(self, x, training=False):
        x = self.featurize(x)
        x = Dropout(self.p_dropout[0])(x, training=training)
        x = self.lin1(x)
        x = Dropout(self.p_dropout[1])(x, training=training)
        x = self.lin2(x)
        x = Dropout(self.p_dropout[2])(x, training=training)
        x = self.lin3(x)
        return x

    def mc_predict(self, x, T):
        predictions = tf.stack([self.call(x, training=True) for _ in range(T)], axis=0)

        #Dimension T x B x C  ~~~ NEED TO TEST
        soft = tf.nn.softmax(predictions, axis=2)
        means = tf.math.reduce_mean(soft, axis=0)
        H_hat = -means * tf.math.log(means)
        H = -1 * tf.math.reduce_mean(soft * tf.math.log(soft), axis = 0)
        mutual_info = H_hat - H
        return means, mutual_info

class CNNDropConnect(CNNDropout):
    def __init__(self, *args, **kwargs):
        super(CNNDropConnect, self).__init__(*args, **kwargs)
        self.lin1 = DropConnect(units = 128, p_dropout=self.p_dropout[0])
        self.lin2 = DropConnect(units = 64, p_dropout=self.p_dropout[1])
        self.lin3 = DropConnect(units = self.num_classes, p_dropout=self.p_dropout[2])
    
    def call(self, x, training=False):
        x = self.featurize(x)
        x = self.lin1(x, training=training)
        x = self.lin2(x, training=training)
        x = self.lin3(x, training=training)
        return x

class CNNPlotter():
    def __init__(self, model:CNNDropout=None, labels:dict={}, img_height=300, img_width=300):
        self.model = model
        self.labels = labels
        self.fig = None
        self.ax = None
        self.img_shape = (1, img_height, img_width, 3)
        self.correct_scores = []
        self.wrong_scores = []

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
        assert isinstance(model, (CNNDropout, type(None))), "CNNPlotter only accepts a CNNmodel model."
        self._model = model
    
    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self, labels):
        assert isinstance(labels, dict), f"Labels should be a dictionary of labels, instead got {labels}"
        self._labels = labels
    
    def _resize_image(self, img):
        if img.shape != self.img_shape:
            _, h, w, _ = self.img_shape
            img = tf.image.resize(img, (h, w))
            img = tf.reshape(img, self.img_shape)
        return img
    
    def _rescale_image(self, img):
        return Rescaling(1/np.max(img))(img)


    def classify_image(self, img, mc = False, T=100):
        self._model_check()
        self._label_check()
        img = self._resize_image(img)
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
        label, score, uncertainty = self.classify_image(img, mc=mc, T=T)
        ax.set_title(f"Predicted Class: {label}\nSoftmax Score: {score:.2f}\nUncetainty: {uncertainty:.2f}")
        img = self._resize_image(img)
        img = self._rescale_image(img)
        if not ax:
            self.fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(tf.squeeze(img))


    def plot_batch(self, images, nrow, ncol, figsize=(8,16), predict=False, mc=False, T=10):
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
    
    def reset_measures(self):
        self.correct_scores = []
        self.wrong_scores = []

    def gather_measures(self, imgs, labels, batch=True, out_sample=False, mc=False, T=100):
        if not batch:
            imgs = self._resize_image(img)
            labels = tf.reshape(labels, (1, len(labels)))
        for img, label in zip(imgs, labels):
            pred_label, score, uncertainty = self.classify_image(img, mc, T)
            measure = uncertainty if mc else 1 - score
            if not out_sample:
                if pred_label == self._labels[label.numpy()]:
                    self.correct_scores.append(measure)
                else:
                    self.wrong_scores.append(measure)
            else:
                self.wrong_scores.append(measure)
    
    def plot_measures(self, title="", out_sample=False, wrong_label="Wrong", wrong_color="tab:red", new_fig=True):
        if new_fig:
            self.fig, self.ax = plt.subplots(figsize=(14, 14), tight_layout=True)
        
        hist_kws = {"alpha": 0.5}

        if not out_sample:
            sns.distplot(self.correct_scores, bins=30, hist_kws=hist_kws, color="tab:cyan", label="Correct", ax=self.ax)
        sns.distplot(self.wrong_scores, bins=30, hist_kws=hist_kws, color=wrong_color, label=wrong_label, ax=self.ax)
        self.ax.legend()
        self.ax.set_xlim(0, 0.5)
        for pos in ["top", "left", "right"]:
            self.ax.spines[pos].set_visible(False)

        self.ax.set(ylabel=None, yticks=[])
        self.ax.set_title(title, fontsize=18)
        self.ax.set_xlabel("Uncertainty Score", fontsize=15)

    def save_fig(self, path):
        if self.fig != None:
            self.fig.savefig(path, facecolor="white", transparent=False)
            print("Fig saved!")
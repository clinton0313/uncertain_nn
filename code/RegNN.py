#%%
from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from CustomDropout import MCDropout, DropConnect

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%%
class MCDropout(Dropout):
    def call(self, x):
        return super().call(x, training=True)


class NNDropout(Model):
    def __init__(self, n_neurons:Sequence=[1024,1024,1024,1024], p_dropout:Sequence=0.2, mc_dropout:bool=False, output:int=1):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.mc_dropout = mc_dropout
        self.p_dropout = p_dropout
        self.output = output

        self.norm = Normalization()
        self.dense0 = Dense(self.n_neurons[0], activation = "relu")
        self.dense1 = Dense(self.n_neurons[1], activation = "relu")
        self.dense2 = Dense(self.n_neurons[2], activation = "relu")
        self.dense3 = Dense(self.n_neurons[3], activation = "relu")
        self.end = Dense(output, activation=None)
        
    @property
    def output(self) -> int:
        return self._output

    @output.setter
    def output(self, output):
        assert(1<=output<=2), "Output should be an integer taking value 1 or 2"
        self._output = output

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
        p = [p for _ in range(self.n_layers)]
        self._p_dropout = p
        
    def call(self, x, training):
        # x = tf.keras.layers.InputLayer(input_shape=(self.output,1))(x)
        x = self.norm(x)
        x = Dropout(self.p_dropout[0])(x, training=training)
        x = self.dense0(x)
        x = Dropout(self.p_dropout[1])(x, training=training)
        x = self.dense1(x)
        x = Dropout(self.p_dropout[2])(x, training=training)
        x = self.dense2(x)
        x = Dropout(self.p_dropout[3])(x, training=training)
        x = self.dense3(x)
        x = Dropout(self.p_dropout[3])(x, training=training)
        x = self.end(x)
        return x

    def mc_predict(self, x, T):
        predictions = tf.stack([self.call(x, training=True) for _ in range(T)], axis=0)

        if self.output==1: # 2015 paper
            y_m = tf.math.reduce_mean(predictions, axis=0)
            y_v = tf.subtract(tf.divide(tf.reduce_sum(tf.math.multiply(predictions, predictions), axis=0),T),tf.math.square(y_m))
            return y_m, y_v

        elif self.output!=1: # 2017 paper, two-headed combined variance
            y_m = tf.math.reduce_mean(predictions[:,:,0][:,np.newaxis], axis=0)
            y_v = tf.math.reduce_variance(predictions[:,:,0][:,np.newaxis], axis=0)
            si2 = tf.math.exp(tf.reduce_mean(predictions[:,:,1][:,np.newaxis], axis=0))
            y_cv = np.mean(si2 + predictions[:,:,0]**2, axis=0) - y_m**2
            return y_m, y_v, si2, y_cv

    def build_graph(self, dim):
        x = tf.keras.Input(shape=dim)
        return tf.keras.Model(inputs=x, outputs=self.call(x, training=True))

    @staticmethod
    def nll(y_train, y_pred):
        y_train = tf.reshape(y_train, [-1])
        mu = y_pred[:,0]
        sigma = y_pred[:,1]
        loss = tf.reduce_mean(tf.math.divide(tf.math.add(sigma, tf.square(y_train-mu)/tf.math.exp(sigma)),2.))
        return loss


class NNDropConnect(NNDropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense0 = DropConnect(units=self.n_neurons[0], p_dropout=self.p_dropout[0], activation = "relu")
        self.dense1 = DropConnect(units=self.n_neurons[1], p_dropout=self.p_dropout[1], activation = "relu")
        self.dense2 = DropConnect(units=self.n_neurons[2], p_dropout=self.p_dropout[2], activation = "relu")
        self.dense3 = DropConnect(units=self.n_neurons[3], p_dropout=self.p_dropout[3], activation = "relu")

    def call(self, x, training=False):
        x = self.norm(x)
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.end(x)
        return x


class NNRegressor():
    def __init__(self, model:NNDropout=None):
        self.model = model
        self.output = model.output

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        assert isinstance(model, (NNDropout, type(None))), "Method requires regression output model."
        self._model = model

    def _model_check(self):
        assert self._model, "Method requires a model to be defined."
    
    def predict(self, Xdata:Sequence, mc:bool=False, T:int=100):
        self.mc = mc
        self.X = Xdata
        self._model_check()
        self.reginstance = True
        zeroes = np.array([0]*len(Xdata))[:,np.newaxis]

        if Xdata.ndim==1: Xdata = Xdata[:,np.newaxis]
        if self.output==1:
            if mc:
                self.y_m, self.y_v = self.model.mc_predict(Xdata, T)
            else:
                self.y_m, self.y_v = self.model.predict(Xdata), zeroes
            return self.y_m, self.y_v
        if self.output==2:
            if mc:
                self.y_m, self.y_v, self.si2, self.y_cv = self.model.mc_predict(Xdata, T)
            else:
                self.y_m, self.y_v, self.si2, self.y_cv = self.model.predict(Xdata), zeroes, zeroes, zeroes
            return self.y_m, self.y_v, self.si2, self.y_cv
        

    def plot(self, Xtrain:Sequence=None, Ytrain:Sequence=None, combined:bool=False, title:str="Baseline", **kwargs):
        assert(self.reginstance == True), "Plot requires a fitted model."
        fig, ax = plt.subplots(1,1,figsize=(12,8))

        if self.output==1:
            if Xtrain is not None:
                ax.scatter(Xtrain, Ytrain, c="purple", label='Training data', alpha=0.2)
            color_pred = (0.0, 101.0 / 255.0, 189.0 / 255.0)
            ax.scatter(self.X, self.y_m, color=color_pred, label='Out-of-Sample')
            if self.mc:
                ax.fill_between(
                    self.X.reshape(len(self.X),), 
                    tf.squeeze(self.y_m - tf.math.sqrt(self.y_v)), 
                    tf.squeeze(self.y_m + tf.math.sqrt(self.y_v)),
                    alpha=0.25,
                    color=color_pred
                )
                ax.fill_between(
                    self.X.reshape(len(self.X),),
                    tf.squeeze(self.y_m - 2.0 * tf.math.sqrt(self.y_v)), 
                    tf.squeeze(self.y_m + 2.0 * tf.math.sqrt(self.y_v)),
                    alpha=0.35,
                    color=color_pred
                )
            ax.legend(loc="lower center")
            fig.tight_layout()

        elif self.output==2:
            if combined:
                var = self.y_cv
                labels = ["Combined uncertainty- 1SD", "Combined uncertainty- 2SD"]
                
            elif not combined:
                var = self.y_v
                labels = ["Epistemic uncertainty- 1SD", "Epistemic uncertainty- 2SD"]

            ax.scatter(Xtrain, Ytrain, c="purple", alpha=0.2)
            if not combined:
                ax2 = ax.twinx()
                ax2.plot(self.X, np.sqrt(tf.squeeze(self.si2)), color="green", label="Aleatoric uncertainty")
                ax2.set_ylabel('Aleatoric Uncertainty')
                ax2.legend(loc="upper left")
            color_pred = (0.0, 101.0 / 255.0, 189.0 / 255.0)
            ax.plot(self.X, np.reshape(self.y_m, (len(self.X),)), color=color_pred)
            ax.set_ylabel("Y")
            ax.fill_between(
                self.X.reshape(len(self.X),), 
                tf.squeeze(self.y_m - np.sqrt(var)), 
                tf.squeeze(self.y_m + np.sqrt(var)),
                alpha=0.25,
                color=color_pred,
                label=labels[0]
            )
            ax.fill_between(
                self.X.reshape(len(self.X),),
                tf.squeeze(self.y_m - 2. * np.sqrt(var)), 
                tf.squeeze(self.y_m + 2. * np.sqrt(var)),
                alpha=0.35,
                color=color_pred,
                label=labels[1]
            )
            ax.legend(loc="upper right")
        if "ylims" in kwargs:
            ylims = kwargs.get("ylims")
            ax.set_ylim(bottom=ylims[0], top=ylims[1])
        fig.tight_layout()
    
        ax.set_title(title)
        return fig
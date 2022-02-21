#%%
from typing import Sequence
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model
from matplotlib import pyplot as plt
import numpy as np
import os, PIL
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
import pathlib, itertools, random
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Global Variables
DATA_DIR = pathlib.Path("horse_or_human")
LABELS = {0: "human", 1: "horse"}
#%%

def plot_images(imgs, nrows, ncols, figsize = (16, 16), tight_layout=True):
    '''Plot a list of image objects in a grid'''
    assert len(imgs) == nrows * ncols, "Not the same number of images as grid"
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    grid_pos = list(itertools.product(range(nrows), range(ncols)))
    for img, (i, j) in zip(imgs, grid_pos):
        ax[i][j].imshow(img)
        ax[i][j].axis("off")
    return fig

def plot_prediction(img, model, ax=None, label_dict= {}):
    img = tf.reshape(img, [-1, 300, 300, 3])
    prediction = model.predict(img)
    score = tf.nn.softmax(prediction)
    class_pred = np.argmax(score) if label_dict == {} else label_dict[np.argmax(score)]
    pred_proba = np.max(score)
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.axis("off")
    ax.imshow(tf.squeeze(img))
    ax.set_title(f"Class: {class_pred} with softmax proba: {round(pred_proba, 1)}")
    return fig, ax

def plot_batch(images, nrow, ncol, model, figsize=(16,16), label_dict={}):
    assert len(images) == nrow * ncol, "Not the same number of images as grid"
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
    grid_pos = list(itertools.product(range(nrow), range(ncol)))
    for img, (i, j) in zip(images, grid_pos):
        plot_prediction(img, model, ax=ax[i][j], label_dict=label_dict)
    return fig
#%%
class HModel(Model):
    def __init__(self, num_classes:int, p:Sequence = [0.2, 0.2, 0.2], mc_dropout:bool = False):
        super(HModel, self).__init__()
        self._mc_dropout = False
        self.p = p
        self.conv1 = layers.Conv2D(16, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.lin1 = layers.Dense(128, activation="relu")
        self.lin2 = layers.Dense(64, activation="relu")
        self.lin3 = layers.Dense(num_classes)

    @property
    def mc_dropout(self) -> bool:
        return self._mc_dropout
    
    @mc_dropout.setter
    def set_mc_dropout(self, mc_dropout:bool):
        self._mc_dropout = mc_dropout

    def featurize(self, x):
        x = layers.experimental.preprocessing.Rescaling(1./255)(x)
        x = self.conv1(x)
        x = layers.MaxPool2D()(x)
        x = self.conv2(x)
        x = layers.MaxPool2D()(x)
        x = self.conv3(x)
        x = layers.MaxPool2D()(x)
        x = layers.Flatten()(x)
        return x
    
    def call(self, x, training=True):
        use_dropout = training or self._mc_dropout
        x = self.featurize(x)
        x = layers.Dropout(self.p[0])(x, training=use_dropout)
        x = self.lin1(x)
        x = layers.Dropout(self.p[1])(x, training=use_dropout)
        x = self.lin2(x)
        x = layers.Dropout(self.p[2])(x, training=use_dropout)
        x = self.lin3(x)
        return x

#%%
#Plot some horses and some humans
humans = list(DATA_DIR.glob('*/humans/*'))
horses = list(DATA_DIR.glob('*/horses/*'))
horse_sample = [PIL.Image.open(str(horse)) for horse in random.sample(horses, 8)]
human_sample = [PIL.Image.open(str(human)) for human in random.sample(humans, 8)]
fig = plot_images(horse_sample, 2, 4)
fig2 = plot_images(human_sample, 2, 4)

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
model = HModel(num_classes=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="crossentropy")


#%%
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
history = model.fit(train_ds, validation_data = val_ds, epochs = 3, verbose = 1)

#%%

#%%

#DONT KNOW WHY THE BELOW DOESNT WORK


# train_loss = tf.keras.metrics.Mean(name="train_loss")
# train_acc = tf.keras.metrics.SparseCategoricalCrossentropy(name="train_acc")
# test_loss = tf.keras.metrics.Mean(name="test_loss")
# test_acc = tf.keras.metrics.SparseCategoricalCrossentropy(name="test_acc")

# @tf.function
# def train_step(images, labels):#, loss_fn, model, train_loss, train_acc):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss = loss_fn(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss(loss)
#     train_acc(labels, predictions)

# @tf.function
# def test_step(images, labels):#, loss_fn, model, test_loss, test_acc, T = 1000, mc_dropout = False):
#     predictions = model(images, training=False)
#     val_loss = loss_fn(labels, predictions)

#     test_loss(val_loss)
#     test_acc(labels, predictions)

# # %%

# EPOCHS=5

# for epoch in range(EPOCHS):
#     train_loss.reset_states()
#     train_acc.reset_states()
#     test_loss.reset_states()
#     test_acc.reset_states()

#     for images, labels in train_ds:
#         train_step(images, labels)#, loss_fn, model, train_loss, train_acc)
    
#     for images, labels in val_ds:
#         test_step(images, labels)#, loss_fn, model, test_loss, test_acc)
    
#     print(
#         f'''
#         Epoch {epoch + 1}
#         Loss: {train_loss.result()}
#         Accuracy: {round(train_acc.result() * 100, 2)}
#         Test Loss: {test_loss.result()}
#         Test Accuracy: {round(test_acc.result() * 100, 2)}
#         '''
#     )

# %%

#%%
from matplotlib import pyplot as plt
import numpy as np
import os, PIL, pathlib, random
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from HH_CNN import CNNDropConnect, CNNDropout, CNNPlotter
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Global Variables
tf.random.set_seed(1234)
DATA_DIR = pathlib.Path("horse_or_human")
LABELS = {0: "horse", 1: "human"}
CHECKPOINT_PATH = "checkpoints"


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

#%%

#Using Data Augmentation

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=(1./255),
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=(1./255)
)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(str(DATA_DIR), "train"),
    target_size=(300,300),
    class_mode="binary",
    batch_size=16,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(str(DATA_DIR), "validate"),
    target_size=(300,300),
    class_mode="binary",
    batch_size=16
)

# %%
#Specify model
model = CNNDropout(num_classes=2, p_dropout=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="crossentropy")
checkpoint = ModelCheckpoint(os.path.join(CHECKPOINT_PATH, "dropconnect_weights_p5"), save_weights_only=True, save_best_only=True)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

#%%
#Compile and fit
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

history = model.fit(train_ds, 
    validation_data = val_ds, 
    epochs = 1, 
    callbacks = [checkpoint, early_stopping],
    verbose = 1)

#Plot our losses

results = pd.DataFrame(history.history)
res_fig, res_ax = plt.subplots(figsize=(14,14))
res_ax = results.plot(ax=res_ax)
res_ax.grid(True)

#%%
#Load Saved Model
model = CNNDropout(num_classes=2, p_dropout=0.5)
model.load_weights(os.path.join(CHECKPOINT_PATH, "saved_weights"))


#%%
#Plot some predictions!
images, labels = next(iter(val_ds))

cnn_plotter.model = model
cnn_plotter.labels = LABELS

cnn_plotter.plot_batch(images, 4, 4, predict=True)
softmax_fig = cnn_plotter.fig
#%%
cnn_plotter.plot_batch(images, 4, 4, predict=True, mc=True, T=1000)
mc_dropout_fig = cnn_plotter.fig
# %%

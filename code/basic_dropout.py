#%%
from matplotlib import pyplot as plt
import numpy as np
import os, PIL, pathlib, random, pickle
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
models = [
    CNNDropout(num_classes=2, p_dropout=0.5), 
    CNNDropout(num_classes =2, p_dropoput=0.7),
    CNNDropConnect(num_classes=2, p_dropout=0.5),
    CNNDropConnect(num_classes=2, p_dropout=0.7)
    ]

modelnames = ["dropout_p5", "dropout_p7", "dropconnect_p5", "dropconnect_p7"]

checkpoints = [ModelCheckpoint(os.path.join(CHECKPOINT_PATH, name), save_weights_only=True, save_best_only=True) for name in modelnames]

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="crossentropy")
checkpoint = ModelCheckpoint(os.path.join(CHECKPOINT_PATH, "data_augmentation_weights_p5"), save_weights_only=True, save_best_only=True)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

#%%
#Compile and fit

for model, checkpoint, name in zip(models, checkpoints, modelnames):

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    history = model.fit(train_ds, 
        validation_data = val_ds, 
        epochs = 100, 
        callbacks = [checkpoint, early_stopping],
        verbose = 1)

    with open(os.path.join(CHECKPOINT_PATH, f"{name}_history.pkl"), "wb") as outfile:
        pickle.dump(history, outfile)

#%%
import os, pathlib, pickle

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from HH_CNN import CNNDropConnect, CNNDropout
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Global Variables
tf.random.set_seed(1234)
DATA_DIR = pathlib.Path("horse_or_human")
LABELS = {0: "horse", 1: "human"}
CHECKPOINT_PATH = "checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

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
    rescale=(1./255),
    zoom_range=0.2,
    rotation_range=10,
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
    # CNNDropout(num_classes=2, p_dropout=0.5), 
    # CNNDropout(num_classes =2, p_dropout=0.7),
    CNNDropConnect(num_classes=2, p_dropout=0.3),
    CNNDropConnect(num_classes=2, p_dropout=0.5)
    ]

modelnames = [
    # "dropout_p5", 
    # "dropout_p7", 
    "dropconnect_p3",
    "dropconnect_p5"
    ]
checkpoints = [ModelCheckpoint(os.path.join(CHECKPOINT_PATH, name), save_weights_only=True, save_best_only=True, monitor="val_accuracy") for name in modelnames]

#%%

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="crossentropy")
early_stopping = EarlyStopping(patience=20, restore_best_weights=True, monitor="val_accuracy")
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=4, min_lr = 1e-8, verbose=1)

#%%
#Compile and fit

for model, checkpoint, name in zip(models, checkpoints, modelnames):


    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    try:
        model.load_weights(os.path.join(CHECKPOINT_PATH, name))
        print("Weights loaded successfully for {name}")
    except Exception as e:
        print(e)
        print("No weights to load, starting to fit instead...")


    history = model.fit(train_ds, 
        validation_data = val_ds, 
        epochs = 100, 
        callbacks = [checkpoint, early_stopping, lr_scheduler],
        verbose = 1)

    with open(os.path.join(CHECKPOINT_PATH, f"{name}_history.pkl"), "wb") as outfile:
        pickle.dump(history.history, outfile)

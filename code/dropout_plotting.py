#%%

import matplotlib.pyplot as plt
import os
import pathlib
import pickle
import PIL
import random

import tensorflow as tf
import tensorflow_datasets as tfds

from helper_functions import evaluate_measures, plot_history, plot_measures, load_measures
from HH_CNN import CNNDropConnect, CNNDropout, CNNPlotter
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Global Variables
DATA_DIR = pathlib.Path("horse_or_human")
LABELS = {0: "horse", 1: "human"}
CHECKPOINT_PATH = "checkpoints"
FIG_PATH = "figs"
MEASURES_PATH = "measures"
modelnames = ["dropout_p5", "dropout_p7", "dropconnect_p3", "dropconnect_p5"]
os.makedirs(FIG_PATH, exist_ok=True)

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

#Load Out of Sample Data

catsdogs = tfds.load("cats_vs_dogs", batch_size=16, as_supervised=True)
flowers = tfds.load("oxford_flowers102", batch_size=16, as_supervised=True)

#%%
#Load Saved Model

#Check our model accuracies
for model in modelnames:
    with open(os.path.join(CHECKPOINT_PATH, f"{model}_history.pkl"), "rb") as infile:
        history = pickle.load(infile)
    print(f"{model} had a max accuracy of {max(history['val_accuracy'])}")

mc_dropout = CNNDropout(num_classes=2, p_dropout=0.5)
mc_dropconnect = CNNDropConnect(num_classes=2, p_dropout=0.3)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="crossentropy")
mc_dropout.compile(optimizer=optimizer, loss=loss_fn)
mc_dropconnect.compile(optimizer=optimizer, loss=loss_fn)

mc_dropout.load_weights(os.path.join(CHECKPOINT_PATH, modelnames[0]))
mc_dropconnect.load_weights(os.path.join(CHECKPOINT_PATH, modelnames[2]))

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
#Plot some predictions!
# images, labels = next(iter(val_ds))

# cnn_plotter.model = model
# cnn_plotter.labels = LABELS

# cnn_plotter.plot_batch(images, 4, 4, predict=True)
# softmax_fig = cnn_plotter.fig

# #%%
# cnn_plotter.plot_batch(images, 4, 4, predict=True, mc=True, T=100)
# mc_dropout_fig = cnn_plotter.fig
# # %%

# images, labels = next(iter(catsdogs["train"]))

# cnn_plotter.plot_batch(images, 4, 4, predict=True, mc=True, T=100)
# catsdogs_fig = cnn_plotter.fig

# # %%

# images, labels = next(iter(flowers["train"]))

# cnn_plotter.plot_batch(images, 4, 4, predict=True, mc=True, T=100)
# flowers_fig = cnn_plotter.fig


#%%
#Plot our training losses

with open(os.path.join(CHECKPOINT_PATH, f"{modelnames[0]}_history.pkl"), "rb") as infile:
    dropout_hist = pickle.load(infile)

with open(os.path.join(CHECKPOINT_PATH, f"{modelnames[2]}_history.pkl"), "rb") as infile:
    dropconnect_hist = pickle.load(infile)

#Plot our losses

dropout_acc_fig = plot_history(dropout_hist, ["accuracy", "val_accuracy"], "MC Dropout with p = 0.5")
dropconnect_acc_fig = plot_history(dropconnect_hist, ["accuracy", "val_accuracy"], "MC Dropconnect with p = 0.3")

dropout_acc_fig.savefig(os.path.join(FIG_PATH, "dropout_acc.png"), facecolor="white", transparent=False)
dropconnect_acc_fig.savefig(os.path.join(FIG_PATH, "dropconnect_acc.png"), facecolor="white", transparent=False)
#%%

#Load Prediction Measures

models = ["softmax", "dropout", "dropconnect"]
datasets = ["horse", "cat", "flower"]
types = ["correct", "wrong"]
colors = ["tab:blue", "tab:orange", "tab:purple", "tab:olive"]
data_titles = {"horse": "Horse and Human", "cat": "Cats vs Dogs", "flower": "Flowers"}

measures = load_measures(MEASURES_PATH, models, datasets)

#%%
#Plot Prediction Measures

datset_figs = []
for data in datasets:
    scores = [f"{m}_{data}_wrong" for m in models]
    labels = [m.capitalize() for m in models]
    datset_figs.append(plot_measures(measures, scores, colors, labels, title=data_titles[data])) 

comp_fig, axs = plt.subplots(3, 1, figsize=(15, 15), tight_layout=True, sharey=False, sharex=True)
for ax, model in zip(axs.ravel(), models):
    plot_measures(
        measures, 
        scores=[f"{model}_horse_{t}" for t in types], 
        colors=colors, 
        labels=[t.capitalize() for t in types], 
        ax=ax, 
        title=model.capitalize()
    )
comp_fig.text(x=0.35, y = 3.35, s="Horse and Human Classification", fontsize=20, transform=ax.transAxes)


#%%

#Measure Evaluation

# model_titles = ["Softmax", "MC Dropout", "MC DropConnect"]
# use_mc = [False, True, True]
# plotters = [
#     CNNPlotter(mc_dropout, labels=LABELS),
#     CNNPlotter(mc_dropout, labels=LABELS),
#     CNNPlotter(mc_dropconnect, labels=LABELS)
# ]
# datasets = [
#     val_ds, 
#     catsdogs["train"],
#     flowers["train"]
# ]
# dataset_names = [
#     "Horse and Human",
#     "Cats and Dogs",
#     "Flowers"
# ]
# out_sample = [False, True, True]
# wrong_color = ["tab:red", "tab:brown", "tab:olive"]

# evaluate_measures(
#     SAVEPATH, 
#     model_titles=["MC DropConnect"], 
#     plotters=[CNNPlotter(mc_dropconnect, labels=LABELS)], 
#     use_mc=[True], 
#     datasets=datasets, 
#     dataset_names=dataset_names, 
#     out_sample=out_sample, 
#     wrong_color=wrong_color,
#     max_batches=16,
#     T=100,
# )

#%%

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import PIL
import random
import regex as re
from tqdm import tqdm

import tensorflow as tf
import tensorflow_datasets as tfds
from HH_CNN import CNNDropConnect, CNNDropout, CNNPlotter
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#Global Variables
tf.random.set_seed(1234)
DATA_DIR = pathlib.Path("horse_or_human")
LABELS = {0: "horse", 1: "human"}
CHECKPOINT_PATH = "checkpoints"
SAVEPATH = "figs"
modelnames = ["dropout_p5", "dropout_p7", "dropconnect_p3", "dropconnect_p5"]
os.makedirs(SAVEPATH, exist_ok=True)

#Helper Function

def plot_history(history, cols, title=""):
    results = pd.DataFrame(history)
    results = results.loc[:, cols]
    fig, ax = plt.subplots(figsize=(14,14))
    ax = results.plot(ax=ax)
    ax.set_title(title)
    ax.grid(True)
    return fig

def evaluate_measures(SAVEPATH, model_titles, plotters, use_mc, datasets, dataset_names, 
                      out_sample, wrong_color, max_batches=5, T = 10):
    for (title, plotter, mc) in zip(model_titles, plotters, use_mc):
        for dataset_name, dataset, out, color in zip(dataset_names, datasets, out_sample, wrong_color):
            loader = iter(dataset)
            plotter.reset_measures()
            for _ in tqdm(range(min(max_batches, len(dataset)))):
                images, labels = next(loader)
                plotter.gather_measures(images, labels, out_sample=out, mc=mc, T=T)
            with open (os.path.join(SAVEPATH, f"{title}_{dataset_name}_measures.pkl"), "wb") as outfile:
                pickle.dump((plotter.correct_scores, plotter.wrong_scores), outfile)
            print(f"Measures saved for {title} {dataset_name}")
            wrong_label = dataset_name if out else "Wrong"
            if out:
                new_fig = False
            else:
                new_fig = True
            plotter.plot_measures(title=f"{title}", out_sample=out, wrong_color=color, 
                                  wrong_label=wrong_label, new_fig=new_fig)
        model_title = title.replace(" ", "_")
        plotter.save_fig(os.path.join(SAVEPATH, f"{model_title.lower()}_eval.png"))


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
#Plot our training losses

with open(os.path.join(CHECKPOINT_PATH, f"{modelnames[0]}_history.pkl"), "rb") as infile:
    dropout_hist = pickle.load(infile)

with open(os.path.join(CHECKPOINT_PATH, f"{modelnames[2]}_history.pkl"), "rb") as infile:
    dropconnect_hist = pickle.load(infile)

#Plot our losses

dropout_acc_fig = plot_history(dropout_hist, ["accuracy", "val_accuracy"], "MC Dropout with p = 0.5")
dropconnect_acc_fig = plot_history(dropconnect_hist, ["accuracy", "val_accuracy"], "MC Dropconnect with p = 0.3")

dropout_acc_fig.savefig(os.path.join(SAVEPATH, "dropout_acc.png"), facecolor="white", transparent=False)
dropconnect_acc_fig.savefig(os.path.join(SAVEPATH, "dropconnect_acc.png"), facecolor="white", transparent=False)
#%%

fig_files = os.listdir(SAVEPATH)

pkl_filter = re.compile(r'^.*pkl')

pkl_files = list(filter(pkl_filter.search, fig_files))

models = ["softmax", "dropout", "dropconnect"]
datasets = ["horse", "cat", "flower"]
scores = ["correct", "wrong"]

measures = {}

for m in models:
    m_filter = re.compile(fr'^.*(?i){m}.*')
    m_files = list(filter(m_filter.search, pkl_files))
    for d in datasets:
        d_filter = re.compile(fr'^.*(?i){d}.*')
        d_files = list(filter(d_filter.search, m_files))
        with open (os.path.join(SAVEPATH, d_files[0]), "rb") as infile:
            correct, wrong = pickle.load(infile)
        measures[f"{m}_{d}_correct"] = correct
        measures[f"{m}_{d}_wrong"] = wrong


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

# %%
model_titles = ["Softmax", "MC Dropout", "MC DropConnect"]
use_mc = [False, True, True]
plotters = [
    CNNPlotter(mc_dropout, labels=LABELS),
    CNNPlotter(mc_dropout, labels=LABELS),
    CNNPlotter(mc_dropconnect, labels=LABELS)
]
datasets = [
    val_ds, 
    catsdogs["train"],
    flowers["train"]
]
dataset_names = [
    "Horse and Human",
    "Cats and Dogs",
    "Flowers"
]
out_sample = [False, True, True]
wrong_color = ["tab:red", "tab:brown", "tab:olive"]

evaluate_measures(
    SAVEPATH, 
    model_titles=["MC DropConnect"], 
    plotters=[CNNPlotter(mc_dropconnect, labels=LABELS)], 
    use_mc=[True], 
    datasets=datasets, 
    dataset_names=dataset_names, 
    out_sample=out_sample, 
    wrong_color=wrong_color,
    max_batches=16,
    T=100,
)


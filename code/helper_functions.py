import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import regex as re
import seaborn as sns
from tqdm import tqdm


def plot_history(history, cols, title=""):
    results = pd.DataFrame(history)
    results = results.loc[:, cols]
    fig, ax = plt.subplots(figsize=(14,14))
    ax = results.plot(ax=ax)
    ax.set_ylim(0,1.1)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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

def load_measures(savepath, models, datasets):
    fig_files = os.listdir(savepath)
    pkl_filter = re.compile(r'^.*pkl')
    pkl_files = list(filter(pkl_filter.search, fig_files))
    measures = {}
    for m in models:
        m_filter = re.compile(fr'^.*(?i){m}.*')
        m_files = list(filter(m_filter.search, pkl_files))
        for d in datasets:
            d_filter = re.compile(fr'^.*(?i){d}.*')
            d_files = list(filter(d_filter.search, m_files))
            with open (os.path.join(savepath, d_files[0]), "rb") as infile:
                correct, wrong = pickle.load(infile)
            measures[f"{m}_{d}_correct"] = correct
            measures[f"{m}_{d}_wrong"] = wrong
    return measures

def plot_measures(measures_dict, scores, colors, labels, ax = None, title=""):
    if ax == None:
        fig, ax = plt.subplots(figsize=(12,12), tight_layout=True)
    colors = colors[:len(scores)]
    for score, color, label in zip(scores, colors, labels):
        sns.distplot(x=measures_dict[score], ax=ax, rug=True, kde=True, hist=False, 
                    kde_kws={"fill": True}, color=color, label=label)
    ax.set_xlabel("Uncertainty Measure")
    ax.axvline(0, color="black", linestyle="dashed")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.05, 0.4)
    ax.set_title(title, fontsize=18)
    ax.legend()
    try:
        return fig
    except UnboundLocalError:
        pass

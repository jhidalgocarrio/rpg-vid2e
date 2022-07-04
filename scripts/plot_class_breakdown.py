import yaml
import matplotlib.pyplot as plt
import os
import numpy as np
import tqdm


def plot_histogram(data, rel=False):
    fig, ax = plt.subplots(figsize=(20,10))#nrows=2)
    data = data["real"]
    keys = sorted(list(data.keys()))
    plot_data = []

    for i, j in enumerate(keys):

        d = data[j]

        # count misclassifications
        incorrect_samples = {}
        num_samples = {}

        for path, topn in tqdm.tqdm(d.items()):
            real_class = os.path.basename(os.path.dirname(path))
            best_class = list(topn[0].keys())[0]

            if real_class not in incorrect_samples :
                incorrect_samples [real_class] = 0
                num_samples[real_class] = 0

            if best_class != real_class:
                incorrect_samples [real_class] += 1

            num_samples[real_class] += 1

        classes = np.array(sorted(list(incorrect_samples.keys())))
        incorrect_samples  = np.array([incorrect_samples [k] for k in classes])
        num_samples = np.array([num_samples[k] for k in classes])
        correct_samples = num_samples - incorrect_samples
        tot_acc = float(correct_samples.sum())/num_samples.sum()

        plot_data += [
            {
                "label": j,
                "tot_acc": tot_acc,
                "correct_samples": correct_samples,
                "num_samples": num_samples
            }
        ]

    sorter = np.argsort(plot_data[0]["num_samples"])
    index = np.arange(len(sorter))
    N = len(keys)+1
    legend = []
    colors = [[[0,0,1], [.7,.7,1]], [[1,0,0],[1,.7,.7]], [[0,0,0],[.7,.7,.7]]]
    for i, pd in enumerate(plot_data):
        shift = (i-N//2)*1.0/N-(N-1)/N*.5
        ax.bar(index+shift, pd["num_samples"][sorter], width=1.0/N, color=colors[i][1])
        ax.bar(index+shift, pd["correct_samples"][sorter], width=1.0/N, color=colors[i][0])
        legend += ["model trained and validated in %s: accuracy=%.3f" % (pd["label"], pd["tot_acc"]), ""]
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_ylabel("Num correctly classified sampled per class")
    plt.xticks(index-.5, classes[sorter], rotation=90)
    plt.legend(legend, loc=2)
    fig.tight_layout()

    #for i, (k, bar_rel) in enumerate(zip(keys, bars_rel)):
    #    colors = ["b", "r"]
    #    shift = (i-N//2)*1.0/N-(N-1)/N*.5
    #    ax[1].bar(index+shift, bar_rel, width=1.0/N, color=colors[i])
    #ax[1].tick_params(axis='x', which='major', labelsize=10)
    #plt.xticks(index+.5, classes, rotation=90)
    #plt.legend(["%s: misclass. rate %.3f" % (k, mcr) for k, mcr in zip(keys, missclassifications_list)], loc=2)

    return fig, ax

with open("/tmp/breakdown.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.Loader)

fig, ax = plot_histogram(data)
fig.savefig("/tmp/ncaltech101_breakdown_histogram.png")
plt.show()
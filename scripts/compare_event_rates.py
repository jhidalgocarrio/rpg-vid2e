import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_events
import tqdm
import sys


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--dataset_1", default="/media/dani/data/ncaltech_experiments/N-Caltech101_numpy/")
    parser.add_argument("--dataset_2", default="/tmp/caltech_bags/")
    parser.add_argument("--label", default="bonsai")
    parser.add_argument("--tmax", type=float, default=0.3)
    parser.add_argument("--bins", type=float, default=100)

    # logging options
    flags = parser.parse_args()

    if not os.path.isdir(flags.dataset_1):
        raise AssertionError("Log directory root %s not found." % flags.dataset_1)
    if not os.path.isdir(flags.dataset_2):
        raise AssertionError("Dataset directory %s not found." % flags.dataset_2)

    return flags

def compute_event_rate(files, tmax=0.3, bins=100):
    data = {}
    for f in files:
        c = os.path.basename(os.path.dirname(f))
        if c not in data:
            data[c] = np.zeros((bins,0))

    for f in tqdm.tqdm(files):
        c = os.path.basename(os.path.dirname(f))
        events = load_events(f)
        t = events[:,2]
        hist, _ = np.histogram(t, bins=bins, range=[0,tmax])
        event_rate = hist.reshape(bins, 1) / (tmax / bins)
        data[c] = np.concatenate([data[c], event_rate], 1)

    for c in data:
        data[c] = [data[c].mean(axis=1), data[c].std(axis=1)]

    return data



def plot_stuff(data1, data2, n):
    rows = n // 5 + 1
    fig, ax = plt.subplots(nrows=rows, ncols=5)
    for i, label in enumerate(sorted(data1.keys())):
        print(data1.keys())
        print(data2.keys())
        mean1, std1 = data1[label]
        mean2, std2 = data2[label]
        row, col = i // 5, i % 5
        ax[row, col].set_title(label)
        ax[row, col].set_ylim([0, 1500000])
        ax[row, col].plot(mean1, color="b")
        ax[row, col].fill_between(np.arange(len(mean1)), mean1-std1, mean1 + std1, color="b", alpha=.5)
        ax[row, col].plot(mean2, color="r")
        ax[row, col].fill_between(np.arange(len(mean1)), mean2-std2, mean2 + std2, color="r", alpha=.5)
        if i == n-1:
            break
    plt.show()

if __name__ == '__main__':
    import glob
    flags = FLAGS()

    dataset_1_files = glob.glob(os.path.join(flags.dataset_1, flags.label, "*"))
    dataset_2_files = glob.glob(os.path.join(flags.dataset_2, flags.label, "*"))

    event_rate_1 = compute_event_rate(dataset_1_files, flags.tmax, flags.bins)
    event_rate_2 = compute_event_rate(dataset_2_files, flags.tmax, flags.bins)

    plot_stuff(event_rate_1, event_rate_2, 20)
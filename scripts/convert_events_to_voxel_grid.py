"""python file for converting folder of rosbags to n-caltech101 classification format.
The rosbag should have the following structure:
- root
    - class1_0001.bag
    - class1_0002.bag
    ...
    - class2_0001.bag
    ...

The rosbags should have the following topics:
    /dvs/events: events, in dvs_EventArray msgs
    /dvs/image_raw: image message
    /dvs_rendering: rendered images with events and frames


The output has the following format:

- output
    - class1
        events_0001.pkl
        images_0001.pkl
        renderings_0001.pkl
        events_0002.pkl
        images_0002.pkl
        renderings_0002.pkl
        :
    - class2
        :
    :

"""

from multiprocessing import Pool
import argparse
import os
import sys
import numpy as np
import glob
import sparse
from PIL import Image
import torch
from utils import save_images, save_events, load_events


def FLAGS():
    parser = argparse.ArgumentParser("Convert Bag folder into N-Caltech folder")
    # root and output folder
    parser.add_argument("--dataset_root", help="Root where the rosbags are.", default="/media/dani/data/ncaltech_experiments/sim-N-Caltech101")
    parser.add_argument("--output_root", help="Root where the output should be.", default="/media/dani/data/ncaltech_experiments/sim-N-Caltech101_vox")
    parser.add_argument("--num_workers", help="Number of threads for converting.", type=int, default=4)
    parser.add_argument("--channels", type=int, help="", default=15)
    parser.add_argument("--resolution", type=int, nargs="+", help="", default=[180,240])
    parser.add_argument("--label", help="", default="*")
    parser.add_argument("--idx", help="", default="*")

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root

    return parser.parse_args()


def parse_file(f):
    """File has shape class1/image_0001.bag"""
    label = os.path.basename(os.path.dirname(f))
    f = os.path.basename(f)
    counter = f[-8:-4]
    return label, counter

def fill_events(events, dimensions, values):
    C, H, W = dimensions
    img = np.zeros((H*W*C,), dtype=np.float32)
    x,y,t,p = events.T

    t_int = t.astype(np.int64)
    r = t-t_int

    lin_idx1 = (x + W * y + W * H * t_int).astype(np.int64)
    lin_idx2 = (x + W * y + W * H * (t_int+1)).astype(np.int64)

    in_bound = (t_int+1 <= C-1)

    np.add.at(img, lin_idx1, (1-r)*values)
    np.add.at(img, lin_idx2[in_bound], (r[in_bound])*values[in_bound])

    img = img.reshape((C, H, W))
    return img

def voxel_grid(events, channels, resolution):
    (H, W), C = resolution, channels

    t = (events[:,2]-events[0,2])/(events[-1,2]-events[0,2])

    if len(events) > 1:
        events[:,2] = (events[:,2]-events[0,2])/(events[-1,2]-events[0,2])
        events[:,2] *= (C-1)

    img_neg = fill_events(events[events[:,-1]==-1], [C, H, W], t[events[:,-1]==-1])
    img_pos = fill_events(events[events[:,-1]==1], [C, H, W], t[events[:,-1]==1])

    vox = np.concatenate([img_neg, img_pos], 0)

    return vox

def convert_1_event_batch(idx_and_file):
    i, file = idx_and_file
    label, counter = parse_file(file)
    label_root = os.path.join(dataset_root, label)
    print("[%4d/%4d] - Converting to voxel grid %s" % (i, len(files), file))
    events = load_events(file)
    events[:,-1] = 2*events[:,-1]-1
    vox = voxel_grid(events, flags.channels, flags.resolution)
    path = os.path.join(label_root, "vox_%s_%s.npy" % (flags.channels, counter))
    np.save(path, vox)

if __name__ == '__main__':
    flags = FLAGS()
    pattern = os.path.join(flags.dataset_root, flags.label, "*%s*"%flags.idx)
    files = glob.glob(pattern)
    print("Found %s files with pattern '%s'" % (len(files), pattern))
    # create output folder
    dataset_root = flags.output_root
    print("Making dataset root at %s." % (dataset_root))
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)
        for folder in glob.glob(os.path.join(flags.dataset_root, "*")):
            os.mkdir(os.path.join(dataset_root, os.path.basename(folder)))

    pool = Pool(flags.num_workers)
    pool.map_async(convert_1_event_batch, enumerate(sorted(files))).get(999999999)  # workaround since otherwise cannot be cancelled

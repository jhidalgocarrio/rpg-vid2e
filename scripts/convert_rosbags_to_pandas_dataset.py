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
from PIL import Image
from io_utils import save_images, save_events


def FLAGS():
    parser = argparse.ArgumentParser("Convert Bag folder into N-Caltech folder")
    # rostopics
    parser.add_argument("--event_topic", help="Topic of the events.", default="" )
    parser.add_argument("--image_topic", help="Topic of the frames.", default="/cam0/image_raw" )
    parser.add_argument("--rendering_topic", help="Topic of the renderings.", default="" )

    # root and output folder
    parser.add_argument("--dataset_root", help="Root where the rosbags are.", default="/media/dani/data/caltech_bags")
    parser.add_argument("--output_root", help="Root where the output should be.", default="/media/dani/data/caltech_image_folders")
    parser.add_argument("--label", help="", default="")
    parser.add_argument("--ros_version", help="version of ROS.", default="melodic")
    parser.add_argument("--num_workers", help="Number of threads for converting.", type=int, default=4)

    parser.add_argument("--event_output_type",
                        help="Type of output for events, can be one of the following: [pandas, numpy].",
                        default="numpy")
    parser.add_argument("--image_output_type",
                        help="Type of output for images, can be one of the following: [pandas, numpy, folder].",
                        default="folder")

    args = parser.parse_args()

    assert args.event_output_type in ["pandas", "numpy"]
    assert args.image_output_type in ["pandas", "numpyz", "folder"]
    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root
    assert os.path.isdir(args.output_root), "%s should be a valid dir." % args.output_root

    return parser.parse_args()


def parse_rosbag_file(f):
    """File has shape class1_0001.bag"""
    label = os.path.basename(os.path.dirname(f))
    f = os.path.basename(f)
    counter = f[-8:-4]
    return label, counter

def load_data_from_rosbag(label_root, counter, f, flags):
    """load data from rosbag"""
    # load ros version
    sys.path.insert(0, "/opt/ros/%s/lib/" % flags.ros_version)
    from bag_to_dataframe import Bag2Events, Bag2Images

    endings = {"frame": "", "numpyz":".npz", "numpy": ".npy", "pandas": ".pkl", "folder": ""}
    image_ending = endings[flags.image_output_type]
    events_ending = endings[flags.event_output_type]

    dataframes = {}

    if flags.image_topic != "":
        base = flags.image_topic.replace("/", "_").strip("_")
        path = os.path.join(label_root, "%s_%s%s" % (base, counter, image_ending))
        if not os.path.exists(path):
            dataframes[base] = (path, Bag2Images(f, flags.image_topic, 0, 0))

    if flags.event_topic != "":
        base = flags.event_topic.replace("/", "_").strip("_")
        path = os.path.join(label_root, "%s_%s%s" % (base, counter, events_ending))
        if not os.path.exists(path):
            dataframes[base] = (path, Bag2Events(f, flags.event_topic, 0, 0))

    return dataframes


def convert_1_bag(idx_and_file):
    i, rosbag_file = idx_and_file
    label, counter = parse_rosbag_file(rosbag_file)

    label_root = os.path.join(dataset_root, label)
    if not os.path.isdir(label_root):
        os.mkdir(label_root)

    print("[%4d/%4d] - Converting rosbag %s -> %s" % (
        i, len(rosbag_files), rosbag_file, os.path.join(label_root, "events_%s.npy" % counter)))

    try:
        dataframes = load_data_from_rosbag(label_root, counter, rosbag_file, flags)
    except Exception as e:
        print("Failed loading sample with class %s and idx %s: " % (label_root, counter))
        raise e

    for key, (path, dataframe) in dataframes.items():
        is_events = "event" in key
        if is_events:
            save_events(dataframe.df, path)
        else:
            save_images(dataframe.df, path)


if __name__ == '__main__':
    flags = FLAGS()
    rosbag_files = glob.glob(os.path.join(flags.dataset_root, flags.label,"outdoor_day1_davis_left_data_and_gt.bag"))

    # create output folder
    dataset_root = os.path.join(flags.output_root, "dataset")
    print("Making dataset root at %s." % (dataset_root))
    if not os.path.isdir(dataset_root):
        os.mkdir(os.path.join(flags.output_root, "dataset"))
        for folder in glob.glob(os.path.join(flags.dataset_root, "*")):
            os.mkdir(os.path.join(dataset_root, os.path.basename(folder)))

    pool = Pool(flags.num_workers)
    pool.map_async(convert_1_bag, enumerate(sorted(rosbag_files))).get(999999999)  # workaround since otherwise cannot be cancelled

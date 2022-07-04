from multiprocessing import Pool
import argparse
import os
import sys
import numpy as np
import glob
from io_utils import save_events


def FLAGS():
    parser = argparse.ArgumentParser("Convert Bag folder into N-Caltech folder")
    # rostopics
    parser.add_argument("--event_topic", help="Topic of the events.", default="" )
    parser.add_argument("--file", help="Topic of the frames.", default="" )

    return parser.parse_args()


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
        i, len(rosbag_files), rosbag_file, os.path.join(label_root, "*_%s.npy" % counter)))

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
    sys.path.insert(0, "/opt/ros/melodic/lib/")
    from bag_to_dataframe import Bag2Events, Bag2Images

    flags = FLAGS()

    events_df = Bag2Events(flags.file, flags.event_topic, 0, 0)
    save_events(events_df.to_xytp(), flags.file + ".txt")

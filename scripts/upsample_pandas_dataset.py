"""Upsamples frames from a Pandas Dataset using the code from Super-SlowMo https://github.com/avinashpaliwal/Super-SloMo"""



import argparse
import os
import glob

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rpg_Super-SlowMo"))
from video_to_slomo import Converter


def FLAGS():
    parser = argparse.ArgumentParser("Upsamples Pandas Dataset")
    # root and output folder
    parser.add_argument("--dataset_root", help="Root of dataset.", default="/home/dani/code/catkin_ws/src/rpg_vid2e/dataset")
    parser.add_argument("--file_prefix", help="File that needs to be converted.", default="images")
    parser.add_argument("--upsampling_factor", type=int, help="Upsampling Factor.", default=2)
    parser.add_argument("--device", type=str, help="Device.", default="cuda:0")
    parser.add_argument("--batch_size", type=int, help="Number of samples per batch.", default=8)

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.datasetf_root

    return parser.parse_args()


def parse_rosbag_file(f):
    """File has shape class1_0001.bag"""
    f = os.path.basename(f)
    label, counter = f[:-15], f[-8:-4]
    return label, counter


if __name__ == '__main__':
    flags = FLAGS()

    files_to_process = glob.glob(os.path.join(flags.dataset_root, "*", flags.file_prefix + "*.pkl"))
    upsampling_script = os.path.join(os.path.dirname(__file__), "rpg_Super-SlowMo", "video_to_slomo.py")
    checkpoint = os.path.join(os.path.dirname(__file__), "SuperSloMo.ckpt")

    converter = Converter(checkpoint, flags.upsampling_factor, flags.device, flags.batch_size)

    for i, frames_file in enumerate(sorted(files_to_process)):
        print("[%4d/%4d] - upsampling %s" % (i, len(files_to_process), frames_file))
        converter.upsample_1(frames_file)
    print("Done")

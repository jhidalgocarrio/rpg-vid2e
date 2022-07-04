import argparse
import os
import glob
import pandas as pd
from PIL import Image
from multiprocessing import Pool


def FLAGS():
    parser = argparse.ArgumentParser("Extract frames from pandas dataframes.")
    # rostopics
    parser.add_argument("--dataset_root", help="Location of dataset.", default="/media/dani/data/tmp/")
    parser.add_argument("--num_workers", help="Number of workers to perform extraction.", type=int, default=1)
    parser.add_argument("--prefix", help="Prefix of images to convert.", default="vide")
    parser.add_argument("--ros_verbosity", help="Verbosity of ros.", default=1)

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root

    return parser.parse_args()


def load_ros_flags(config):
    flag_list = []
    with open(config, "r") as fh:
        lines = fh.readlines()

    for l in lines:
        # commented or empty line
        if l.startswith("#") or "--" not in l:
            continue

        flag_list += l.replace("\n", "").split("=")

    return flag_list


def convert_frames(idx_f):
    i, f = idx_f

    print("[%4d/%4d] - converting frames from - %s" % (i, len(folders), f))

    # load flagfile
    esim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rpg_esim_private"))
    config_file = os.path.join(esim_root, "event_camera_simulator", "esim_ros", "cfg", "test_esim_pytorch.conf")

    ros_flags = load_ros_flags(config_file)

    # update --path_to_output_bag to point to output
    if "--v" in ros_flags:
        ros_flags[ros_flags.index("--v") + 1] = str(flags.ros_verbosity)
    else:
        ros_flags += ["--v", str(flags.ros_verbosity)]

    # update --path_to_output_bag to point to output
    if "--path_to_data" in ros_flags:
        ros_flags[ros_flags.index("--path_to_data") + 1] = f
    else:
        ros_flags += ["--path_to_data_folder", f]

    # update calib, texture and
    calib_filename = os.path.join(esim_root, "event_camera_simulator", "esim_ros", "cfg", "calib", "pinhole_mono_nodistort_1080.yaml")
    ros_flags[ros_flags.index("--calib_filename")+1] = calib_filename

    # set datasource to 2 (read from folder)
    ros_flags[ros_flags.index("--data_source")+1] = str(2)

    # set bag output name
    path_to_output_bag = os.path.normpath(f) + ".bag"
    if "--path_to_output_bag" in ros_flags:
        ros_flags[ros_flags.index("--path_to_output_bag") + 1] = path_to_output_bag
    else:
        ros_flags += ["--path_to_output_bag", path_to_output_bag]

    # write file to tmp
    with open("/tmp/flags_%s.txt" % i, "w") as fh:
        for j in range(len(ros_flags)//2):
            fh.write(ros_flags[2*j] + "=" + ros_flags[2*j+1] + "\n")

    # cp calibration file to image folder
    os.system("cp %s %s" % (calib_filename, os.path.join(f, "calib.yaml") ))

    # create cmd
    silent=""#2>/dev/null"
    cmd = "rosrun esim_ros esim_node --flagfile=/tmp/flags_%s.txt %s --v=3" % (i,silent)
    os.system(cmd)


if __name__ == '__main__':
    flags = FLAGS()

    # get all files to process and process them
    pattern = os.path.join(flags.dataset_root, flags.prefix + "*")
    folders = glob.glob(pattern)
    folders = [p for p in folders if os.path.isdir(p) ]

    if len(folders) == 0:
        print("No folders found to process.")
        exit(1)

    folders = sorted(folders)
    pool = Pool(flags.num_workers)
    pool.map_async(convert_frames, enumerate(sorted(folders))).get(999999999)  # workaround since otherwise cannot be cancelled
import os
import argparse

def FLAGS():
    parser = argparse.ArgumentParser("Convert Bag folder into N-Caltech folder")
    # root and output folder
    parser.add_argument("--dataset_root", help="Root where the rosbags are.", default="/media/dani/Samsung_T53/datasets/ncaltech_bags/")
    parser.add_argument("--output_root", help="Root where the output should be.", default="/media/dani/Samsung_T53/datasets/")

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root
    assert os.path.isdir(args.output_root), "%s should be a valid dir." % args.output_root

    return parser.parse_args()


if __name__ == '__main__':
    s1 = "/home/dani/code/catkin_ws/src/rpg_vid2e/scripts/convert_rosbags_to_pandas_dataset.py"

    flags = FLAGS()

    # convert from rosbag to pandas
    os.system("python %s --event_topic %s --dataset_root %s --output_root %s --num_workers %s" % \
              (
                s1,
                "/cam0/events",
                flags.dataset_root,
                flags.output_root,
                8
              ))
    # convert from pandas to




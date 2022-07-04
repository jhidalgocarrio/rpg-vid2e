import argparse
import os
import glob
import pandas as pd
from PIL import Image
from multiprocessing import Pool


def FLAGS():
    parser = argparse.ArgumentParser("Create dataset split for reproducibility")
    # rostopics
    parser.add_argument("--dataset_root", help="Location of dataset.", default="/home/dani/code/catkin_ws/src/rpg_vid2e/dataset")
    parser.add_argument("--num_workers", help="Number of workers to perform extraction.", type=int, default=4)
    parser.add_argument("--prefix", help="Prefix of images to convert.", default="cam0_image_raw")

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root

    return parser.parse_args()


def extract_frames(idx_f):
    i, f = idx_f
    df = pd.read_pickle(f)

    # mkdir
    dirname = os.path.dirname(f)
    basename = os.path.basename(f)
    extraction_folder = os.path.join(dirname, "extracted_" + basename.split(".")[-2])
    cam_folder = os.path.join(extraction_folder, "frame", "cam0")
    os.system("mkdir -p %s" % cam_folder)

    # save frames
    print("[%4d/%4d] - extracting file - %s" % (i, len(files), f))
    for i, row in df.iterrows():
        img = row["image"]
        img_pil = Image.fromarray(img)
        if img.shape[-1] == 3:
            img_pil = img_pil.convert('RGB')
        frame_name = "frame_%s.png" % str(i).zfill(4)
        img_pil.save(os.path.join(cam_folder, frame_name))
        with open(os.path.join(extraction_folder, "images.csv"), "a") as f_handle:
            f_handle.write("%s %s\n" % (int(row["time"]*1e9), frame_name))


if __name__ == '__main__':
    flags = FLAGS()

    files = glob.glob(os.path.join(flags.dataset_root, "*", flags.prefix + "*.pkl"))
    if len(files) == 0:
        print("No files found to extract.")
        exit(1)
    pool = Pool(flags.num_workers)
    pool.map_async(extract_frames, enumerate(sorted(files))).get(999999999)  # workaround since otherwise cannot be cancelled
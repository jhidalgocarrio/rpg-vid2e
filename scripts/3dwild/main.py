import argparse
import os
import glob
import sys
import numpy as np
import skvideo.io
import subprocess
import pandas as pd
import tqdm
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2



def extract_frames(f, output_folder, flags):
    print("Extracting %s -> %s" % (f, output_folder))
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder, "images"))

    os.system("ffmpeg -i %s %s/image_%%05d.png > /dev/null 2>&1" % (f, os.path.join(output_folder, "images")))
    num_images = len(os.listdir(os.path.join(output_folder, "images")))
    t = np.array([float(i)/flags.fps for i in range(num_images)]).reshape((-1,1))
    np.savetxt(os.path.join(output_folder, "timestamps.txt"), t)

def upsample_frames(f, flags):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rpg_Super-SlowMo"))

    from video_to_slomo import Converter

    meta = skvideo.io.ffprobe(f)
    meta = {"it": skvideo.io.vreader(f), "meta": meta}
    fps = eval(meta["meta"]["video"]["@avg_frame_rate"])

    converter = Converter(meta=meta,
                          adaptive=flags.adaptive,
                          device=flags.upsample_device,
                          batch_size=flags.upsample_batchsize,
                          resize_dim=flags.resize_dim)

    output_folder = f.split(".")[0] + "_upsampled"

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder, "images"))

    if os.path.isfile(os.path.join(output_folder, "timestamps.txt")):
        os.system("rm %s" % os.path.join(output_folder, "timestamps.txt"))

    idx = 0

    for frames, ts in converter:
        for frame, t in zip(frames, ts):
            path = os.path.join(output_folder, "images", "image_%05d.png" % (idx+1))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(path, frame)
            with open(os.path.join(output_folder, "timestamps.txt"), "a") as fh:
                string = "\n%i" % (t*1e9) if idx > 0 else "%i" % (t*1e9)
                fh.write(string)

            idx += 1

        if idx >= flags.n_upsample_frames and flags.n_upsample_frames > 0:
            break

    print("Upsampled: %s frames" % (-1+len(os.listdir(os.path.join(output_folder, "images")))))

    return output_folder, fps

def load_ros_flags(config):
    flag_list = {}
    with open(config, "r") as fh:
        lines = fh.readlines()

    for l in lines:
        # commented or empty line
        if l.startswith("#") or "--" not in l:
            continue

        flag_list[l.replace("\n", "").split("=")[0]] = l.replace("\n", "").split("=")[1]

    return flag_list

def convert_to_events(input_folder, flags, fps):
    # load flagfile
    output_folder = os.path.normpath(input_folder) + "_events"
    print("Converting events in %s -> %s" % (input_folder, output_folder))
    esim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rpg_esim_private"))
    config_file = os.path.join(esim_root, "event_camera_simulator", "esim_ros", "cfg", "test_esim_pytorch.conf")

    ros_flags = load_ros_flags(config_file)

    # update --path_to_output_bag to point to output
    ros_flags["--path_to_video"] = os.path.join(input_folder, "images.avi")
    ros_flags["--video_frame_timestamps"] = os.path.join(input_folder, "timestamps.txt")

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    assert os.path.isdir(output_folder), output_folder

    # update --path_to_output_bag to point to output
    if len(flags.C_range) > 0:
        Cm, Cp = flags.C_range[0] + np.random.random((2,)) * (flags.C_range[1]-flags.C_range[0])
    else:
        Cp = flags.contrast_threshold_pos
        Cm = flags.contrast_threshold_neg

    f_txt = os.path.join(output_folder, "events.txt")
    ros_flags["--path_to_output_text_file"] = f_txt
    ros_flags["--data_source"] = str(5)
    ros_flags["--path_to_output_bag"] = ""
    ros_flags["--contrast_threshold_pos"] = Cp
    ros_flags["--contrast_threshold_neg"] = Cm
    ros_flags["--log_eps"] = .001
    ros_flags["--use_log_image"] = flags.use_log_image
    ros_flags["--renderer_preprocess_gaussian_blur"] = 0
    ros_flags["--renderer_preprocess_median_blur"] = 0
    ros_flags["--v"] = 1

    # write file to tmp
    flags_path = "%s/flags.txt" % output_folder
    with open(flags_path, "w") as fh:
        for k, v in ros_flags.items():
            fh.write("%s=%s\n" % (k, v))

    with open(os.path.join(output_folder, "config.yaml"), "w") as fh:
        meta = skvideo.io.ffprobe(ros_flags["--path_to_video"])
        W = eval(meta["video"]["@width"])
        H = eval(meta["video"]["@height"])
        fh.write("resolution: [%s, %s]\n" % (H, W))

    # create cmd
    silent = ""#"> /dev/null"
    cmd = "roslaunch esim_ros mpii.launch config:=%s v:=1 %s" % (flags_path, silent)

    os.system(cmd)

    # convert events to memmap
    print("Generating memmap.")
    f_dat = os.path.join(output_folder, "events.dat")
    index_file = os.path.join(output_folder, "index.npy")
    convert_txt_to_memmap(f_txt, f_dat, index_file, fps)

    return output_folder

def convert_txt_to_memmap(f_txt, f_dat, index_file, fps):
    num_lines = sum(1 for _ in open(f_txt))
    print("Processing %s events" % num_lines)
    memmap = np.memmap(f_dat, dtype="int64", mode="w+", shape=(num_lines, 4))

    event_index = []
    counter = 1

    with open(f_txt, "r") as fh:
        for index in tqdm.tqdm(range(num_lines)):
            line = fh.readline()

            t_next = counter * 1e9 / fps

            x, y, t, p = line[:-1].split(" ")
            memmap[index, :] = np.array([int(x), int(y), int(t), int(p)], dtype=np.int64)

            if index % 10000 == 0:
                memmap.flush()

            index += 1

            if int(t) < t_next :
                continue

            event_index += [[t_next, index - 1]]
            counter += 1

    del memmap
    os.remove(f_txt)

    np.save(index_file, np.array(event_index).astype(np.int64))


def process(f, flags):
    if flags.upsample_frames:
        upsampled_frames_folder, fps = upsample_frames(f, flags)
    else:
        upsampled_frames_folder = f
        fps = flags.fps

    if flags.convert_to_events:
        events_folder = convert_to_events(upsampled_frames_folder, flags, fps)


def FLAGS():
    parser = argparse.ArgumentParser("Extract frames from pandas dataframes.")
    # rostopics
    parser.add_argument("--files", help="Location of dataset.", nargs="+", default=[])

    parser.add_argument("--extract_frames", help="Verbosity of ros.", action="store_true", default=False)
    parser.add_argument("--fps", help="Verbosity of ros.", type=int, default=50)

    parser.add_argument("--upsample_frames", help="Verbosity of ros.", action="store_true", default=False)
    parser.add_argument("--upsample_factor", help="Verbosity of ros.", type=int, default=2)
    parser.add_argument("--upsample_batchsize", help="Verbosity of ros.", type=int, default=1)
    parser.add_argument("--downsample_factor", help="Verbosity of ros.", type=int, default=4)
    parser.add_argument("--upsample_device", help="", default="cuda:0")
    parser.add_argument("--n_upsample_frames", help="Verbosity of ros.", type=int, default=-1)

    parser.add_argument("--convert_to_events", help="Verbosity of ros.", action="store_true", default=False)
    parser.add_argument("--frame_buffer", help="Verbosity of ros.", type=int, default=15)
    parser.add_argument("--generate_events_every_n_frames", help="Verbosity of ros.", type=int, default=15)
    parser.add_argument("--contrast_threshold_pos", help="Verbosity of ros.", type=float, default=0.1)
    parser.add_argument("--contrast_threshold_neg", help="Verbosity of ros.", type=float, default=0.1)
    parser.add_argument("--C_range", help="Verbosity of ros.", type=float, nargs="+", default=[])
    parser.add_argument("--log_eps", help="Verbosity of ros.", type=float, default=.001)
    parser.add_argument("--use_log_image", help="Verbosity of ros.", action="store_true", default=False)
    parser.add_argument("--renderer_preprocess_gaussian_blur", type=int, default=0)
    parser.add_argument("--renderer_preprocess_median_blur", type=int, default=1)
    parser.add_argument("--iterator", action="store_true", default=False)
    parser.add_argument("--convert_events_on_the_fly", action="store_true", default=False)
    parser.add_argument("--adaptive", action="store_true", default=False)
    parser.add_argument("--resize_dim", nargs="+", type=int, default=[346,346])

    return parser.parse_args()


if __name__ == '__main__':
    flags = FLAGS()

    assert len(flags.files) > 0
    print("Processing files: %s" % flags.files)
    for file in flags.files:
        process(file, flags)

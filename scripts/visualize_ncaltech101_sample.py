import numpy as np
import argparse
import sys, os
import tqdm
import skvideo.io
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
from io_utils import load_events
import cv2
from io_utils import Visualizer
import glob


class NCaltech101Visualizer(Visualizer):
    def get_sensor_size(self, data):
        return cv2.imread(data[0][1]["paths"][0]).shape[:2]

    def render(self, data, t):
        t_first = t - self.dt

        resolution = self.sensor_size

        # extract data between times
        events, images = data

        data_between_time = events[(events[:,2] < t) & (events[:,2] >= t_first)]

        event_image = self.event_image(data_between_time, resolution)

        idx = max([np.searchsorted(images["timestamps"], t)-1, 0])
        image = cv2.imread(images["paths"][idx])

        return_image = np.concatenate([event_image, image], 0)

        return return_image

    def event_image(self, events, sensor_size):
        """x,y,p"""

        H, W = sensor_size
        lin_idx = (events[:,0] + events[:,1]*W).astype(np.int64)
        pos = np.zeros(H*W, dtype=np.float)
        neg = np.zeros(H*W, dtype=np.float)
        none = np.zeros(H*W, dtype=np.float)

        np.add.at(pos, lin_idx[events[:,-1]==1], events[events[:,-1]==1,-1]**2)
        np.add.at(neg, lin_idx[events[:,-1]==-1], events[events[:,-1]==-1,-1]**2)

        img = np.stack([pos,neg,none], -1)
        img = img.reshape((H, W, 3))

        # convert to uint8
        img = np.clip(255*img/3, 0, 255).astype(np.uint8)
        return img


def FLAGS():
    parser = argparse.ArgumentParser("Create Visualization of Sample")
    # rostopics

    parser.add_argument("--events", help="data sample.", nargs="+",
                        default=[
                            "/media/dani/data/ncaltech/temp/car_side/image_0001.npz",
                            "/media/dani/data/ncaltech/temp_downsampled_factor_16_upsampled_factor_16/car_side/image_0001.npz",
                            "/media/dani/data/ncaltech/temp_downsampled_factor_16/car_side/image_0001.npz",
                        ] )
    parser.add_argument("--images", help="data sample.", nargs="+",
                        default=[
                            "/media/dani/data/ncaltech/temp/car_side/image_0001/",
                            "/media/dani/data/ncaltech/temp_downsampled_factor_16_upsampled_factor_16/car_side/image_0001/",
                            "/media/dani/data/ncaltech/temp_downsampled_factor_16/car_side/image_0001/"
                        ] )
    parser.add_argument("--dt", help="Time resolution.", default=0.01, type=float)
    parser.add_argument("--tmax", help="Time resolution.", default=0.35, type=float)
    parser.add_argument("--tmin", help="Time resolution.", default=0, type=float)
    parser.add_argument("--fps", help="Time resolution.", default=500, type=float)
    parser.add_argument("--video_path", default="/tmp/car_side_16x_downsampled.avi")
    parser.add_argument("--video_framerate", type=int, default=30)

    args = parser.parse_args()

    if args.video_path != "":
        assert os.path.isdir(os.path.dirname(args.video_path)), "%s should be valid dir." % os.path.dirname(args.video_path)

    return parser.parse_args()

def load_images(f):
    paths = sorted(glob.glob(os.path.join(f, "images", "*")))
    timestamps = np.genfromtxt(os.path.join(f, "timestamps.txt")).astype("float32")/1e9

    return {"paths": paths, "timestamps": timestamps}

if __name__ == '__main__':
    flags = FLAGS()

    print("Loading events...")
    events = [load_events(s) for s in flags.events]
    images = [load_images(s) for s in flags.images]

    data = list(zip(events, images))

    viz = NCaltech101Visualizer(dt=flags.dt,
                                fps=flags.fps,
                                data=data,
                                tmin=flags.tmin,
                                tmax=flags.tmax)

    if flags.video_path != "":
        viz.write_to_video(flags.video_path, flags.video_framerate)
    else:
        viz.run()

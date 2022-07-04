import numpy as np
import argparse
import sys, os
import tqdm
import skvideo.io
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from io_utils import load_events
from PIL import Image, ImageDraw
from io_utils import Visualizer
import cv2
import glob
import matplotlib.pyplot as plt


def generate_event_image(events, sensor_size):
    """x,y,p"""

    H, W = sensor_size
    lin_idx = (events[:,0] + events[:,1]*W).astype(np.int64)
    pos = np.zeros(H*W, dtype=np.float)
    neg = np.zeros(H*W, dtype=np.float)
    none = np.zeros(H*W, dtype=np.float)

    np.add.at(pos, lin_idx[events[:,-1]==1], np.ones_like(events[events[:,-1]==1,-1]))
    np.add.at(neg, lin_idx[events[:,-1]==0], np.ones_like(events[events[:,-1]==0,-1]))

    img = np.stack([pos,neg,none], -1)
    img = img.reshape((H, W, 3))

    # convert to uint8
    img = np.clip(255*(img).astype(np.float32), 0, 255).astype(np.uint8)
    return img

def draw_poses(image, x, y, minimal=False):
    x, y = x.astype(int), y.astype(int)
    body_points = ['head_top',
                   'left_shoulder', 'right_shoulder',
                   'left_elbow', 'right_elbow',
                   'left_hand', 'right_hand',
                   'left_hip', 'right_hip',
                   'left_knee', 'right_knee',
                   'left_foot', 'right_foot']
    point_lines = [
        ['head_top', 'right_shoulder'],
        ['right_shoulder', 'right_elbow'],
        ['right_elbow', 'right_hand'],
        ['head_top', 'left_shoulder', ], ["left_shoulder", "left_elbow"], ["left_elbow", "left_hand"],
        ["right_shoulder", "left_shoulder"], ["right_shoulder", 'right_hip'], ["left_shoulder", 'left_hip'], ['left_hip', 'right_hip'],
        ['left_shoulder', 'right_hip'], ['left_hip', 'right_shoulder'],
        ['right_hip', 'right_knee'], ['right_knee', 'right_foot'],
        ['left_hip', 'left_knee'], ['left_knee', 'left_foot']
        ]
    all_body_parts = [
        'spine3', 'spine4', 'spine2', 'spine', 'pelvis', 'neck', 'head', 'head_top', 'left_clavicle',
        'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder',
        'right_elbow', 'right_wrist', 'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot',
        'left_toe', 'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe'
        ]

    rel_parts = body_points if minimal else all_body_parts


    cmap = plt.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0,1, len(point_lines)))

    image = Image.fromarray(image)

    image1 = image.copy()
    draw = ImageDraw.Draw(image)
    s = 1
    for j, (c, (p1, p2)) in enumerate(zip(colors, point_lines)):
        c = tuple([int(255*ci) for ci in c])

        i1 = rel_parts.index(p1)
        x1, y1 = x[i1], y[i1]
        i2 = rel_parts.index(p2)
        x2, y2 = x[i2], y[i2]

        draw.line(((x1, y1), (x2, y2)), fill=c)
        draw.ellipse(((x1-s, y1-s), (x1+s, y1+s)), fill=c)
        draw.ellipse(((x2-s, y2-s), (x2+s, y2+s)), fill=c)

    image = Image.blend(image, image1, 0.5)

    image = np.array(image)

    return image


class MPIPoseVisualizer(Visualizer):
    def get_range(self, data):
        tmin = np.inf
        tmax = -1
        for d in data:
            e = d[0]
            if len(e)>0:
                tmin_, tmax_ = e[0, 2], e[-1, 2]
                tmin = min([tmin, tmin_])
                tmax = max([tmax, tmax_])
        return tmin*1e-9, tmax*1e-9

    def render(self, data, i, resolution):
        t_last = self.t_range[0] + float(i) / self.fps
        t_first = t_last - self.dt

        # extract data between times
        events, (images, t_image), poses = data
        data_between_time = events[(events[:,2]*1e-9 < t_last) & (events[:,2]*1e-9>= t_first)]
        event_image = generate_event_image(data_between_time, resolution)

        # find image
        img_idx = np.searchsorted(t_image, t_last)-1
        if img_idx == -1:
            img_idx = 0
        image = cv2.imread(images[img_idx])[...,[2,1,0]]

        # find pose205532808
        pose_idx = np.searchsorted(poses[:,0], t_last)-1
        if pose_idx == -1:
            pose_idx = 0
        pose_x = poses[pose_idx,1:][::2]
        pose_y = poses[pose_idx,1:][1::2]

        H, W, C = image.shape
        image = cv2.resize(image, (W, H))
        mask = (event_image[...,0]==0) & (event_image[...,1]==0)
        event_image[mask,0] = image[mask, 0]
        event_image[mask,1] = image[mask, 0]
        event_image[mask,2] = image[mask, 0]

        image = draw_poses(event_image, pose_x, pose_y)

        return image


def FLAGS():
    parser = argparse.ArgumentParser("Create Visualization of Sample")
    # rostopics

    parser.add_argument("--samples", help="data sample.", nargs="+",
                        default=["a", "/media/dani/data/3dwild/mpi_inf_3dhp/S4/Seq2/imageSequence/video_0"] )

    parser.add_argument("--dt", help="Time resolution.", default=0.03, type=float)
    parser.add_argument("--tmax", help="Time resolution.", default=30, type=float)
    parser.add_argument("--tmin", help="Time resolution.", default=10, type=float)
    parser.add_argument("--fps", help="Time resolution.", default=100, type=float)
    parser.add_argument("--video_path", default="/tmp/s4_seq2_vid0.avi")

    args = parser.parse_args()

    if args.video_path != "":
        assert os.path.isdir(os.path.dirname(args.video_path)), "%s should be valid dir." % os.path.dirname(args.video_path)

    return parser.parse_args()

def load_samples(s):
    dirs = [d for d in glob.glob(s + "*") if os.path.isdir(d)]
    for d in dirs:
        if d.endswith("events"):
            index = np.load(os.path.join(d,"index.npy"))
            events = np.memmap(os.path.join(d, "events.dat"), dtype="int64", mode="r", shape=(index[-1,-1], 4))
        elif "fps" in d:
            images = sorted(glob.glob(os.path.join(d, "images", "*.png")))
            image_timestamps = np.genfromtxt(os.path.join(d, "timestamps.txt"))*1e-9

    base = os.path.basename(s)
    dirname = os.path.dirname(s)
    vicon_file = os.path.join(dirname, "vicon_%s.npy" % base[-1])

    vicon_data = np.load(vicon_file)

    return events, (images, image_timestamps), vicon_data


if __name__ == '__main__':
    flags = FLAGS()

    labels = flags.samples[::2]
    samples = flags.samples[1::2]

    print("Loading events...")
    event_samples = [s + "events.npy" for s in samples]
    
    data = [load_samples(s) for s in samples]

    H, W, _ = cv2.imread(data[0][1][0][0]).shape

    scale = H/2048
    for i in range(len(data)):
        data[i][-1][:,1:] *=scale

    resolutions = [[H,W]] * 3

    viz = MPIPoseVisualizer(flags.dt, flags.fps, labels, data, resolutions, flags.tmax, flags.tmin, precompute=False)

    if flags.video_path != "":
        viz.write_to_video(flags.video_path)
    else:
        viz.run()
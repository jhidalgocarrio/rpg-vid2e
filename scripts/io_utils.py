import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import skvideo
import tqdm
import cv2

def load_numpy(f):
    return np.load(f)

def load_pandas(f):
    return pd.read_pickle(f).as_matrix(["x", "y", "time", "polarity"]).astype(np.float32)

def load_bag(f):
    sys.path.insert(0, "/opt/ros/%s/lib/" % "melodic")
    from bag_to_dataframe import Bag2Events, Bag2Images
    return Bag2Events(f).df.as_matrix(["x", "y", "time", "polarity"]).astype(np.float32)

def load_events(f):
    if f.endswith(".bag"):
        return load_bag(f)
    elif f.endswith(".pkl"):
        return load_pandas(f)
    elif f.endswith(".npy"):
        return load_numpy(f)
    elif f.endswith(".dat"):
        return load_memmap(f)
    elif f.endswith(".npz"):
        return load_npz(f)

def load_npz(f):
    events = np.load(f)
    xyp = events["xyp"].astype("float32")
    t = events["t"]
    p = 2*xyp[:,-1:]-1
    events = np.concatenate([xyp[:,:2], t, p], -1)
    return events

def load_memmap(f):
    num_events = int(os.stat(f).st_size / 32)
    events_handle = np.memmap(f, dtype="int64", mode="r", shape=(num_events, 4))
    events = np.array(events_handle)

    t = events[:, 2]
    t -= t[0]
    t = t.astype("float32")*1e-9

    events = events.astype(np.float32)
    events[:, -1] = 2 * events[:, -1] - 1
    events[:, 2] = t

    return events

def save_images(df, path):
    if path.endswith(".npz"):
        images = np.stack(list(df["images"]))
        times = np.array(list(df["time"]))
        np.savez(path, images=images, times=times)
    elif path.endswith(".pkl"):
        df.to_pickle(path)
    else:
        save_images_to_folder(df, path)

def save_images_to_folder(df, path):
    path += "_folder"
    cam_folder = os.path.join(path, "frame", "cam0")
    os.system("mkdir -p %s" % cam_folder)

    # save frames
    for i, row in df.iterrows():
        img = row["image"]
        img_pil = Image.fromarray(img)
        if img.shape[-1] == 3:
            img_pil = img_pil.convert('RGB')
        frame_name = "frame_%s.png" % str(i).zfill(4)
        img_pil.save(os.path.join(cam_folder, frame_name))
        with open(os.path.join(path, "images.csv"), "a") as f_handle:
            f_handle.write("%s %s\n" % (int(row["time"]*1e9), frame_name))

def save_events(df, path):
    if path.endswith(".pkl"):
        df.to_pickle(path)
    elif path.endswith(".npy"):
        np.save(path, df.as_matrix(["x", "y", "time", "polarity"]).astype(np.float32))
    elif path.endswith(".txt"):
        np.savetxt(path, df, fmt="%i %i %.8f %i")
    else:
        raise ValueError("Path must end with either .pkl or .npy")


class Visualizer:
    def __init__(self, dt, fps, data, sensor_size=None, tmax=-1, tmin=-1):
        self.data = data  # [[d11, ..., d1N], [d21, ..., d2N], ..., [dM1, ..., dMN]]
        self.t_range = [tmin, tmax]
        self.dt, self.fps = dt, fps

        self.sensor_size = sensor_size
        if self.sensor_size is None:
            self.sensor_size = self.get_sensor_size(data)

        self.data = data

        # precompute
        self.i_max = np.ceil((self.t_range[1] - self.t_range[0])*self.fps).astype(int)

        self.fig = None
        self.ax = None
        self.image_handle = None

    def get_sensor_size(self, data):
        raise NotImplementedError

    def render(self, data, i):
        raise NotImplementedError

    def run(self):
        i = 0
        while True:
            t = self.t_range[0] + float(i) / self.fps
            images = [self.render(d, t) for d in self.data]
            image = np.concatenate(images, 1)

            self.plot(image)

            i = (i + 1) % self.i_max

    def plot(self, image):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

        if self.image_handle is None:
            self.image_handle = self.ax.imshow(image)
            plt.show(block=False)
            self.fig.canvas.draw()
        else:
            self.image_handle.set_data(image)
            #self.ax.draw_artist(self.image_handle)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def write_to_video(self, f, framerate):
        H_single, W_single = self.sensor_size
        H = H_single * len(self.data[0])
        W = W_single * len(self.data)

        writer = skvideo.io.FFmpegWriter(f, inputdict={'-width': str(W), '-height': str(H)},
                                         outputdict={'-r': str(framerate), '-vcodec': 'libx264', '-b': '30000000000'})

        for i in tqdm.tqdm(range(self.i_max)):
            t = self.t_range[0] + float(i) / self.fps
            images = [self.render(d, t) for d in self.data]
            image = np.concatenate(images, 1)
            writer.writeFrame(image)

        writer.close()


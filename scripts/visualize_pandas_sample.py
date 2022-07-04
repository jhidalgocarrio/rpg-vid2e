"""Will create visualization of sample"""


import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import tqdm


def FLAGS():
    parser = argparse.ArgumentParser("Create Visualization of Sample")
    # rostopics

    parser.add_argument("--dataset_root", help="Folder where data is.", default="/home/dani/code/catkin_ws/src/rpg_vid2edataset/garfield" )
    parser.add_argument("--label", help="Class which you want to visualize.", default="")
    parser.add_argument("--idx", help="Folder where data is.", type=int, default=1 )
    parser.add_argument("--ignore", help="Which tags to ignore.", nargs="+", default=["renderings"] )
    parser.add_argument("--video_path", default="" )

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root
    assert os.path.isdir(os.path.join(args.dataset_root, args.label)), "%s should be valid dir." % os.path.join(args.dataset_root, args.label)
    if args.video_path != "":
        assert os.path.isdir(os.path.dirname(args.video_path)), "%s should be valid dir." % os.path.dirname(args.video_path)

    return parser.parse_args()


def to_file(root, counter, type):
    return os.path.join(root, "%s_%s.pkl" % (type, str(counter).zfill(4)))


class Visualizer:
    def __init__(self, data):
        self.data = data
        self.buffers = {}
        self.dt, self.t_range, self.sensor_size = self.get_range_and_increment(data)
        self.plots = {}

    def get_range_and_increment(self, data):
        dt = []
        tmin = []
        tmax = []
        sensor_size = None
        for k, d in data.items():
            times = d["time"]
            if "image" in d.keys():
                dt.append(times.diff().min())
                images = d["image"]
                sensor_size = images.iloc[0].shape[:2]

            tmin.append(times.min())
            tmax.append(times.max())

        return np.min(dt), [np.min(tmin), np.max(tmax)], sensor_size

    def render(self, i):
        t_first = self.t_range[0] + i * self.dt
        t_last = self.t_range[0] + i * self.dt + self.dt
        # extract data between times
        data_between_time = {}
        for k, v in self.data.items():
            data_between_time[k] = v.loc[(v["time"] < t_last) & (v["time"] >= t_first)]

        rendered = {}
        for k, v in data_between_time.items():
            if len(v) == 0:
                rendered[k] = None
                continue

            if "image" in v.keys():
                rendered[k] = v["image"].iloc[0]
            else:
                rendered[k] = self.event_image(v.as_matrix(["x", "y", "polarity"]))

        return rendered

    def event_image(self, events):
        """x,y,p"""
        H, W = self.sensor_size
        lin_idx = (events[:,0] + events[:,1]*W).astype(np.int64)
        img = np.zeros(H*W, dtype=np.float)

        np.add.at(img, lin_idx, events[:,-1])

        img = img.reshape((H, W))

        # convert to uint8
        img = np.clip(255*(img + 5)/10, 0, 255).astype(np.uint8)
        return img

    def update_plots(self, i, rendered):
        if len(self.plots) == 0:
            self.fig, self.ax = plt.subplots(ncols=len(rendered))

        t_first = self.t_range[0] + i * self.dt
        t_last = self.t_range[0] + i * self.dt + self.dt
        self.fig.suptitle("t_first = %.2f\n"
                          "t_last  = %.2f\n"
                          "t_0     = %.2f\n"
                          "t_1     = %.2f\n" % (self.t_range[0], self.t_range[1], t_first, t_last) )

        for i, k in enumerate(sorted(rendered.keys())):
            rendering = rendered[k]

            if rendering is None:
                continue

            if k not in self.plots:
                handle = self.ax if len(rendered) == 1 else self.ax[i]
                self.plots[k] = handle.imshow(rendering)
                handle.set_title(k)
                plt.show(block=False)
            else:
                self.plots[k].set_data(rendering)

            self.fig.canvas.draw()
            plt.pause(0.02)

    def run(self):
        i_max = np.ceil((self.t_range[1] - self.t_range[0])/self.dt).astype(int)
        i = 0
        while True:
            if i in self.buffers:
                images = self.buffers[i]
            else:
                images = self.render(i)
                self.buffers[i] = images

            self.update_plots(i, images)
            i = (i + 1) % i_max

    def write_to_video(self, f, codec="DIVX", frame_rate=10):
        i_max = np.ceil((self.t_range[1] - self.t_range[0])/self.dt).astype(int)
        vid = None
        for i in tqdm.tqdm(range(i_max)):
            images = self.render(i)

            if vid is None:
                vid = cv2.VideoWriter(f, cv2.VideoWriter_fourcc(*codec), frame_rate,
                                (len(images) * self.sensor_size[1], self.sensor_size[0]))

            images_color = {}
            for k, v in images.items():
                if v is None:
                    v = np.zeros(self.sensor_size, dtype=np.uint8)
                if len(v.shape) == 2:
                    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
                images_color[k] = v
            images_color = np.concatenate([images_color[k] for k in sorted(images.keys())], 1)
            vid.write(images_color)
        vid.release()


def collect_data_to_read(root, idx):
    idx = str(idx).zfill(4)
    files = glob.glob(os.path.join(root, "*_%s.*" % idx))
    data_to_read = []
    for f in files:
        data_to_read.append(os.path.basename(f).split(".")[-2].replace("_%s" % idx, ""))
    return data_to_read


if __name__ == '__main__':
    flags = FLAGS()

    label_folder = os.path.join(flags.dataset_root, flags.label)

    data_to_read = collect_data_to_read(label_folder, flags.idx)

    print("Will visualize the following data: %s" % data_to_read)
    data_found = {}

    for sample in data_to_read:
        if sample in flags.ignore:
            continue

        f = to_file(label_folder, flags.idx, sample)
        if os.path.isfile(f):
            data_found[sample] = pd.read_pickle(f)

    # visualize data
    viz = Visualizer(data_found)

    if flags.video_path != "":
        viz.write_to_video(flags.video_path)

    try:
        viz.run()
    except KeyboardInterrupt:
        exit(1)
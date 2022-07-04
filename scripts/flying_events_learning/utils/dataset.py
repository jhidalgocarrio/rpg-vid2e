import numpy as np
from os import listdir
from os.path import join
import yaml
import glob
import os
import torch
import torch.nn.functional as F
from utils.esim_py import ESIM
import cv2

def fill_events(events, dimensions, values):
    C, H, W = dimensions
    img = np.zeros((H*W*C,), dtype=np.float32)
    x,y,t,p = events.T

    t_int = t.astype(np.int64)
    r = t-t_int

    lin_idx1 = (x + W * y + W * H * t_int).astype(np.int64)
    lin_idx2 = (x + W * y + W * H * (t_int+1)).astype(np.int64)

    in_bound = (t_int+1 <= C-1)

    np.add.at(img, lin_idx1, (1-r)*values)
    np.add.at(img, lin_idx2[in_bound], (r[in_bound])*values[in_bound])

    img = img.reshape((C, H, W))
    return img

def voxel_grid(events, channels, resolution):
    (H, W), C = resolution, channels

    if len(events) > 1:
        t = (events[:,2]-events[0,2])/(events[-1,2]-events[0,2])
        events[:,2] = t
        events[:,2] *= (C-1)

        img_neg = fill_events(events[events[:,-1]==-1], [C, H, W], t[events[:,-1]==-1])
        img_pos = fill_events(events[events[:,-1]==1], [C, H, W], t[events[:,-1]==1])
    
        vox = np.concatenate([img_neg, img_pos], 0)
    else:
        vox = np.zeros((2*C, H, W), dtype=np.float32)

    return vox



def random_shift_events(events, shifts=(20,20), resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = shifts
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, p=0.5, resolution=(180, 240)):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

def load_split_files(split_folder, split, classes):
    file = "%s.txt" % split
    with open(join(split_folder, file), "r") as f:
        split_mapping = yaml.load(f)
        mapping = {}
        for k, v in split_mapping.items():
            vs = []
            for vi in v:
                if type(vi) is str:
                    vi = vi.replace(".jpg", "")
                else:
                    vi = ("%s" % vi).zfill(4)
                vs.append(vi)
            mapping[k.lower()] = vs
        split_mapping = mapping

    file_counters = []
    labels = []
    for label in sorted(classes):
        mapping = split_mapping[label.replace("_easy", "")]
        file_counters += mapping
        labels += len(mapping) * [classes.index(label)]

    return file_counters, labels

def add_noise(events, noise_level):
    num_noisy_events = int(noise_level * len(events))
    minx, maxx = np.min(events[:,0]), np.max(events[:,0])
    miny, maxy = np.min(events[:,1]), np.max(events[:,1])
    mint, maxt = np.min(events[:,2]), np.max(events[:,2])

    t = (maxt-mint) * np.random.random((num_noisy_events, 1)) + mint
    x = np.random.randint(minx, maxx+1, size=(num_noisy_events, 1))
    y = np.random.randint(miny, maxy+1, size=(num_noisy_events, 1))
    p = np.random.random((num_noisy_events, 1)) > .5

    fake_events = np.concatenate([x,y,t,p],1).astype(np.float32)

    events = np.concatenate([events, fake_events], 0)
    events = events[events[:,2].argsort()]

    return events

def random_flip(vox, flip):
    if flip:
        return np.ascontiguousarray(vox[...,::-1])
    else:
        return vox

def random_crop_resize(vox, shifts):
    max_shift=20
    x_shift, y_shift = shifts

    to_2 = False
    if len(vox.shape) == 2:
        to_2 = True
        vox = vox[None,...]

    C, H, W = vox.shape
    vox_padded = np.pad(vox, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)), mode="constant",
                        constant_values=0)
    cx, cy = W//2, H//2
    hw = cy
    vox_cropped_torch = torch.from_numpy(vox_padded[:, max_shift+cy-hw-y_shift:max_shift+cy+hw-y_shift,
                                         max_shift+cx-hw-x_shift:max_shift+cx+hw-x_shift])
    vox_cropped_resized = F.interpolate(vox_cropped_torch[None,...], size=(224,224))[0].numpy()

    if to_2:
        vox_cropped_resized = vox_cropped_resized[0]

    return vox_cropped_resized

def normalize_vox(vox):
    eps = 1e-5
    nonzero = np.abs(vox)>0
    mean = np.mean(vox[nonzero])
    std = np.std(vox[nonzero])
    vox[nonzero] = (vox[nonzero] - mean) / (std+eps)
    return vox

def create_split(root, dest_root, split, counters, labels, classes):
    split_folder = os.path.join(dest_root, split)
    if not os.path.exists(split_folder):
        os.system("mkdir -p %s" % split_folder)

    for label, c in zip(labels, counters):
        fc = str(c).zfill(4)

        cls = classes[label]

        label_folder = os.path.join(split_folder, cls)
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)

        # find all files which loo like that
        pattern = join(root, cls, "*%s*" % fc)
        files = glob.glob(pattern)
        src_file = files[0]
        file_name = os.path.basename(src_file)
        dest_file = os.path.join(label_folder, file_name)

        print("cp %s %s" % (src_file, dest_file))
        #os.system("cp %s %s" % (src_file, dest_file))

def split_dataset(split_folder, dataset_root, destination_root):
    classes = sorted(os.listdir(dataset_root))
    train_file_counters, train_labels = load_split_files(split_folder, "train", classes)
    valid_file_counters, valid_labels = load_split_files(split_folder, "valid", classes)
    test_file_counters, test_labels = load_split_files(split_folder, "test", classes)

    create_split(dataset_root, destination_root, "train", train_file_counters, train_labels, classes)
    create_split(dataset_root, destination_root, "valid", valid_file_counters, valid_labels, classes)
    create_split(dataset_root, destination_root, "test", test_file_counters, test_labels, classes)


class NCaltech101Base:
    def __init__(self, root,
                 split_folder,
                 split="train",
                 augmentation=False,
                 num_classes=-1,
                 noise_level=0,
                 return_path=False):
        self.root = root
        classes = sorted(listdir(root))
        if num_classes > 0:
            classes = classes[:num_classes]
        self.classes = [c.lower() for c in classes]
        self.split_folder = split_folder

        self.augmentation = augmentation
        self.noise_level = noise_level
        self.return_path = return_path
        self.file_counters, self.labels = load_split_files(split_folder, split, self.classes)
        self.split = split
        print("Dataset at %s : len %s" % (root, len(self)))
        print("Classes: %s" % self.classes)

    def __len__(self):
        return len(self.file_counters)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        fc = str(self.file_counters[idx]).zfill(4)

        # find all files which loo like that
        pattern = join(self.root, self.classes[label], "*%s*" % fc)
        files = glob.glob(pattern)
        assert len(files) > 0, "Did not find anything with pattern: '%s'." % pattern

        output = {"label": label, "files": files}

        return output, fc


class NCaltech101VoxelGrid(NCaltech101Base):
    def __getitem__(self, idx):
        output, fc = NCaltech101Base.__getitem__(self, idx)

        for f in output["files"]:
            if self.return_path:
                output["path"] = f

            if self.augmentation:
                max_shift = 20
                p = np.random.random()>.5
                shifts = np.random.randint(-max_shift, max_shift + 1, size=(2,))

            vox = np.load(f)
            key = "events"
            if self.augmentation:
                vox = random_flip(vox, p)
                vox = random_crop_resize(vox, shifts)
            else:
                vox = random_crop_resize(vox, [0,0])

            output[key] = vox

        return output


class NCaltech101Events(NCaltech101Base):
    def __getitem__(self, idx):
        output, fc = NCaltech101Base.__getitem__(self, idx)
        have_events = False
        for f in output["files"]:
            if self.return_path:
                output["path"] = f

            key = os.path.basename(f).replace("_"+fc, "").replace("label", "image")

            if self.augmentation:
                max_shift = 20
                p = np.random.random() > .5
                shifts = np.random.randint(-max_shift, max_shift + 1, size=(2,))

            if key.endswith(".npy") and not have_events:
                events = np.load(f)
                events[:, -1] = 2 * events[:, -1] - 1
                if len(events) > 0:
                    events[:, 2] = (events[:, 2] - events[0, 2]) / ((events[-1, 2] - events[0, 2]))
            elif key.endswith(".npz") and not have_events:
                events = np.load(f)
                events = np.concatenate([events["xyp"], events["t"]], -1)
                events = events[:,[0,1,3,2]]
                events[:,-1] = 2*events[:,-1]-1

                if len(events) > 0:
                    events[:,2] = (events[:,2] - events[0,2]) / ((events[-1,2] - events[0,2]) )
            else:
                continue

            if self.augmentation:
                events = random_shift_events(events, shifts)
                events = random_flip_events_along_x(events, p)

                if self.noise_level > 0:
                    events = add_noise(events, self.noise_level)

            output[key] = events
            have_events = True

        return output

class MergedNCaltech101Events:
    def __init__(self, roots,
                 split_folder,
                 split="train",
                 augmentation=False,
                 num_classes=-1,
                 noise_level=0,
                 return_path=False):

        self.datasets = [NCaltech101Events(root, split_folder, split, augmentation, num_classes, noise_level, return_path)
                            for root in roots]

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        for j, d in enumerate(self.datasets):
            if idx >= len(d):
                idx -= len(d)
                continue
            break
        else:
            raise ValueError("Idx too high for len %s" % len(self))

        return self.datasets[j][idx]


class NCaltech101ImageFolder(NCaltech101Base):
    def __getitem__(self, idx):
        output, fc = NCaltech101Base.__getitem__(self, idx)

        for f in output["files"]:
            if self.return_path:
                output["path"] = f

            key = os.path.basename(f).replace(fc, "").rstrip("_")

            if self.augmentation:
                max_shift = 20
                p = np.random.random() > .5
                shifts = np.random.randint(-max_shift, max_shift + 1, size=(2,))
                Cp, Cn = .05+.25*np.random.random((2,))
            else:
                shifts = [0, 0]
                p = False
                Cp, Cn = 0.06, 0.06
            
            try:
                events, resolution = self.generate_virtual_events(f, shifts, p, Cp, Cn)
            except Exception:
                print("Error with generating events with Cp=%s Cn=%s path=%s" % (Cp, Cn, f))

            events[:, -1] = 2 * events[:, -1] - 1

            vox = voxel_grid(events, 15, resolution)

            output["events"] = vox

        return output

    def load_frame(self, f, shifts, p):
        frame0 = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY).astype(np.float32)
        frame0 = random_crop_resize(frame0, shifts)
        frame0 = random_flip(frame0, p)
        frame0 = frame0/255
        return frame0

    def generate_virtual_events(self, f, shifts, p, Cp, Cn):
        esim = ESIM(Cp, Cn, 0)

        with open(f + "/images.csv", "r") as fh:
            lines = fh.readlines()

        timestamps = np.array([float(l.split(" ")[0])*1e-9 for l in lines])

        events = np.zeros((0, 4), dtype=np.float32)

        frame_path = os.path.join(f, "frame", "cam0", lines[0].split(" ")[1][:-1])
        frame0 = self.load_frame(frame_path, shifts, p)

        reference_values = frame0
        t0 = timestamps[0]

        for i in range(1,len(timestamps)):
            t = timestamps[i]
            frame_path = os.path.join(f, "frame", "cam0", lines[i].split(" ")[1][:-1])
            frame = self.load_frame(frame_path, shifts, p)

            new_events = esim.generate_events1(frame, frame0, t, t0, reference_values)

            events = np.concatenate([events, new_events], 0)
            t0, frame0 = t, frame

        events = events[events[:, 2].argsort()]

        return events, frame0.shape

if __name__ == '__main__':
    root = "/media/dani/data/caltech_image_folders/dataset"

    dataset = NCaltech101ImageFolder(root,
                                     split_folder="/media/dani/data/ncaltech_experiments/split_train_0.5_val_0.25_test_0.25",
                                     augmentation=True)

    dataset[0]

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
import tqdm


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

def random_flip_events_and_pose(events, pose, H):
    if np.random.random() > .5:
        # perform horizontal flip
        events[:, 1] = H - 1 - events[:, 1]
        pose[:, 1] = H - 1 - pose[:, 1]

        # switch left and right
        left_indices = [1,3,5,7,9,11]
        right_indices = [2,4,6,8,10,12]

        pose[left_indices], pose[right_indices] = pose[right_indices], pose[left_indices]

    if np.random.random() > .5:
        # perform vertical flip
        events[:, 0] = H - 1 - events[:, 0]
        pose[:, 0] = H - 1 - pose[:, 0]

        # switch left and right
        left_indices = [1, 3, 5, 7, 9, 11]
        right_indices = [2, 4, 6, 8, 10, 12]

        pose[left_indices], pose[right_indices] = pose[right_indices], pose[left_indices]

    return events, pose

def random_rotate_events_and_pose(events, pose, H):
    """
    :param events: N x 4
    :param pose: N x 2
    :return:
    """
    root_pos = np.mean(pose, axis=0)[None, :]

    events[:,:2] -= root_pos
    pose -= root_pos

    max_angle = np.pi/3
    angle = (2*np.random.random()-1) * max_angle

    def rotate_by_angle(x, a):
        x[:, 0], x[:, 1] = \
            np.cos(a) * x[:, 0] - np.sin(a) * x[:, 1], \
            np.sin(a) * x[:, 0] + np.cos(a) * x[:, 1]
        return x

    events[:,:2] = rotate_by_angle(events[:,:2], angle)
    pose = rotate_by_angle(pose, angle)

    pose += root_pos
    events[:,:2] += root_pos

    events[:,:2] += np.random.random(events[:,:2].shape)-.5
    events[:,:2] = events[:,:2].astype(int)

    # discard events that lie outside of fov
    valid = (events[:,0]>=0) & (events[:,1]>=0) & (events[:,0]<H) & (events[:,1]<H)
    events = events[valid]

    return events, pose


class HumanPoseDataset:
    poses = None
    body_points = [
        'head_top', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hand', 'right_hand',
        'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_foot', 'right_foot'
    ]
    point_lines = [
        ['head_top', 'right_shoulder'], ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_hand'],
        ['head_top', 'left_shoulder'], ["left_shoulder", "left_elbow"], ["left_elbow", "left_hand"],
        ["right_shoulder", "left_shoulder"], ["right_shoulder", 'right_hip'], ["left_shoulder", 'left_hip'],
        ['left_hip', 'right_hip'],
        ['left_shoulder', 'right_hip'], ['left_hip', 'right_shoulder'],
        ['right_hip', 'right_knee'], ['right_knee', 'right_foot'],
        ['left_hip', 'left_knee'], ['left_knee', 'left_foot']
    ]
    all_body_parts = [
        'spine3', 'spine4', 'spine2', 'spine', 'pelvis', 'neck', 'head', 'head_top', 'left_clavicle',
        'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand', 'right_clavicle', 'right_shoulder',
        'right_elbow', 'right_wrist', 'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot',
        'left_toe', 'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe'
    ]

    valid_indices = None

    def __init__(self, root,
                 split,
                 augmentation=False,
                 number_events=7500,
                 step=None,
                 drop_events_rate=0,
                 debug=False):

        self.step = step
        self.drop_events_rate = drop_events_rate
        self.number_events = number_events
        self.root = root
        self.split = split
        self.augmentation = augmentation
        self.debug = debug

        event_files, index_files, vicon_files, meta_data = self.load_files(root, split)

        assert len(event_files) > 0
        assert len(index_files) > 0
        assert len(vicon_files) > 0

        self.meta_data = meta_data

        print("Loading vicon data...")
        self.vicon_data = []
        for k in tqdm.tqdm(vicon_files):
            if k is not None:
                self.vicon_data += [(k, np.load(k).astype(np.float32))]
            else:
                self.vicon_data += [(None, None)]

        print("Preparing event handles...")
        self.events = []
        self.indices = []
        self.key = []
        self.tot_num_events = 0
        for events, index_file in tqdm.tqdm(zip(event_files, index_files)):
            # hack to find number of events, because each entry has 8 bytes and there are 4 columns so 8*4=32
            num_events = int(os.stat(events).st_size / 32)
            self.events += [(events, np.memmap(events, dtype="int64", mode="r", shape=(num_events, 4)))]
            self.tot_num_events += self.events[-1][1].shape[0]
            self.tmax = self.events[-1][1][-1,2]

            if index_file is not None:
                index = np.load(index_file).astype(np.int64)
                # remove entries in index that do not have enough events yet
                index = index[index[:,1]>self.number_events]

                self.indices += [(index_file, index)]

                if len(self.key) == 0:
                    self.key += [len(index)]
                else:
                    self.key += [len(index)+ self.key[-1]]
            else:
                self.indices += [(None, None)]

        if HumanPoseDataset.valid_indices is None:
            HumanPoseDataset.valid_indices = [HumanPoseDataset.all_body_parts.index(bp) for bp in HumanPoseDataset.body_points]

        self.just_events = len(self.key) == 0

    def __len__(self):
        if self.just_events:
            return self.tmax // int(1e9*self.step)
        return self.key[-1]

    def load_files(self, root, split):
        raise NotImplementedError

    def compute_events(self, handle, idx, number_events):
        lower_idx  = max([0, idx-number_events])
        upper_idx  = min([handle.shape[0], idx])
        events = np.array(handle[lower_idx:upper_idx])
        t = events[:, 2]
        t -= t[0]
        t = t.astype("float32")
        t = t/t[-1]
        events = events.astype(np.float32)
        events[:, -1] = 2 * events[:, -1] - 1
        events[:, 2] = t
        return events

    def compute_pose(self, data, t_ns, scale):
        vicon_idx = np.searchsorted(data[:, 0], t_ns * 1e-9) - 1
        vicon_idx = max([0, vicon_idx])

        pose_x = data[vicon_idx, 1::2]
        pose_y = data[vicon_idx, 2::2]
        pose = np.stack([pose_x, pose_y], -1)

        # from mpi dataset
        if pose.shape[0] > 13:
            pose = pose[HumanPoseDataset.valid_indices, :]
            pose *= scale

        return pose

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        H, W = 346, 346

        scale = H / 2048

        # find index of handle in which current events are stored
        offset = np.searchsorted(self.key, idx+.001)

        _, vicon_data = self.vicon_data[offset]
        _, events_handle = self.events[offset]
        _, index = self.indices[offset]

        # find the sample idx within the current batch of data
        if not self.just_events:
            sample_idx = idx - self.key[offset-1] if offset > 0 else idx

            # get the events that ocurred at sample idx, minus N events
            timestamp, event_idx = index[sample_idx]

            pose = self.compute_pose(vicon_data, timestamp, scale)
            head_size = np.sum((pose[HumanPoseDataset.all_body_parts.index("head_top")]
                                - pose[HumanPoseDataset.all_body_parts.index("neck")]) ** 2) ** .5
        else:
            event_time = self.step*(idx+1+100)*1e9
            event_idx = np.searchsorted(events_handle[:,2], event_time)

        # randomly discard events
        number_events = self.number_events
        if self.augmentation:
            number_events = int(np.ceil(number_events*(1-np.random.random()*self.drop_events_rate)))

        events = self.compute_events(events_handle, event_idx, number_events)
        px, py = pose.T
        mask = (px >=0) & (px < W) & (py >=0) & (py < H)
        pose = np.concatenate([pose, mask[:,None]], -1)

        data = {"events": events}

        if not self.just_events:
            data = {"events": events, "poses": pose, "headsize": head_size, "timestamp": timestamp}

        if self.meta_data is not None:
            data.update(**self.meta_data[offset])

        if self.debug:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            mpl.use("TkAgg")
            img = np.zeros((H*W,))
            np.add.at(img, (events[:,0]+W*events[:,1]).astype("int64"), events[:,-1])
            img = img.reshape((H,W))
            plt.imshow(img)
            plt.scatter(*pose.T)
            plt.show()

        return data


class DHP19(HumanPoseDataset):
    def load_files(self, root, split):
        memmap_files = sorted(glob.glob(os.path.join(root, "processed", "h5_dataset_7500_events", "346x260", "*.dat")))
        index_files = sorted(glob.glob(os.path.join(root, "processed", "h5_dataset_7500_events", "346x260", "*.npy")))
        vicon_files = sorted(glob.glob(os.path.join(root, "data", "Vicon_data", "*.npy")))

        idx = int(len(memmap_files) * .8)
        if split == "train":
            memmap_files = memmap_files[:idx]
            index_files = index_files[:idx]
            vicon_files = vicon_files[:idx]
        else:
            memmap_files = memmap_files[idx:]
            index_files = index_files[idx:]
            vicon_files = vicon_files[idx:]

        meta_data = [self.to_meta_data(f) for f in vicon_files]

        return memmap_files, index_files, vicon_files, meta_data

    @staticmethod
    def to_meta_data(f):
        # vicon files have form 'S9_5_7_4.npy'
        f = os.path.basename(f)
        subject, sequence, movement, view = f[1:-4].split("_")

        return {
            "view": view,
            "movement": movement,
            "sequence": sequence,
            "subject": subject
        }


class MPI(HumanPoseDataset):
    def load_files(self, root, split):
        memmap_files = sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "events.dat")))
        index_files = sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "index.npy")))
        vicon_files_temp = sorted(glob.glob(os.path.join(root, "*", "*", "*", "vicon_*")))

        vicon_files = []
        for f in memmap_files:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(f)))
            for vf in vicon_files_temp:
                vicon_idx = vf[-5]
                if vf.startswith(root_dir) and vicon_idx in os.path.basename(os.path.dirname(f)):
                    vicon_files.append(vf)
                    break
        
        idx = int(len(memmap_files)*0.9)
        if split == "train":
            memmap_files = memmap_files[:idx]
            index_files = index_files[:idx]
            vicon_files = vicon_files[:idx]
        else:
            memmap_files = memmap_files[idx:]
            index_files = index_files[idx:]
            vicon_files = vicon_files[idx:]

        meta_data = [self.to_meta_data(f) for f in vicon_files]

        return memmap_files, index_files, vicon_files, meta_data

    @staticmethod
    def to_meta_data(f):
        # vicon file has form
        # data/3dwild/mpi_inf_3dhp/S1/Seq1/imageSequence/vicon_0.npy
        view = f[-5]
        sequence = os.path.basename(os.path.dirname(os.path.dirname(f)))[-1]
        subject = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))[-1]

        return {
            "view": view,
            "movement": 0,  # no info so far of movement
            "session": sequence,
            "subject": subject
        }


class SingleSequence(HumanPoseDataset):
    def load_files(self, event_file, gt_file):
        vicon_files = [gt_file]
        event_files = [event_file]
        index_files = [event_file[:-4] + "_index.npy"]

        if not os.path.exists(index_files[0]):
            index_files[0] = os.path.join(os.path.dirname(event_file), "index.npy")

        return event_files, index_files, vicon_files

class JustEvents(HumanPoseDataset):
    def load_files(self, event_file, gt_file):
        return [event_file], [None], [None], None

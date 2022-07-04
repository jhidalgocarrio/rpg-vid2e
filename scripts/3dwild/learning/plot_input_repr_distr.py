import argparse
import torch
import torchvision
import os
import numpy as np
import tqdm

from utils.models import HumanPoseNet, pose_resnet
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import PoseLoss
from utils.dataset import HumanPoseDataset
import skvideo.io

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from visualize_pose import generate_event_image, draw_poses

torch.manual_seed(1)
np.random.seed(1)


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--event_file", default="")
    parser.add_argument("--gt_file", default="")
    parser.add_argument("--root", default="" )

    parser.add_argument("--checkpoint", default="", required=True)
    parser.add_argument("--video_path", default="", required=True)
    parser.add_argument("--nframes", type=int, default=-1)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_joints", type=int, default=13)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=-1)
    parser.add_argument("--draw_gt", action="store_true", default=False)
    parser.add_argument("--output_dir", default="")

    flags = parser.parse_args()

    return flags


def robust_min_max_norm1(img):
    robust_max_vals = percentile(img, .99)
    robust_min_vals = percentile(img, .01)

    img = (img - robust_min_vals) / (robust_max_vals - robust_min_vals)
    img = torch.clamp(img, 0, 1)

    return img

def robust_min_max_norm(img, ret=False, rob=None):
    if rob is None:
        robust_max_vals = percentile(img, .99)
    else:
        robust_max_vals = rob

    img = img / robust_max_vals

    img = torch.clamp(img, 0, 1)

    if ret:
        return img, robust_max_vals

    return img


def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(q * (C * H * W - 1))
    result = t.reshape(B, -1).kthvalue(k).values
    return result[:, None, None, None]


def create_image(representation):
    representation = representation.sum(1, keepdim=True)

    # do robust min max norm
    representation = representation.detach().cpu()
    representation = robust_min_max_norm(representation)
    representation = torch.clamp(255 * representation, 0, 255).byte()

    representation = torchvision.utils.make_grid(representation)

    return representation


def plot_frame(events, heatmaps_prediction, predicted_joint_positions, gt_joint_positions=None):
    events = events.cpu().numpy()
    heatmaps_prediction = heatmaps_prediction.detach().cpu().numpy()
    predicted_joint_positions = predicted_joint_positions.detach().cpu().numpy()
    gt_joint_positions = gt_joint_positions.cpu().numpy() if gt_joint_positions is not None else gt_joint_positions

    B, C, H, W = heatmaps_prediction.shape

    events[:,-2] = (events[:,-2]+1)/2

    event_image = generate_event_image(events[:,:-1], (H,W))   # H, W, 3

    pose_x, pose_y = predicted_joint_positions[0,:].T
    image = draw_poses(event_image, pose_x, pose_y, minimal=True)

    return image

def single_heatmap(idx, heat_map, mapper, representation, pred_joint_positions, pose, rob):
    heat_map_single = heat_map[:,idx:idx+1,...]
    heat_map_single, rob = robust_min_max_norm(heat_map_single, ret=True, rob=rob)
    heat_map_single = heat_map_single[:, 0, ...].detach().cpu()
    heat_map_color = torch.from_numpy(mapper(heat_map_single)[..., :3]).permute(0, 3, 1, 2).float()  # B x 3 x H x W

    # do robust min max norm
    image = .5 * heat_map_color + .5 * representation  # alpha blending

    B, _, H, W = image.shape

    x, y = pose[0,idx,:2].int()
    xmin, ymin = max([0, x - 4]), max([0, y - 4])
    xmax, ymax = min([W - 1, x + 4]), min([H - 1, y + 4])
    image[0, 0, ymin:ymax, xmin:xmax] = torch.ones((ymax - ymin, xmax - xmin))
    image[0, 1, ymin:ymax, xmin:xmax] = torch.zeros((ymax - ymin, xmax - xmin))
    image[0, 2, ymin:ymax, xmin:xmax] = torch.ones((ymax - ymin, xmax - xmin))

    x, y = pred_joint_positions[0,idx,:].int()
    xmin, ymin = max([0, x - 4]), max([0, y - 4])
    xmax, ymax = min([W - 1, x + 4]), min([H - 1, y + 4])
    image[0, 0, ymin:ymax, xmin:xmax] = torch.ones((ymax - ymin, xmax - xmin))
    image[0, 1, ymin:ymax, xmin:xmax] = torch.zeros((ymax - ymin, xmax - xmin))
    image[0, 2, ymin:ymax, xmin:xmax] = torch.zeros((ymax - ymin, xmax - xmin))

    # generate per joint images
    return image, rob

def plot_heatmap(heat_map, pose, representation, pred_joint_positions, target_heatmap, ret=False, rob=None):
    import matplotlib.cm as cm
    mapper = cm.get_cmap("jet")

    heat_map[heat_map<1e-4] = 0

    representation = representation.sum(1, keepdim=True)
    representation = representation.detach().cpu()
    representation = robust_min_max_norm(representation)
    representation_color = representation.repeat(1, 3, 1, 1)

    images = []
    for idx in range(13):
        image, rob = single_heatmap(idx, heat_map, mapper, representation_color, pred_joint_positions, pose, rob)
        images.append(image)

    B, C, H, W = images[0].shape
    img = torch.zeros(1, C, 3*H, 5*W)

    for i in range(13):
        row = i // 5
        col = i % 5
        img[:,:,H*row:H*row+H, H*col:H*col+H] = images[i]

    images = torch.clamp(255 * img, 0, 255).byte()
    images = torchvision.utils.make_grid(images[:8])

    if ret:
        return images, rob

    return images


class Writer:
    def __init__(self, root):
        if not os.path.isdir(root):
            os.makedirs(root)

        self.curr_writers = None
        self.root = root

    def update(self, data, err, confidence, pred_joint_pos):
        sequence = data["sequence"]
        movement = data["movement"]
        subject = data["subject"]
        view = data["view"]

        for i in range(err.shape[0]):
            target_dir = os.path.join(self.root, "view_%s" % view[i], "movement_%s" % movement[i], "subject_%s" % subject[i], "sequence_%s" % sequence[i])
            self.log_single(target_dir, data["timestamp"][i], data["poses"][i], pred_joint_pos[i], err[i], confidence[i])

    def log_single(self, dir, t, target_pose, pred_pose, error, confidence):
        pred_dir = os.path.join(dir, "predicted")
        target_dir = os.path.join(dir, "target")

        if not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)
            os.makedirs(target_dir)

            if self.curr_writers is not None:
                self.curr_writers[0].close()
                self.curr_writers[1].close()

            self.curr_writers = (SummaryWriter(pred_dir), SummaryWriter(target_dir))

        average = (error*target_pose[:,-1]).sum() / (1e-3+target_pose[:,-1].sum())
        self.curr_writers[0].add_scalar("joint_average_error", average, t)
        for j in range(pred_pose.shape[0]):
            self.curr_writers[0].add_scalar("confidence_joint_%s" % j, confidence[j], t)
            for writer, pose in zip(self.curr_writers, [pred_pose, target_pose]):
                writer.add_scalar("position_joint_%s_x" % j, pose[j,0], t)
                writer.add_scalar("position_joint_%s_y" % j, pose[j,1], t)
                writer.add_scalar("error_joint_%s" % j, error[j], t)


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    flags = FLAGS()
    # datasets, add augmentation to training set
    if flags.root == "":
        dataset = HumanPoseDataset(flags.event_file, flags.gt_file, single=True)
    else:
        dataset = HumanPoseDataset(flags.root, "all", real=False)

    validation_loader = Loader(dataset, flags, sequential=False)

    # model, and put to device
    model = pose_resnet(batch_size=flags.batch_size,
                        num_classes=flags.num_joints,
                        resnet_layers=34)

    model = model.to(flags.device)

    # save flags to log dir
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])

    curr_pred_joint_positions = None
    event_key = None

    model = model.eval()
    rob = None
    sum_loss = 0
    data_list = []
    for j, data in enumerate(tqdm.tqdm(validation_loader)):
        # find key which contains events
        if event_key is None:
            for k in data:
                if "event" in k:
                    event_key = k
                    break

        with torch.no_grad():
            pred = model(data[event_key], batchsize=len(data["poses"]), only_repr=True)

        data_list.extend(pred[pred.abs()>0].view(-1).cpu().numpy().tolist())

        if j > 1000:
            break

    plt.hist(data_list, range=(0,3), bins=1000, normed=True)
    plt.show()
    print("Loss: ", sum_loss/j)
    #writer.close()
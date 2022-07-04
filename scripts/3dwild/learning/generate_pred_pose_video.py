import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm
import torch.nn.functional as F

from utils.models import HumanPoseNet, pose_resnet
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import PoseLoss
from utils.dataset import HumanPoseDataset, JustEvents
import skvideo.io
from plotting import plot_heatmap

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
    parser.add_argument("--dataset", default="" )

    parser.add_argument("--checkpoint", default="", required=True)
    parser.add_argument("--video_path", default="", required=True)
    parser.add_argument("--nframes", type=int, default=-1)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_joints", type=int, default=13)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=-1)
    parser.add_argument("--draw_gt", action="store_true", default=False)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--step", type=int, default=0.01)
    parser.add_argument("--number_events", type=int, default=7500)

    flags = parser.parse_args()

    return flags




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
    flags = FLAGS()

    writer = skvideo.io.FFmpegWriter(flags.video_path, inputdict={'-width': str(5*346), '-height': str(3*346)},
                                     outputdict={'-r': str(30), '-vcodec': 'libx264',
                                                 '-b': '30000000000'})

    #writer = Writer("/tmp/log1")

    # datasets, add augmentation to training set
    try:
        dataset_cls = eval(flags.dataset)
    except Exception:
        raise ValueError(f"flags.dataset {flags.dataset} must point either to 'MPI' or 'DHP19' dataset.")

    dataset = dataset_cls(flags.event_file, flags.gt_file, step=flags.step, number_events=flags.number_events)
    validation_loader = Loader(dataset, flags, sequential=True)

    # model, and put to device
    model = pose_resnet(batch_size=flags.batch_size,
                        num_classes=flags.num_joints,
                        resnet_layers=34)

    loss_func = PoseLoss(return_confidence=True)

    model = model.to(flags.device)

    # save flags to log dir
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])

    curr_pred_joint_positions = None
    event_key = None

    model = model.eval()
    rob = None
    rob_repr = None
    sum_loss = 0
    for j, data in enumerate(tqdm.tqdm(validation_loader)):
        # find key which contains events
        if event_key is None:
            for k in data:
                if "event" in k:
                    event_key = k
                    break

        with torch.no_grad():
            pred, representation, _ = model(data[event_key], batchsize=flags.batch_size)
            poses = data.get("poses", None)
            headsize = data.get("headsize", None)
            loss, per_joint_loss, pred_joint_positions, target_heatmap, pCKh50, position_error_2d, confidence = loss_func(pred, poses, headsize)

        #print(confidence.max())
        # only update pose if confidence is > threshold
        if flags.confidence_threshold > 0:

            if curr_pred_joint_positions is None:
                curr_pred_joint_positions = -1*torch.ones_like(pred_joint_positions)

            mask = confidence > flags.confidence_threshold
            curr_pred_joint_positions[mask] = pred_joint_positions[mask]
        else:
            curr_pred_joint_positions = pred_joint_positions


        #writer.update(data, position_error_2d, confidence, pred_joint_positions)
        #frame = plot_frame(events=data[event_key],
        #                   heatmaps_prediction=pred,
        #                   predicted_joint_positions=curr_pred_joint_positions,
        #                   gt_joint_positions=data["poses"] if flags.draw_gt else None)
        frame, rob, rob_repr = plot_heatmap(pred, poses, representation, pred_joint_positions, ret=True, rob=rob, rob_repr=rob_repr)
        frame = frame.permute(1,2,0).numpy()
        writer.writeFrame(frame)

        if per_joint_loss is not None:
            sum_loss += per_joint_loss.cpu().item()

        #print("Loss: ", loss.cpu().item())

        if j == flags.nframes:
            break

    print("Loss: ", sum_loss/j)
    writer.close()
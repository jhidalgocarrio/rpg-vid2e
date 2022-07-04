import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm

from utils.models import HumanPoseNet, pose_resnet
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import PoseLoss
from utils.dataset import MPI, DHP19
from utils.pose_hrnet import get_pose_net
from plotting import plot_heatmap, plot_grad_flow

torch.manual_seed(1)
np.random.seed(1)


def valid_step(model, loss_func, validation_loader, flags, iteration):
    sum_loss = 0
    sum_per_joint_loss = 0
    sum_pCKh50 = 0
    model = model.eval()
    event_key = None
    rob_repr = None
    rob = None
    for j, data in enumerate(tqdm.tqdm(validation_loader)):
        # find key which contains events
        if event_key is None:
            for k in data:
                if "event" in k:
                    event_key = k
                    break

        with torch.no_grad():
            pred, representation, _ = model(data[event_key], batchsize=len(data["poses"]))
            loss, per_joint_loss, pred_joint_positions, target_heatmap, pCKh50, _ = loss_func(pred, data["poses"],
                                                                                              data["headsize"])

        if j == 0:
            # visualize representation
            frame, rob, rob_repr = plot_heatmap(pred, data["poses"], representation, pred_joint_positions, ret=True, rob=rob, rob_repr=rob_repr)
            writer.add_image("validation/pose", frame, iteration)

        sum_loss += loss
        sum_per_joint_loss += per_joint_loss
        sum_pCKh50 += pCKh50

        if j == flags.num_valid_steps:
            break

    if flags.num_valid_steps > 0:
        valid_steps = min([flags.num_valid_steps, len(validation_loader)])
    else:
        valid_steps = len(validation_loader)

    validation_loss = sum_loss.item() / valid_steps
    validation_per_joint_loss = sum_per_joint_loss.item() / valid_steps
    validation_pCKh50 = sum_pCKh50.item() / valid_steps

    writer.add_scalar("validation/per_joint_loss", validation_per_joint_loss, iteration)
    writer.add_scalar("validation/loss", validation_loss, iteration)
    writer.add_scalar("validation/pCKh50", validation_pCKh50, iteration)

    print(f"Validation Loss {validation_loss:.4f}")

    return validation_loss

def test_step(model, loss_func, validation_loader, flags):
    sum_loss = 0
    sum_per_joint_loss = 0
    sum_pCKh50 = 0
    event_key = None
    rob = None
    for j, data in enumerate(tqdm.tqdm(real_loader)):
        # find key which contains events
        if event_key is None:
            for k in data:
                if "event" in k:
                    event_key = k
                    break

        with torch.no_grad():
            pred, representation, _ = model(data[event_key], batchsize=len(data["poses"]))
            loss, per_joint_loss, pred_joint_positions, target_heatmap, pCKh50, _ = loss_func(pred, data["poses"],
                                                                                              data["headsize"])

        if j == 0:
            # visualize representation
            frame, robo= plot_heatmap(pred, data["poses"], representation, pred_joint_positions, target_heatmap,ret=True, rob=rob)
            writer.add_image("test/pose", frame, iteration)

        sum_loss += loss
        sum_per_joint_loss += per_joint_loss
        sum_pCKh50 += pCKh50

        if j == flags.num_test_steps:
            break

    if flags.num_test_steps > 0:
        valid_steps = min([flags.num_test_steps, len(validation_loader)])
    else:
        valid_steps = len(validation_loader)

    validation_loss = sum_loss.item() / valid_steps
    validation_per_joint_loss = sum_per_joint_loss.item() / valid_steps
    validation_pCKh50 = sum_pCKh50.item() / valid_steps

    writer.add_scalar("real/per_joint_loss", validation_per_joint_loss, iteration)
    writer.add_scalar("real/loss", validation_loss, iteration)
    writer.add_scalar("real/pCKh50", validation_pCKh50, iteration)

    print(f"Real Loss {validation_loss:.4f}")

def transform_pose(H,P, sensor_size):
    P = P.clone()
    h, w = sensor_size

    scale_square = H[:,:2,:1].pow(2).sum(1, keepdim=True)  # B x 1 x 1
    H_inv = H[:,:2,:2].transpose(1,2) / scale_square

    x, y = P[..., 0], P[..., 1]

    x -= w/2
    y -= h/2

    x, y = x * H_inv[:, 0, 0, None] + y * H_inv[:, 0, 1, None],\
           x * H_inv[:, 1, 0, None] + y * H_inv[:, 1, 1, None]

    x += w/2
    y += h/2
    
    left_indices = [1, 3, 5, 7, 9, 11]
    right_indices = [2, 4, 6, 8, 10, 12]

    # switch left and right
    visible = (x >=0) & (x < w) & (y >= 0) & (y < h)
    visible = visible.float()

    P[..., 0], P[..., 1], P[..., 2] = x, y, visible
    flip = H[:,0,0] < 0
    
    # select poses to flip left and right
    P_flipped = P[flip]

    if len(P_flipped)>0:
        for left_idx, right_idx in zip(left_indices, right_indices):
            temp = P_flipped[:,right_idx].clone()
            P_flipped[:,right_idx] = P_flipped[:,left_idx]
            P_flipped[:,left_idx] = temp
    
    P[flip] = P_flipped

    return P

def train_step(model, loss_func, train_loader, iteration, min_validation_loss):
    sum_per_joint_loss = 0
    sum_loss = 0
    sum_pCKh50 = 0
    event_key = None
    model = model.train()
    rob = None
    rob_repr = None
    for j, data in enumerate(tqdm.tqdm(train_loader)):
        if event_key is None:
            for k in data:
                if "event" in k:
                    event_key = k
                    break

        optimizer.zero_grad()

        pred, representation, homography = model(data[event_key], batchsize=len(data["poses"]), train=True)

        B, C, H, W = pred.shape
        target_poses = transform_pose(homography, data["poses"], [H, W])

        loss, per_joint_loss, pred_joint_positions, target_heatmap, pCKh50, _ = loss_func(pred, target_poses, data["headsize"])

        loss.backward()
        optimizer.step()

        sum_loss += loss
        sum_per_joint_loss += per_joint_loss
        sum_pCKh50 += pCKh50

        iteration += 1
        if iteration % flags.num_train_steps_between_logs == flags.num_train_steps_between_logs - 1:
            log_training(iteration,
                         iteration % flags.num_train_steps_between_logs,
                         sum_loss,
                         sum_per_joint_loss,
                         sum_pCKh50)

            frame, rob, rob_repr = plot_heatmap(pred, target_poses, representation, pred_joint_positions, ret=True, rob=rob, rob_repr=rob_repr)
            writer.add_image("training/pose", frame, iteration)

            sum_loss = 0
            sum_per_joint_loss = 0
            sum_pCKh50 = 0

            # perform validation step
            print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
            validation_loss = valid_step(model, loss_func, validation_loader, flags, iteration)

            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                save_checkpoint(model, optimizer, min_validation_loss, iteration, i, is_best=True)
                print("New best at ", validation_loss)

            model = model.train()


    log_training(iteration,
                 iteration % flags.num_train_steps_between_logs,
                 sum_loss,
                 sum_per_joint_loss,
                 sum_pCKh50)

    return iteration, min_validation_loss

def log_training(iteration, N, sum_loss, sum_per_joint_loss, sum_pCKh50):
    training_loss = sum_loss.item() / N
    training_per_joint_loss = sum_per_joint_loss.item() / N
    training_pCKh50 = sum_pCKh50.item() / N

    writer.add_scalar("training/per_joint_loss", training_per_joint_loss, iteration)
    writer.add_scalar("training/loss", training_loss, iteration)
    writer.add_scalar("training/pCKh50", training_pCKh50, iteration)

    fig = plot_grad_flow(model.named_parameters())
    writer.add_figure("training/gradients", fig, iteration)

def save_checkpoint(model, optimizer, min_validation_loss, iteration, epoch, is_best=False):
    state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()

    name = "model_best" if is_best else f"checkpoint_{iteration}"

    torch.save({
        "state_dict": state_dict,
        "opt_state_dict": opt_state_dict,
        "min_val_loss": min_validation_loss,
        "iteration": iteration,
        "epoch": epoch
    }, f"{flags.log_dir}/{name}.pth")


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--dataset", default="MPI", required=True)
    parser.add_argument("--root", default="", required=True)
    parser.add_argument("--real_root", default="")

    # logging options
    parser.add_argument("--log_dir", default="", required=True)

    parser.add_argument("--num_joints", type=int, default=13)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--num_epochs", type=int, default=30000)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_factor", type=float, default=1)

    parser.add_argument("--num_valid_steps", type=int, default=300)
    parser.add_argument("--num_test_steps", type=int, default=300)
    parser.add_argument("--num_train_steps_between_logs", type=int, default=300)
    parser.add_argument("--decrease_every_n_epochs", type=int, default=5)

    parser.add_argument("--drop_events_rate", type=float, default=0)
    parser.add_argument("--voxel_dimensions", type=int, nargs="+",  default=[5, 346, 346])
    parser.add_argument("--debug", action="store_true", default=False)
    flags = parser.parse_args()

    return flags


if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    try:
        dataset_cls = eval(flags.dataset)
    except Exception:
        raise ValueError(f"flags.dataset {flags.dataset} must point either to 'MPI' or 'DHP19' dataset.")

    training_dataset = dataset_cls(flags.root, split="train", debug=flags.debug, drop_events_rate=flags.drop_events_rate, augmentation=True)
    validation_dataset = dataset_cls(flags.root, split="val", augmentation=False)
    if flags.real_root != "":
        real_dataset = DHP19(flags.real_root, split="val", augmentation=False)
        real_loader = Loader(real_dataset, flags, sequential=True)
    
    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags)
    validation_loader = Loader(validation_dataset, flags, sequential=True)

    model = HumanPoseNet(voxel_dimension=flags.voxel_dimensions, num_joints=flags.num_joints, batch_size=flags.batch_size)
    #pose_resnet(batch_size=flags.batch_size,
            #            num_classes=flags.num_joints,
            #            resnet_layers=18)

    model = model.to(flags.device)

    loss_func = PoseLoss()

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=flags.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, flags.lr_decay_factor)

    writer = SummaryWriter(flags.log_dir)
    iteration = 0
    min_validation_loss = 1000

    # save flags to log dir
    if flags.resume:
        ckpt = torch.load(os.path.join(flags.log_dir, "model_best.pth"))
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        iteration = ckpt["iteration"]
        min_validation_loss = ckpt["min_val_loss"]
        epoch = ckpt["epoch"]
    else:
        torch.save(flags, os.path.join(flags.log_dir, "flags.pth"))

    for i in range(flags.num_epochs):
        if flags.real_root != "":
            # testing
            print(f"Real step [{i:3d}/{flags.num_epochs:3d}]")
            test_step(model, loss_func, real_loader, flags)

        # periodic saving
        if i % flags.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, min_validation_loss, iteration, i)

        # training
        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
        iteration, min_validation_loss = train_step(model, loss_func, training_loader, iteration, min_validation_loss)

        if i % flags.decrease_every_n_epochs == flags.decrease_every_n_epochs-1:
            lr_scheduler.step()

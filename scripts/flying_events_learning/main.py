import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm

from utils.models import Classifier
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy
from utils.dataset import *


torch.manual_seed(1)
np.random.seed(1)


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--training_root", nargs="+", required=True)
    parser.add_argument("--validation_root", nargs="+", required=True)
    parser.add_argument("--split_folder", default="", required=True)

    # logging options
    parser.add_argument("--log_dir", default="", required=True)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=3)

    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--log_video", action="store_true", default=False)
    parser.add_argument("--log_events", action="store_true", default=False)
    parser.add_argument("--num_classes", default=101, type=int)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--trainable", action="store_true", default=False)
    parser.add_argument("--voxel_grid", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--dataset", default="NCaltech101Events")

    flags = parser.parse_args()

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_root}\n"
          f"validation_dataset: {flags.validation_root}\n"
          f"----------------------------")

    return flags

def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:,None,None,None]

def create_image(representation):
    B, C, H, W = representation.shape
    representation = representation.view(B, 3, C // 3, H, W).sum(2)

    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99)
    robust_min_vals = percentile(representation, 1)

    representation = (representation - robust_min_vals)/(robust_max_vals - robust_min_vals)
    representation = torch.clamp(255*representation, 0, 255).byte()

    representation = torchvision.utils.make_grid(representation)

    return representation


if __name__ == '__main__':
    flags = FLAGS()

    if len(flags.training_root) > 1:
        dataset_cls = MergedNCaltech101Events
        training_root = flags.training_root
        validation_root = flags.validation_root
    else:
        training_root = flags.training_root[0]
        validation_root = flags.validation_root[0]

        try:
            dataset_cls = eval(flags.dataset)
        except Exception:
            raise ValueError("Dataset class not known: %s" % flags.dataset_cls)

    split1, split2 = "train", "val"

    # datasets, add augmentation to training set
    training_dataset = dataset_cls(training_root, flags.split_folder,
                                   split=split1,
                                   augmentation=True,
                                   num_classes=flags.num_classes,
                                   noise_level=flags.noise_level)

    validation_dataset = dataset_cls(validation_root, flags.split_folder,
                                     split=split2,
                                     augmentation=False,
                                     num_classes=flags.num_classes,
                                     noise_level=0)

    training_dataset[5000]
    training_dataset[2]

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, voxel_grid=flags.voxel_grid)
    validation_loader = Loader(validation_dataset, flags, voxel_grid=flags.voxel_grid)

    # model, and put to device
    model = Classifier(batch_size=flags.batch_size,
                       num_classes=flags.num_classes,
                       normalize=flags.normalize,
                       trainable=flags.trainable,
                       voxel_grid=flags.voxel_grid)

    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1)

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
    else:
        torch.save(flags, os.path.join(flags.log_dir, "flags.pth"))

    for i in range(flags.num_epochs):
        sum_accuracy = 0
        sum_loss = 0
        model = model.eval()
        event_key = None
        print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
        for data in tqdm.tqdm(validation_loader):
            # find key which contains events
            if event_key is None:
                for k in data:
                    if "image" in k or "event" in k:
                        event_key = k
                        break

            with torch.no_grad():
                pred_labels, representation = model(data[event_key], batchsize=len(data["label"]))
                loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, data["label"])

            sum_accuracy += accuracy
            sum_loss += loss

        validation_loss = sum_loss.item() / len(validation_loader)
        validation_accuracy = sum_accuracy.item() / len(validation_loader)

        writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
        writer.add_scalar("validation/loss", validation_loss, iteration)

        # visualize representation
        representation_vizualization = create_image(representation)
        writer.add_image("validation/representation", representation_vizualization, iteration)

        if flags.log_video and "cam0_image_raw" in data:
            video = data["cam0_image_raw"]  # B x T x 3 x H x W
            writer.add_video("validation/video", video, global_step=iteration)

        #if flags.log_events:
        #    events = data["cam0_events"]
        #    events_single_batch = events[events[:, -1] == 0, :-1].unsqueeze(0)
#
        #    # dilate time dimension
        #    events_single_batch[0, :, 2] *= 100
#
        #    colors = torch.zeros_like(events_single_batch)
        #    colors[:, events_single_batch[0, :, -1] == 1, -1] = 255
        #    colors[:, events_single_batch[0, :, -1] == -1, 0] = 255
        #    events_single_batch = events_single_batch[:, :, :-1]
        #    writer.add_mesh("validation/events", events_single_batch, colors, global_step=iteration)

        print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            state_dict = model.state_dict()
            opt_state_dict = optimizer.state_dict()

            torch.save({
                "state_dict": state_dict,
                "opt_state_dict": opt_state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, f"{flags.log_dir}/model_best.pth")
            print("New best at ", validation_loss)

        sum_accuracy = 0
        sum_loss = 0

        model = model.train()
        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
        for data in tqdm.tqdm(training_loader):
            optimizer.zero_grad()

            pred_labels, representation = model(data[event_key], batchsize=len(data["label"]))
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, data["label"])

            loss.backward()

            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1

        if i % 10 == 9:
            lr_scheduler.step()

        training_loss = sum_loss.item() / len(training_loader)
        training_accuracy = sum_accuracy.item() / len(training_loader)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)

        representation_vizualization = create_image(representation)
        writer.add_image("training/representation", representation_vizualization, iteration)

        if flags.log_video and "cam0_image_raw" in data:
            video = data["cam0_image_raw"]  # B x T x 3 x H x W
            writer.add_video("training/video", video, global_step=iteration)

        #if flags.log_events:
        #    events = data["cam0_events"]
        #    events = data["cam0_events"]
        #    events_single_batch = events[events[:,-1] == 0, :-1].unsqueeze(0)
#
        #    # dilate time dimension
        #    events_single_batch[0,:,2] *= 100
#
        #    colors = torch.zeros_like(events_single_batch)
        #    colors[:,events_single_batch[0,:,-1]==1,-1] = 255
        #    colors[:,events_single_batch[0,:,-1]==-1,0] = 255
        #    events_single_batch = events_single_batch[:, :, :-1]
        #    writer.add_mesh("training/events", events_single_batch, colors, global_step=iteration)




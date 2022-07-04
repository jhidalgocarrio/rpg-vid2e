from os.path import dirname
import argparse
import torch
import tqdm
import os

from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy
from utils.models import Classifier
from utils.dataset import NCaltech101Events


def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--checkpoint", default="", required=True)
    parser.add_argument("--split_folder", default="", required=True)
    parser.add_argument("--test_root", default="", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--dataset", default="NCaltech101Events")

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    assert os.path.isdir(flags.test_root), f"Test dataset directory {flags.test_root} not found."

    print(f"----------------------------\n"
          f"Starting testing with \n"
          f"checkpoint: {flags.checkpoint}\n"
          f"test_dataset: {flags.test_root}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"----------------------------")

    return flags


if __name__ == '__main__':
    flags = FLAGS()

    try:
        dataset_cls = eval(flags.dataset)
    except Exception:
        raise ValueError("Dataset class not known: %s" % flags.dataset)

    # datasets, add augmentation to training set
    test_dataset = dataset_cls(flags.test_root, flags.split_folder,
                                   split="test",
                                   augmentation=False,
                                   num_classes=101,
                                   noise_level=0)

    # construct loader, responsible for streaming data to gpu
    test_loader = Loader(test_dataset, flags, flags.device)

    # model, load and put to device
    model = Classifier(batch_size=flags.batch_size,
                       num_classes=101,
                       normalize=flags.normalize,
                       trainable=False,
                       voxel_grid=False)

    ckpt = torch.load(flags.checkpoint, map_location=flags.device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(flags.device)

    model = model.eval()
    sum_accuracy = 0
    sum_loss = 0

    print("Test step")
    event_key = None
    for data in tqdm.tqdm(test_loader):
        with torch.no_grad():
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

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy.item() / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
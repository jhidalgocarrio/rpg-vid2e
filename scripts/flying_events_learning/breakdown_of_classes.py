import argparse
import os
import torch
import tqdm
import yaml

from utils.dataset import NCaltech101Events
from utils.loader import Loader
from utils.models import Classifier


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--datasets", nargs="+", default=[
        #"sim", "/media/dani/data/ncaltech_experiments/sim-N-Caltech101_vox",
        "real", "/media/dani/data/ncaltech_experiments/N-Caltech101"
    ])
    parser.add_argument("--split_folder", default="/media/dani/data/ncaltech_experiments/split_train_0.5_val_0.25_test_0.25")
    parser.add_argument("--output_path", default="/tmp/breakdown.yaml")

    # logging options
    parser.add_argument("--checkpoints", nargs="+", default=[
        "sim", "/media/dani/data/ncaltech_experiments/log/train_sim_val_sim_learnable_aug_normalized/model_best.pth",
        "real",  "/media/dani/data/ncaltech_experiments/log/train_real_val_real_learnable_normalized/model_best.pth",
    ])

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--num_classes", default=101, type=int)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--voxel_grid", action="store_true", default=False)
    parser.add_argument("--topn", type=int, default=5)

    flags = parser.parse_args()

    assert os.path.isdir(flags.split_folder), f"Split directory {flags.split_folder} not found."

    return flags


def load_loaders(datasets, split_folder, voxel_grid):
    labels = datasets[::2]
    roots = datasets[1::2]

    loaders = {}
    for label, root in zip(labels, roots):
        dataset = NCaltech101Events(root, split_folder,
                                       split="test",
                                       augmentation=False,
                                       num_classes=flags.num_classes,
                                       noise_level=0,
                                       return_path=True)

        loaders[label] = Loader(dataset, flags, sequential=True, voxel_grid=voxel_grid)

    labels = sorted(labels)
    loaders = [loaders[l] for l in labels]

    return loaders, labels

def load_models(checkpoints, flags):
    labels = checkpoints[::2]
    checkpoints = checkpoints[1::2]

    models = {}
    for label, ckpt in zip(labels, checkpoints):
        model = Classifier(batch_size=flags.batch_size,
                           num_classes=flags.num_classes,
                           normalize=flags.normalize,
                           voxel_grid=flags.voxel_grid,
                           trainable=False)

        ckpt = torch.load(ckpt, flags.device)

        try:
            model.load_state_dict(ckpt["state_dict"])
        except Exception as e:
            print("Something went wrong with loading model %s" % ckpt)
            raise e

        model = model.to(flags.device)
        model = model.eval()
        models[label] = model

    labels = sorted(labels)
    models = [models[l] for l in labels]

    return models, labels

def find_topn(classes, predictions, topn):
    """
    :param classes: list (len N)
    :param predictions: torch.Tensor B x N
    :param topn:
    :return:
    """
    data = []
    softmax = torch.nn.Softmax(dim=1)
    predictions, idx = predictions.sort(-1)
    predictions = softmax(predictions)
    for i in range(predictions.shape[0]):
        d = []
        for j in range(topn):
            d += [{classes[idx[i,-j-1]]: predictions[i,-j-1].item()}]
        data.append(d)
    return data

if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    loaders, dataset_labels = load_loaders(flags.datasets, flags.split_folder, flags.voxel_grid)
    models, model_labels = load_models(flags.checkpoints, flags)
    classes = loaders[0].loader.dataset.classes

    summary_data = {}
    event_key = None
    for data in tqdm.tqdm(zip(*loaders)):
        for dataset_label, dataset_data in zip(dataset_labels, data):
            if event_key is None:
                for k in dataset_data:
                    if "event" in k:
                        event_key = k
                        break

            if dataset_label not in summary_data:
                summary_data[dataset_label] = {}

            for model_label, model in zip(model_labels, models):
                with torch.no_grad():
                    pred_labels, _ = model(dataset_data[event_key], batchsize=len(dataset_data["label"]))
                    topn_data = find_topn(classes, pred_labels, flags.topn)

                    if model_label not in summary_data[dataset_label]:
                        summary_data[dataset_label][model_label] = {}

                    for i, p in enumerate(dataset_data["path"]):
                        summary_data[dataset_label][model_label][p] = topn_data[i]

    with open(flags.output_path, 'w') as outfile:
        yaml.dump(summary_data, outfile, default_flow_style=False)

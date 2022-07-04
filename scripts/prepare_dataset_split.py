import argparse
import glob
import os


def FLAGS():
    parser = argparse.ArgumentParser("Create dataset split for reproducibility")
    # rostopics
    parser.add_argument("--dataset_root", help="Location of dataset.", default="/home/dani/code/catkin_ws/src/rpg_vid2e/dataset")
    parser.add_argument("--split", help="Split [training, validation, test] e.g. 0.5 0.25 0.25", nargs="+", type=float, default=[.5,.25,.25])
    parser.add_argument("--output_root", help="Where the split will be saved.", default="/home/dani/code/catkin_ws/src/rpg_vid2e/dataset")

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root
    assert os.path.isdir(args.output_root), "%s should be a valid dir." % args.output_root

    return parser.parse_args()


def collect_mapping(files):
    """creates a mapping of the form label: [idx0, idx1,...], ..."""
    mapping = {}
    for f in files:
        label = os.path.basename(os.path.dirname(f))
        file_name = os.path.basename(f)
        counter = file_name.split(".")[-2].split("_")[-1]

        if label not in mapping:
            mapping[label] = []

        if counter not in mapping[label]:
            mapping[label].append(counter)

    return mapping

def parse_split(split):
    if len(split) == 2:
        tr, v = split
        te  = 1 - v - tr
    elif len(split) == 3:
        tr, v, te = split
        assert tr + v + te == 1, "Split needs three numbers that sum to 1."
    else:
        raise ValueError("Split needs three numbers that sum to 1.")

    return tr, v, te

def perform_split(data, split_ratios):
    splits = ["train", "val", "test"]
    split_data_mapping = {}
    for label, counts in data.items():

        counter = 0
        for split, ratio in zip(splits, split_ratios):

            num_elements_split = int(ratio * len(counts))
            split_data = counts[counter:counter+num_elements_split]

            if split not in split_data_mapping:
                split_data_mapping[split] = {}
            if label not in split_data_mapping[split]:
                split_data_mapping[split][label] = split_data

            counter += num_elements_split
    return split_data_mapping


if __name__ == '__main__':
    flags = FLAGS()
    files = glob.glob(os.path.join(flags.dataset_root, "*", "*"))
    mapping = collect_mapping(files)

    train_split, val_split, test_split = parse_split(flags.split)

    # performs split for each label, so that the splits are well balanced
    split_ratios = [train_split, val_split, test_split]
    split_data = perform_split(mapping, split_ratios)

    # save files to disk, is of the form train.txt
    folder_name = os.path.join(flags.output_root, "split_train_%s_val_%s_test_%s" % (train_split, val_split, test_split))
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    print("Writing split files to %s:" % folder_name)
    for i, split in enumerate(["train", "val", "test"]):
        print("\t%5s: ratio: %.3f file: %s" % (split, split_ratios[i], os.path.join(folder_name, split + ".txt")))
        with open(os.path.join(folder_name, split + ".txt"), "w") as f:
            for label in sorted(split_data[split].keys()):
                f.write(label + ":\n")
                for count in split_data[split][label]:
                    f.write("  - " + count + "\n")





from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import yaml
import copy

def merge_datasets(augmented_classes, merged_augmented_classes):
    classes = sorted(os.listdir(augmented_classes))
    for _, i in enumerate(classes):
        source_item = os.path.join(augmented_classes, i+"/*_au.*")
        target_folder = os.path.join(merged_augmented_classes, i)
        os.system("mkdir %s"%target_folder)
        print("cp %s %s" %(source_item, target_folder))
        os.system("cp %s %s" %(source_item, target_folder))

merge_datasets("/home/javi/uzh-rpg/datasets/a-caltech101_bis/classes", "/home/javi/uzh-rpg/datasets/a-caltech101_bis/split/classes")
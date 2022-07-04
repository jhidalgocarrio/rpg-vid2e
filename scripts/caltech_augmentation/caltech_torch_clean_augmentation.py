# License: BSD

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def check_dataset(model, image_loaders, device, ipath):
    correct = 0
    total = 0
    cmd = "mkdir "+ipath
    os.system(cmd)
    with torch.no_grad():
        model.eval()
        list_tensors = []
        for idx, (inputs, labels, paths) in enumerate(image_loaders['classes']):
            #print ("idx %d" % idx, end = '\r')
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(inputs, labels, paths)
            #out = torchvision.utils.make_grid(inputs)

            #imshow(out, title=[class_names[x] for x in labels])

            image_outputs = model_ft(inputs) #b batch size * number of classes
            _, image_preds = torch.max(image_outputs, 1)

            list_tensors.append(image_outputs)
            print ("Outputs ", image_outputs[:1,:])
            x_distribution =  torch.softmax(image_outputs[:1,:], dim = 1)
            print ("distribution ", x_distribution)
            #print ("Predicts %s" % image_preds)
            #print ("Labels %s" % labels)
            total += labels.size(0)
            correct += (image_preds == labels).sum().item()
            #print("idx %d" % total, end = '\r')
            for (i, j, p) in zip(enumerate(image_preds), enumerate(labels), enumerate(paths)):
                #print("\npredicted %s wirth label %s" % (class_names[i[0]], class_names[j[0]]))
                #print (i[1].item(), p[1], j[1].item())
                if i[1].item() is not j[1].item():
                    label_ipath = os.path.join(ipath,class_names[j[1].item()])
                    if not os.path.exists(label_ipath):
                        os.system("mkdir "+label_ipath)
                    #print("\npredicted %s wirth label %s" % (class_names[i[1].item()], class_names[j[1].item()]))
                    #print ("remove %s" % p[1])
                    filename_w_ext = os.path.basename(p[1])
                    filename, file_extension = os.path.splitext(filename_w_ext)
                    if filename.find('_au.') > -1:
                        copy_cmd = "mv "+p[1]+" "+label_ipath +"/"+filename+"_"+class_names[i[1].item()]+file_extension
                        os.system(copy_cmd)
    print("size of list_tensors: ", len(list_tensors))
    final_tensor = torch.cat(list_tensors,0)
    print("size of final_tensor: ", final_tensor.shape)
    np_tensor = final_tensor.cpu().detach().numpy()
    print ("np_tensor type is ", type(np_tensor))
    print ("np_tensor size is ", np_tensor.shape)
    np.save("car_sips_sequence.npy", np_tensor)
    print('accuracy of the network on the %d images: %d %%' % (total, ( 100 * correct / total)))


if __name__  == "__main__":

    image_transforms = {
        # should have the same transform that validation
        'classes' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the model
    model_ft = torch.load("caltech101_resnet34_augmented.pth", device)

    #data_dir = '/media/javi/rpg-archive/Datasets/vid2e/caltech101/'
    #data_dir = '/home/javi/uzh-rpg/datasets/a-caltech101_bis/'
    data_dir = '/home/javi/uzh-rpg/datasets/roberto_trail/imgs_car_dataset-20191114-191035/'
    plt.ion()   # interactive mode

    # Test data set
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                          image_transforms[x])
                    for x in ['classes']}

    # Load the test sets
    image_loaders = {x :torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=False,
        num_workers=4)
        for x in ['classes']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['classes']}

    class_names = image_datasets['classes'].classes

    invalid_images_path = os.path.join(data_dir, "invalid")
    check_dataset(model_ft, image_loaders, device, invalid_images_path)

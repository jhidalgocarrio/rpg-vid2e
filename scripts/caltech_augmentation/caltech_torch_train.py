# License: BSD

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
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

def train_model(model, criterion, optimizer, scheduler, writer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                # the scheduler updates the learning rate. parameters of the model
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            if phase == 'valid':
                writer.add_scalar('Loss/valid', epoch_loss, epoch)
                writer.add_scalar('Accuracy/valid', epoch_acc, epoch)
 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def test_model(model, num_images=6):
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            test_outputs = model_ft(inputs)
            _, test_preds = torch.max(test_outputs, 1)
            #for i in range(len(labels)):
            #    print("test pred: %s label: %s" % (class_names[test_preds[i]], class_names[labels[i]]))

            total += labels.size(0)
            correct += (test_preds == labels).sum().item()

    print('Accuracy of the network on the %d test images: %f %%' % (total, 100 * correct / total))


def fivecrop_test_model(model, num_images=6):
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            bs, ncrops, c, h, w = inputs.size()
            print (bs, ncrops, c, h, w)
            test_outputs = model_ft(inputs.view(-1, c, h, w))
            test_outputs_avg = test_outputs.view(bs, ncrops, -1).mean(1) # avg over crops
            _, test_preds = torch.max(test_outputs_avg, 1)

            total += labels.size(0)
            correct += (test_preds == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (total,  100 * correct / total))

if __name__  == "__main__":
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), #randomly crop
            transforms.RandomHorizontalFlip(), # random horizontal flip
            transforms.ToTensor(), # To tensor which is based on numpy ndarray
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize pixel values (mean and standard deviation)
        ]),
        # validation. No perform random crop of flip. 
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # this is good for validation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    test_transforms = {
        # test should have the same transform that validation
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }
 
    fivecrop_test_transforms = {
        # test should have the same transform that validation
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224), # this is a list of PIL Images
            transforms.Lambda(lambda crops: [ transforms.ToTensor()(crop) for crop in crops]),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
            ]),
    }
 
    data_dir = '/home/javi/uzh-rpg/datasets/a-caltech101_bis/split'
    print ("Loading dataset: %s\n"%data_dir)

    # prepare the data set for the loader. applied transform and assign classes
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'valid']}

    # The loader. workers are the number of processes. batch_size the number of samples each process load.
    # shuffle is true if randomly select samples
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    plt.ion()   # interactive mode

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
 
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Tensorboard  Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    #imshow(out, title=[class_names[x] for x in classes])
    writer.add_image('images', out, 0)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, writer, num_epochs=25)

    #visualize_model(model_ft)

    # Test data set
    test_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          test_transforms[x])
                    for x in ['test']}

    # Load the test sets
    test_loaders = {x :torch.utils.data.DataLoader(
        test_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=4)
        for x in ['test']}

    # Saving the entery model
    model_filename = "caltech101_resnet34.pth"
    torch.save(model_ft, model_filename)
 
    # Test the model
    test_model(model_ft)

    writer.close()
import torch
import torchvision
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from visualize_pose import generate_event_image, draw_poses

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters, lr=1):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    figsize = (10,10)
    fig, ax = plt.subplots(figsize=figsize)

    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and p.grad is not None:
            layers.append(n)
            ave_grads.append(lr*p.grad.abs().mean())
            max_grads.append(lr*p.grad.abs().max())
            min_grads.append(lr*p.grad.abs().min())

    ax.bar(3*np.arange(len(max_grads)), max_grads, lw=2, color="r")
    ax.bar(3*np.arange(len(max_grads)), ave_grads, lw=2, color="m")
    ax.bar(3*np.arange(len(max_grads)), min_grads, lw=2, color="b")

    ax.set_xticks(range(0, 3*len(ave_grads), 3))
    labels = ax.set_xticklabels(layers)
    for l in labels:
        l.update({"rotation": "vertical"})

    ax.set_xlim(left=0, right=3*len(ave_grads))
    ax.set_ylim(bottom=1e-11*lr, top=1e2*lr)
    ax.set_yscale("log")# zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="r", lw=4),
               Line2D([0], [0], color="m", lw=4),
               Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient', 'min-gradient'])

    fig.tight_layout()
    return fig



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
    image = (.5 * heat_map_color.float() + .5 * representation.float())  # alpha blending

    B, _, H, W = image.shape

    for b in range(B):
        s = 2
        x, y = pred_joint_positions[b, idx, :].int()
        xmin, ymin = max([0, x - s]), max([0, y - s])
        xmax, ymax = min([W - 1, x + s]), min([H - 1, y + s])
        image[b, 0, ymin:ymax, xmin:xmax] = torch.ones((ymax - ymin, xmax - xmin))
        image[b, 1, ymin:ymax, xmin:xmax] = torch.zeros((ymax - ymin, xmax - xmin))
        image[b, 2, ymin:ymax, xmin:xmax] = torch.zeros((ymax - ymin, xmax - xmin))

    if pose is None:
        return image, rob

    for b in range(B):
        x, y = pose[b,idx,:2].int()

        # generate per joint images
        if x >= 0 and y >= 0 and x < H and y < W:
            s = 2
            xmin, ymin = max([0, x - s]), max([0, y - s])
            xmax, ymax = min([W - 1, x + s]), min([H - 1, y + s])
            image[b, 0, ymin:ymax, xmin:xmax] = torch.ones((ymax - ymin, xmax - xmin))
            image[b, 1, ymin:ymax, xmin:xmax] = torch.zeros((ymax - ymin, xmax - xmin))
            image[b, 2, ymin:ymax, xmin:xmax] = torch.ones((ymax - ymin, xmax - xmin))

    return image, rob

def plot_heatmap(heat_map, pose, representation, pred_joint_positions, ret=False, rob=None, rob_repr=None):
    import matplotlib.cm as cm
    mapper = cm.get_cmap("jet")

    heat_map[heat_map<1e-4] = 0

    representation = representation.sum(1, keepdim=True)
    representation = representation.detach().cpu()
    representation, rob_repr = robust_min_max_norm(representation, ret=True, rob=rob_repr)
    representation_color = representation.repeat(1, 3, 1, 1)

    images = []
    for idx in range(13):
        image, rob = single_heatmap(idx, heat_map, mapper, representation_color, pred_joint_positions, pose, rob)
        images.append(image)

    B, C, H, W = images[0].shape
    img = torch.zeros(B, C, 3*H, 5*W)

    for i in range(13):
        row = i // 5
        col = i % 5
        img[:,:,H*row:H*row+H, W*col:W*col+W] = images[i]

    images = torch.clamp(255 * img, 0, 255).byte()
    images = torchvision.utils.make_grid(images[:8])

    if ret:
        return images, rob, rob_repr

    return images

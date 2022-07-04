import torch
import math
import numpy  as np

# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class PoseLoss(torch.nn.Module):
    def __init__(self, kernel_size=31, sigma=2, channels=13, return_confidence=False):
        torch.nn.Module.__init__(self)
        self.gaussian_kernel = get_gaussian_kernel(kernel_size, sigma, channels)
        self.num_joints = channels
        self.return_confidence = return_confidence

    def PCKh50(self, errors, headsize, visible):
        threshold = headsize*0.5
        inliers = (errors <= threshold[:,None])
        eps = 1e-3
        PCKh50 = (inliers.float() * visible).sum() / (eps+visible.sum())
        return PCKh50

    def construct_target_heatmap(self, prediction, targets):
        B, C, H, W = prediction.shape

        targets_long = targets.long()
        r = targets - targets_long.float()
        x_long, y_long = targets_long[...,0], targets_long[...,1]
        rx, ry = r[...,0], r[...,1]

        lin_idx00 = x_long + y_long*W  # B x C
        batch_offset = C*H*W*torch.arange(B, dtype=torch.long, device=targets.device).view(-1,1)
        channel_offset = H*W*torch.arange(C, dtype=torch.long, device=targets.device).view(1,-1)
        lin_idx00 = lin_idx00 + batch_offset + channel_offset
        lin_idx10 = lin_idx00 + 1
        lin_idx01 = lin_idx00 + W
        lin_idx11 = lin_idx01 + 1

        mask00 = (x_long>=0) & (y_long>=0) & (x_long<H) & (y_long<H)
        mask01 = (x_long>=0) & (y_long+1>=0) & (x_long<H) & (y_long+1<H)
        mask10 = (x_long+1>=0) & (y_long>=0) & (x_long+1<H) & (y_long<H)
        mask11 = (x_long+1>=0) & (y_long+1>=0) & (x_long+1<H) & (y_long+1<H)

        target_heat_map = torch.zeros_like(prediction)

        target_heat_map.put_(lin_idx00[mask00], (1-rx[mask00])*(1-ry[mask00]), accumulate=True)
        target_heat_map.put_(lin_idx01[mask01], (1-rx[mask01])*(ry[mask01]), accumulate=True)
        target_heat_map.put_(lin_idx10[mask10], (rx[mask10])*(1-ry[mask10]), accumulate=True)
        target_heat_map.put_(lin_idx11[mask11], (rx[mask11])*(ry[mask11]), accumulate=True)

        target_heat_map = self.gaussian_kernel(target_heat_map)

        return target_heat_map

    def forward(self, prediction, targets, head_size):
        loss = None
        mean_position_error_2d = None
        target_heat_map = None
        pCKh50 = None
        position_error_2d = None

        self.gaussian_kernel = self.gaussian_kernel.to(prediction.device)

        # find preprocessed labels
        if targets is not None:
            target_heat_map = self.construct_target_heatmap(prediction, targets)

        # compute per joint distance
        B, C, H, W = prediction.shape
        # set padding region to 0
        px = (int(np.ceil(W/8)*8) - W)//2
        py = (int(np.ceil(H/8)*8) - H)//2

        prediction[:,:,:py,:px] = 0
        prediction[:,:,:py,W-px:] = 0
        prediction[:,:,H-py:,:px] = 0
        prediction[:,:,H-py:,W-px:] = 0

        max_confidence, indices = prediction.reshape(B, C, -1).max(-1)
        x_pred = indices.float() % W
        y_pred = indices.float() // W
        pred_joints = torch.stack([x_pred, y_pred], -1)

        if targets is not None:
            visible = targets[...,2]
            position_error_2d = (pred_joints - targets[...,:2]).pow(2).sum(-1).pow(.5)
            eps = 1e-3
            mean_position_error_2d = (position_error_2d * visible).sum() / (eps+visible.sum())

            # compute PCKh@0.5 - this computes the fraction of keypoints that are within a threshold L from groundtruth
            # L is computed from alpha * headsize. Here alpha = 0.5
            pCKh50 = self.PCKh50(position_error_2d, head_size, visible)

            # compute loss
            loss = (target_heat_map - prediction).pow(2).view(B,C, -1).sum(-1).mean()

        if self.return_confidence:
            return loss, mean_position_error_2d, pred_joints, target_heat_map, pCKh50, position_error_2d, max_confidence

        return loss, mean_position_error_2d, pred_joints, target_heat_map, pCKh50, position_error_2d

if __name__ == '__main__':
    import torch
    import numpy as np
    import time

    b = 10
    c = 13
    test_array = torch.Tensor(b,c,200,200).float().normal_()
    targets = torch.from_numpy(np.random.randint(1, 199, (b, c,2)) + np.random.random()).float()
    head_size = torch.from_numpy(np.random.randint(1, 199, (b,)) + np.random.random()).float()


    pose_loss = PoseLoss(channels=c)

    test_array = test_array.cuda()
    targets = targets.cuda()
    head_size = head_size.cuda()

    ts = []
    pose_loss.forward(test_array, targets, head_size)
    for i in range(100):
        t1 = time.perf_counter()

        #l2 = pose_loss.construct_target_heatmap1(test_array, targets)
        l1 = pose_loss.construct_target_heatmap(test_array, targets)
        #print(l1)

        #print((l1-l2).abs().max())
        #print(l2)

        dt = time.perf_counter() - t1

        ts.append(dt)

    print(np.median(dt))
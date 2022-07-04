import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=32):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation
        self.num_channels = num_channels

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        return
        if False:#isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        return self.trilinear_kernel(x, self.num_channels)
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-5)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(10000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

        with torch.no_grad():
            ts = torch.arange(-1,1,.01).view(1,-1)
            gt_values = self.trilinear_kernel(ts, num_channels)
            values = self.forward(ts)
            loss = (values - gt_values).abs().mean()



    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts >= 0] = (1 - (num_channels-1) * ts)[ts >= 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts >= 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 batch_size=None):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim
        self.B = batch_size
        self.C = dim[0]
        self.vox = torch.zeros(batch_size, 2*dim[0], dim[1], dim[2])

    def forward(self, events, batchsize=None):
        self.vox = self.vox.to(events["t"].device)
        vox = self.vox[:batchsize]
        vox.zero_()
        B, C, H, W = vox.shape
        C = C//2
        vox = vox.view(-1)

        # get values for each channel
        x, y, p, b = events["xypb"].t()
        t = events["t"]

        p = (p+1).int()/2  # maps polarity to 0, 1

        idx_before_bins = x.int() + W * y.int() + 0 + W * H * C * p + W * H * C * 2 * b.int()

        for i_bin in range(self.C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(15,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 20, 20, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True,
                 batch_size=None,
                 normalize=False,
                 trainable=False,
                 voxel_grid=False):

        nn.Module.__init__(self)
        self.trainable = trainable
        self.voxel_grid = voxel_grid
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation, batch_size=batch_size)
        self.classifier = resnet34(pretrained=pretrained)
        self.normalize = normalize
        if self.normalize:
            self.norm = torch.nn.InstanceNorm2d(voxel_dimension[0])

        self.crop_dimension = crop_dimension

        # replace fc layer and first convolutional layer
        input_channels = 2*voxel_dimension[0]
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution)

        return x

    def forward(self, x, batchsize):
        if self.voxel_grid:
            vox = x
            vox_cropped = x
        else:
            vox = self.quantization_layer.forward(x, batchsize)
            vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)

        if self.normalize:
            vox_cropped = self.norm(vox_cropped)

        # normalize by
        pred = self.classifier.forward(vox_cropped)
        return pred, vox

def fill_events(events, dimensions, values):
    C, H, W = dimensions
    img = np.zeros((H*W*C,), dtype=np.float32)
    x,y,t,p = events.T

    t_int = t.astype(np.int64)
    r = t-t_int

    lin_idx1 = (x + W * y + W * H * t_int).astype(np.int64)
    lin_idx2 = (x + W * y + W * H * (t_int+1)).astype(np.int64)

    in_bound = (t_int+1 <= C-1)

    np.add.at(img, lin_idx1, (1-r)*values)
    np.add.at(img, lin_idx2[in_bound], (r[in_bound])*values[in_bound])

    img = img.reshape((C, H, W))
    return img

def voxel_grid(events, channels, resolution):
    (H, W), C = resolution, channels

    t = (events[:,2]-events[0,2])/(events[-1,2]-events[0,2])

    if len(events) > 1:
        events[:,2] = (events[:,2]-events[0,2])/(events[-1,2]-events[0,2])
        events[:,2] *= (C-1)

    img_neg = fill_events(events[events[:,-1]==0], [C, H, W], t[events[:,-1]==0])
    img_pos = fill_events(events[events[:,-1]==1], [C, H, W], t[events[:,-1]==1])

    vox = np.concatenate([img_neg, img_pos], 0)

    return vox

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    events = np.load("/media/dani/data/ncaltech_experiments/sim-N-Caltech101/accordion/cam0_events_0002.npy")
    events = events.astype(np.float32)
    events = events[(events[:,0]==140) & (events[:,1]==75)][:2]
    events[:,-1] = 0
    test_vox = voxel_grid(events, 15, (180, 240))

    model = Classifier(batch_size=1)
    N = len(events)
    events = np.concatenate([events, np.zeros((N,1))], -1).astype(np.float32)
    events = torch.from_numpy(events)
    events[:,-2] = 2*events[:,-2]-1

    print(len(events))
    _, vox = model(events, batchsize=1)
    vox3 = np.load("/media/dani/data/ncaltech_experiments/sim-N-Caltech101_vox/accordion/vox_15_0002.npy")

    img = np.zeros((180,3*240))
    i = 5
    error = (vox - torch.from_numpy(test_vox)).abs()
    m = error.max()
    print(m)
    #img[:,:240] = test_vox[i,...]
    #img[:,240:2*240] = vox[0,i,...]
    img[:,2*240:] = error[0,i,...]

    print((vox-torch.from_numpy(test_vox)).abs().view(30,-1).sum(-1), vox3.sum())
    print((vox-torch.from_numpy(test_vox)).abs().view(30,-1).sum())
    print("learnable",(vox).abs().view(30,-1).sum(-1))
    print("voxelgrid function",torch.from_numpy(test_vox).abs().view(30,-1).sum(-1))
    #img = np.clip((255*img), 0,255).astype(np.uint8)
    img = error[0].sum(0)
    print(img.shape)
    im = plt.imshow(img, vmin=0, vmax=1)
    plt.show()#block = False)


   # for j in range(30):
   #     #img[:, :240] = test_vox[j, ...]
   #     #img[:, 240:2*240] = vox[0, j, ...]
   #     img[:, 2*240:] = error[0, j, ...]
   #     im.set_data(img)
   #     plt.pause(1)


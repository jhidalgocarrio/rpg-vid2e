from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from torchvision.models.resnet import BasicBlock, Bottleneck
import torchgeometry as tgm


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

        self.vox = torch.zeros(int(2 * np.prod(self.dim) * batch_size))

    def forward(self, events, batchsize=None):
        # points is a list, since events can have any size
        num_voxels = int(2 * np.prod(self.dim) * batchsize)
 
        device = events.device if type(events) is not dict else events["p"].device

        self.vox = self.vox.to(device)
        self.vox.zero_()

        C, H, W = self.dim

        # get values for each channel
        if type(events) is dict: 
            (x,y),p,b,t = events["xy"].t(), events["p"][:,0], events["b"][:,0], events["t"][:,0]
        else:
            x, y, t, p, b = events.t()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x.int() \
                          + W * y.int() \
                          + 0 \
                          + W * H * C * p.int() \
                          + W * H * C * 2 * b.int()

        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            self.vox.put_(idx.long(), values, accumulate=True)

        self.vox = self.vox.view(-1, 2, C, H, W)
        self.vox = torch.cat([self.vox[:, 0, ...], self.vox[:, 1, ...]], 1)

        #scales = []
        for i in range(batchsize):
            mask = self.vox[i].abs() > 0
            eps = 1e-3
            vox_mask = self.vox[i,mask]
            self.vox[i, mask] = (vox_mask-vox_mask.mean())/(vox_mask.std(unbiased=True)+eps)

        return self.vox[:batchsize]


class HumanPoseNet(nn.Module):
    def __init__(self,
                 voxel_dimension=(15,346,346),  # dimension of voxel will be C x 2 x H x W
                 num_joints=13,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 batch_size=None):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation, batch_size=batch_size)
        self.human_net = HumanNet(num_joints=num_joints, num_channels=2*voxel_dimension[0])
        self.augmentation = AugmentationModule(voxel_dimension[1:])

    def forward(self, x, batchsize, train=False):
        vox = self.quantization_layer.forward(x, batchsize)
        if train:
            vox, hom = self.augmentation(vox)
        else:
            hom = None
        pred = self.human_net.forward(vox)
        return pred, vox, hom

class HumanNet(torch.nn.Module):
    def __init__(self, num_joints, num_channels):
        torch.nn.Module.__init__(self)

        self.num_joints = num_joints
        self.num_channels = num_channels

        self.mp = torch.nn.MaxPool2d(2, 2)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.relu = torch.nn.LeakyReLU()

        nc = num_channels

        self.bn01 = lambda x : x #torch.nn.BatchNorm2d(16)
        self.bn02 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn03 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn04 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn05 = lambda x : x #torch.nn.BatchNorm2d(64)
        self.bn06 = lambda x : x #torch.nn.BatchNorm2d(64)
        self.bn07 = lambda x : x #torch.nn.BatchNorm2d(64)
        self.bn08 = lambda x : x #torch.nn.BatchNorm2d(64)
        self.bn09 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn10 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn11 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn12 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn13 = lambda x : x #torch.nn.BatchNorm2d(32)
        self.bn14 = lambda x : x #torch.nn.BatchNorm2d(16)
        self.bn15 = lambda x : x #torch.nn.BatchNorm2d(16)
        self.bn16 = lambda x : x #torch.nn.BatchNorm2d(16)

        self.conv01 = torch.nn.Conv2d(nc, 16, 3, 1, 1)
        self.conv02 = torch.nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.conv03 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv04 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv05 = torch.nn.Conv2d(32, 64, 3, 1, 2, bias=False, dilation=2)
        self.conv06 = torch.nn.Conv2d(64, 64, 3, 1, 2, bias=False, dilation=2)
        self.conv07 = torch.nn.Conv2d(64, 64, 3, 1, 2, bias=False, dilation=2)
        self.conv08 = torch.nn.Conv2d(64, 64, 3, 1, 2, bias=False, dilation=2)
        self.conv09 = torch.nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.conv10 = torch.nn.Conv2d(32, 32, 3, 1, 2, bias=False, dilation=2)
        self.conv11 = torch.nn.Conv2d(32, 32, 3, 1, 2, bias=False, dilation=2)
        self.conv12 = torch.nn.Conv2d(32, 32, 3, 1, 2, bias=False, dilation=2)
        self.conv13 = torch.nn.Conv2d(32, 32, 3, 1, 2, bias=False, dilation=2)
        self.conv14 = torch.nn.Conv2d(32, 16, 3, 1, 1, bias=False)
        self.conv15 = torch.nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.conv16 = torch.nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.conv17 = torch.nn.Conv2d(16, 13, 3, 1, 1, bias=False)

    def forward(self, x):

        B, C , H, W = x.shape

        padding_h = int(np.ceil(H / 4)*4) - H
        padding_w = int(np.ceil(W / 4)*4) - W

        self.padder = torch.nn.ConstantPad2d((padding_w//2, padding_w-padding_w//2, padding_h//2, padding_h-padding_h//2), 0)
        self.unpadder = torch.nn.ConstantPad2d((-padding_w//2, -padding_w+padding_w//2, -padding_h//2, -padding_h+padding_h//2), 0)

        x = self.padder(x)

        x = self.bn01(self.mp(self.relu(self.conv01(x))))
        x = self.bn02(self.relu(self.conv02(x)))
        x = self.bn03(self.relu(self.conv03(x)))
        x = self.bn04(self.mp(self.relu(self.conv04(x))))
        x = self.bn05(self.relu(self.conv05(x)))
        x = self.bn06(self.relu(self.conv06(x)))
        x = self.bn07(self.relu(self.conv07(x)))
        x = self.bn08(self.relu(self.conv08(x)))
        x = self.bn09(self.relu(self.conv09(self.up(x))))
        x = self.bn10(self.relu(self.conv10(x)))
        x = self.bn11(self.relu(self.conv11(x)))
        x = self.bn12(self.relu(self.conv12(x)))
        x = self.bn13(self.relu(self.conv13(x)))
        x = self.bn14(self.relu(self.conv14(self.up(x))))
        x = self.bn15(self.relu(self.conv15(x)))
        x = self.bn16(self.relu(self.conv16(x)))
        x = self.conv17(x)

        x = self.unpadder(x)

        return x

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


# Specification
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')
}

class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.features.append(
                nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x


class ResPoseNet(nn.Module):
    def __init__(self, repr, backbone, head, sensor_size=(346,346)):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.norm = torch.nn.InstanceNorm2d(2*repr.dim[0])
        self.repr = repr

        self.conv1 = nn.Conv2d(2*repr.dim[0], 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)
        self.pad = None

        self.augmentation = AugmentationModule(sensor_size)
        #self.down = torch.nn.UpsamplingBilinear2d(scale_factor=1/1.5)

    def forward(self, x, batchsize, train=False, only_repr=False):
        if train:
            repr = self.repr.forward(x, batchsize, train)
        else:
            repr = self.repr.forward(x, batchsize, train)

        if only_repr:
            return repr

        homography = None

        if train:
            repr, homography = self.augmentation(repr)

        B, C, H, W = repr.shape
        Wp = int(np.ceil(W / 8)) * 8
        Hp = int(np.ceil(H / 8)) * 8
        px = (Wp - W) // 2
        py = (Hp - H) // 2

        if self.pad is None:
            self.pad = torch.nn.ReplicationPad2d([px, px, py, py])

        x = self.pad(repr)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.backbone(x)
        x = self.head(x)
        x = self.upsample(x)

        x = x[...,py:H+py,px:W+px]

        if train:
            return x, repr, homography
        else:
            return x, repr, homography


# Helper functions
def get_default_network_config():
    config = dict()
    config["from_model_zoo"] = True
    config["pretrained"] = ''
    config["num_layers"] = 50
    config["num_deconv_layers"] = 3
    config["num_deconv_filters"] = 256
    config["num_deconv_kernel"] = 4
    config["final_conv_kernel"] = 1
    config["depth_dim"] = 1
    config["input_channel"] = 3
    return config

def init_pose_net(pose_net, name):
    org_resnet = model_zoo.load_url(model_urls[name])
    # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
    org_resnet.pop('fc.weight', None)
    org_resnet.pop('fc.bias', None)
    org_resnet.pop('conv1.weight', None)
    org_resnet.pop('conv1.bias', None)
    pose_net.backbone.load_state_dict(org_resnet)
    print("Init Network from model zoo")

# create network
def pose_resnet(voxel_dimension=(15, 260, 346),
                mlp_layers=(1,30,30,1),
                **kwargs):

    cfg = get_default_network_config()

    repr_block = QuantizationLayer(dim=voxel_dimension,
                                   mlp_layers=mlp_layers)

    block_type, layers, channels, name = resnet_spec[kwargs['resnet_layers']]
    backbone_net = ResNetBackbone(block_type, layers)
    head_net = DeconvHead(
        channels[-1], cfg["num_deconv_layers"],
        cfg["num_deconv_filters"], cfg["num_deconv_kernel"],
        cfg["final_conv_kernel"], kwargs['num_classes'], cfg["depth_dim"]
    )
    pose_net = ResPoseNet(repr_block,
                          backbone_net,
                          head_net)

    init_pose_net(pose_net, name)
    return pose_net


class AugmentationModule(nn.Module):
    def __init__(self, sensor_size, scale_range=(.7,1.1), rot_range=(-np.pi/3, np.pi/3)):
        nn.Module.__init__(self)
        H, W = sensor_size
        self.warper = tgm.HomographyWarper(H, W)
        self.scale_range = scale_range
        self.rot_range = rot_range

        self.homography = None
        self.angles = None
        self.flips = None
        self.scales = None

    def forward(self, x):
        if self.homography is None:
            self.homography = x.new_full([x.shape[0], 3, 3], fill_value=0).to(x.device)
            self.homography[:,0,0] = 1
            self.homography[:,1,1] = 1
            self.homography[:,2,2] = 1
            self.angles = torch.zeros(x.shape[0]).to(x.device)
            self.scales = torch.zeros(x.shape[0]).to(x.device)
            self.flips = torch.zeros(x.shape[0]).to(x.device)

        # sample random scale and rotation and flip
        self.angles.uniform_(*self.rot_range)
        self.scales.uniform_(*self.scale_range)
        self.flips.bernoulli_()

        homography = self.homography.clone()

        # construct new homography
        homography[:,0,0] = -(2*self.flips-1)*self.scales*torch.cos(self.angles)
        homography[:,0,1] = (2*self.flips-1)*self.scales*torch.sin(self.angles)
        homography[:,1,0] = self.scales*torch.sin(self.angles)
        homography[:,1,1] = self.scales*torch.cos(self.angles)

        output = self.warper(x, homography)

        return output, homography


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

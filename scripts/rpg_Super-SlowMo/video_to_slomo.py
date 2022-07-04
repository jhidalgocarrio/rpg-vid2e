#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
import numpy as np
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import model
import dataloader
import platform
import tqdm
import pandas as pd
import cv2

# For parsing commandline arguments



class Converter:
    def __init__(self,
                 device,
                 batch_size,
                 file=None,
                 fps=30,
                 sf=None,
                 keyframe=None,
                 img_range=None,
                 downsample_factor=1,
                 adaptive=False,
                 meta=None,
                 resize_dim=None):

        checkpoint = os.path.join(os.path.dirname(__file__), "..", "SuperSloMo.ckpt")
        self.adaptive = adaptive
        self.keyframe = keyframe
        self.img_range = img_range
        self.fps = fps
        self.downsample_factor = downsample_factor
        self.file = file

        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.sf = sf
        self.device = device

        # Initialize transforms
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        mean = [0.429, 0.431, 0.397]
        std = [1, 1, 1]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        negmean= [x * -1 for x in mean]
        self.negmean = torch.Tensor([x * -1 for x in mean]).to(self.device).view(3, 1, 1)
        self.std = std
        revNormalize = transforms.Normalize(mean=negmean, std=std)

        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.TP = transforms.Compose([revNormalize])

        # Initialize model
        self.flowComp = model.UNet(6, 4)
        self.flowComp.to(device)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = model.UNet(20, 5)
        self.ArbTimeFlowIntrp.to(device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        self.flowBackWarp = None

        self.dict1 = torch.load(checkpoint, map_location='cpu')
        self.ArbTimeFlowIntrp.load_state_dict(self.dict1['state_dictAT'])
        self.flowComp.load_state_dict(self.dict1['state_dictFC'])

        self.total_frames = []
        self.timestamps = []

        self.meta = meta
        self.resize_dim = resize_dim
        if meta is None:
            videoFrames = dataloader.Video(root=self.file,
                                           transform=self.transform,
                                           keyframe=self.keyframe,
                                           img_range=self.img_range,
                                           fps=self.fps,
                                           downsample_factor=self.downsample_factor)

            self.origDim = videoFrames.origDim

            self.videoFramesloader = torch.utils.data.DataLoader(videoFrames,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)

            self.videoFramesloader = self.videoFramesloader.__iter__()
            dim = (videoFrames.dim[1], videoFrames.dim[0])
            self.num_frames = len(videoFrames)
        else:
            print("Loading from video.")
            self.origDim = (int(meta["meta"]["video"]["@height"]), int(meta["meta"]["video"]["@width"]))

            self.fps = eval(meta["meta"]["video"]["@avg_frame_rate"])
            self.videoFramesloader = meta["it"]

            self.num_frames = int(meta["meta"]["video"]["@nb_frames"])
            dim = int(np.ceil(self.origDim[0] / 32)) * 32, int(np.ceil(self.origDim[1] / 32)) * 32

        # get video dim
        if self.resize_dim is not None:
            self.origDim = resize_dim
            dim = int(np.ceil(self.origDim[0] / 32)) * 32, int(np.ceil(self.origDim[1] / 32)) * 32
            print("Resizing to %s x %s" % dim)

        self.flowBackWarp = model.backWarp(dim[1], dim[0], self.device)
        self.flowBackWarp.to(self.device)

    #@profile
    def upsample_adaptive(self, I0, I1, time0, time1, F_0_1, F_1_0, total_frames, timestamps):
        B, _, _, _ = F_0_1.shape

        flow_mag_0_1_max, _ = F_0_1.pow(2).sum(1).pow(.5).view(B,-1).max(-1)
        flow_mag_1_0_max, _ = F_1_0.pow(2).sum(1).pow(.5).view(B,-1).max(-1)

        flow_mag_max, _ = torch.stack([flow_mag_0_1_max, flow_mag_1_0_max]).max(0)
        flow_mag_max = max([np.ceil(flow_mag_max.cpu().numpy()).astype(int), 1])

        for i in range(B):
            #print("Upsampling by adaptive factor: %s" % flow_mag_max[i])
            for intermediateIndex in range(1, flow_mag_max[i]):
                t = float(intermediateIndex) / flow_mag_max[i]
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
                
                g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

                intrpOut = self.ArbTimeFlowIntrp(
                    torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                Ft_p_norm = Ft_p[i] - self.negmean

                total_frames += [Ft_p_norm]
                timestamps += [(time0[i] + t * (time1[i] - time0[i])).item()]

    #@profile
    def upsample_fixed(self, I0, I1, time0, time1, F_0_1, F_1_0, total_frames, timestamps):
        for intermediateIndex in range(1, self.sf):
            t = float(intermediateIndex) / self.sf
            temp = -t * (1 - t)
            fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

            intrpOut = self.ArbTimeFlowIntrp(
                torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

            wCoeff = [1 - t, t]

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                    wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            # Save intermediate frame
            # Save intermediate frame
            for batchIndex in range(Ft_p.shape[0]):
                Ft_p_norm = Ft_p[batchIndex] - self.negmean

                total_frames += [Ft_p_norm]
                timestamps += [(time0[batchIndex] + t * (time1[batchIndex] - time0[batchIndex])).item()]

    def resize_img(self, img, dim, center=None):
        # center crop
        H, W, C = img.shape
        H_d, W_d = dim

        scale_x = W_d/W
        scale_y = H_d/H

        # take larger scale
        scale = max([scale_x, scale_y])
        H_rescaled = int(H*scale)
        W_rescaled = int(W*scale)

        img = cv2.resize(img, (W_rescaled, H_rescaled))

        H, W, C = img.shape

        # crop around center
        if center is None:
            center = (0, 0)

        if H > H_d:
            py = (H - H_d) // 2
            img = img[py+center[0]:-py+center[0], ...]
        if W > W_d:
            px = (W - W_d) // 2
            img = img[:, px+center[1]:-px+center[1], ...]

        H, W, C = img.shape
        pad_x = (int(np.ceil(W/32)*32)-W)//2
        pad_y = (int(np.ceil(H/32)*32)-H)//2

        img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")

        return img, (H, W)

    def set_files(self, file):
        videoFrames = dataloader.Video(root=file,
                                       transform=self.transform,
                                       keyframe=self.keyframe,
                                       img_range=self.img_range,
                                       fps=self.fps,
                                       downsample_factor=self.downsample_factor)

        self.origDim = videoFrames.origDim

        self.videoFramesloader = torch.utils.data.DataLoader(videoFrames,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)

        self.videoFramesloader = self.videoFramesloader.__iter__()
        dim = (videoFrames.dim[1], videoFrames.dim[0])
        self.num_frames = len(videoFrames)    

    def __iter__(self):
        # Load data
        with torch.no_grad():
            for i in tqdm.tqdm(range(self.num_frames)):
                if self.meta is None:
                    try:
                        (frame0, frame1), (time0, time1) = next(self.videoFramesloader)
                    except StopIteration:
                        break
                    orig_dim = None
                else:
                    try:
                        frame1 = next(self.videoFramesloader)
                    except StopIteration:
                        break
                    if self.resize_dim is not None:
                        frame1, orig_dim = self.resize_img(frame1, self.resize_dim)

                    if i == 0:
                        frame0 = self.transform(frame1)[None, ...]
                        time0 = torch.Tensor([float(i) / self.fps])
                        continue

                    time1 = torch.Tensor([float(i) / self.fps])
                    frame1 = self.transform(frame1)[None, ...]

                total_frames = []
                timestamps = []

                I0 = frame0.to(self.device)
                I1 = frame1.to(self.device)

                flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:, :2, :, :]
                F_1_0 = flowOut[:, 2:, :, :]

                # Save reference frames in output folder
                for batchIndex in range(frame0.shape[0]):
                    total_frames += [self.TP(I0[batchIndex])]
                    timestamps += [(time0[batchIndex]).item()]

                # Generate intermediate frames
                if self.adaptive:
                    self.upsample_adaptive(I0, I1, time0, time1, F_0_1, F_1_0, total_frames, timestamps)
                else:
                    self.upsample_fixed(I0, I1, time0, time1, F_0_1, F_1_0, total_frames, timestamps)


                sorted_indices = np.argsort(timestamps)

                total_frames = torch.stack([total_frames[j] for j in sorted_indices])

                if orig_dim is not None:
                    total_frames = self.crop_to(total_frames, orig_dim)

                timestamps = [timestamps[i] for i in sorted_indices]
                total_frames = _to_numpy_image(total_frames)

                frame0 = frame1
                time0 = time1

                yield total_frames, timestamps

    #@profile
    def crop_to(self, img, dim):
        H, W = dim
        _, c, h, w  = img.shape

        py = (h-H)//2
        px = (w-W)//2
        img = img[:, :, py:H+py, px:W+px]

        return img

#@profile
def _to_numpy_image(img):
    img = np.clip(255 * img.cpu().numpy(), 0, 255).astype(np.uint8)
    img = np.transpose(img, (0, 2,3,1))
    return img

def FLAGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe')
    parser.add_argument("--video", type=str, required=False, help='path of video to be converted')
    parser.add_argument("--checkpoint", type=str,
                        default="/home/dani/code/catkin_ws/src/rpg_vid2e/scripts/super_slow_mo_upsampling/SuperSloMo.ckpt",
                        required=False, help='path of checkpoint for pretrained model')
    parser.add_argument("--fps", type=float, default=30, help='specify fps of output video. Default: 30.')
    parser.add_argument("--sf", type=int, required=False, default=3,
                        help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
    parser.add_argument("--output", type=str, default="output.mp4",
                        help='Specify output file name. Default: output.mp4')
    parser.add_argument("--pandas_dataframe", type=str,
                        default="/home/dani/code/catkin_ws/src/rpg_vid2e/dataset/garfield/images_0001.pkl",
                        help='pandas dataframe where frames are saved.')
    parser.add_argument("--device", type=str, default="cpu", help='type of device to be used.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = FLAGS()
    converter = Converter(args.checkpoint, args.sf, args.device, args.batch_size)
    converter.upsample_1(args.pandas_dataframe)

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=5)
cls = "wheelchair"
idx = "0008"
factor = 16
x, y = 110,100


f1 = f"/media/dani/data/ncaltech/temp_downsampled_factor_{factor}_upsampled_factor_{factor}/{cls}/image_{idx}.npz"
f2 = f"/media/dani/data/ncaltech/temp_downsampled_factor_{factor}/{cls}/image_{idx}.npz"
f3 = f"/media/dani/data/ncaltech/temp/{cls}/image_{idx}.npz"

dt = .3/500
n_samples = 500

events1 = np.load(f1)
events2 = np.load(f2)
events3 = np.load(f3)

events = [events1, events2, events3]
labels = [f"down_and_upsampled_{factor}", f"downsampled_{factor}", "ground_truth"]

data = {}
for label, e in zip(labels, events):
    label_data = {}
    t = e["t"]
    label_data["num_events"] = len(t)
    t_min = 0
    t_max = 0.3
    event_rate = []
    for i in range(n_samples):
        t0 = (t_max - t_min) * (i + 1) / n_samples

        num_events = len(t[(t <= t0) & (t > t0 - dt)])
        event_rate += [num_events / dt]
    label_data["rates"] = event_rate

    data[label] = label_data




for label, datum in data.items():
    ax[0].plot(datum["rates"], label=label)

ax[0].legend()

import glob
f1 = f"/media/dani/data/ncaltech/temp_downsampled_factor_{factor}_upsampled_factor_{factor}/{cls}/image_{idx}"
f2 = f"/media/dani/data/ncaltech/temp_downsampled_factor_{factor}/{cls}/image_{idx}"
f3 = f"/media/dani/data/ncaltech/temp/{cls}/image_{idx}"

t1 = np.genfromtxt(f1+ "/timestamps.txt")
t2 = np.genfromtxt(f2+ "/timestamps.txt")
t3 = np.genfromtxt(f3+ "/timestamps.txt")
ts = [t1,t2,t3]

images1 = sorted(glob.glob(f1+"/images/*.png"))
images2 = sorted(glob.glob(f2+"/images/*.png"))
images3 = sorted(glob.glob(f3+"/images/*.png"))
images = [images1, images2, images3]

import cv2
i1 = [cv2.imread(i)[y,x,0].astype("float32") for i in images1]
i2 = [cv2.imread(i)[y,x,0].astype("float32") for i in images2]
i3 = [cv2.imread(i)[y,x,0].astype("float32") for i in images3]
iss = [i1,i2,i3]

for j, (label, i) in enumerate(zip(labels, iss)):
    ax[1].plot(ts[j], i, label=label)

ax[1].legend()

xyps = [e["xyp"] for e in events]
masks = [((xyp[:,0]==x)&(xyp[:,1]==y)) for xyp in xyps]

data = {}
for j, (label, events) in enumerate(zip(labels, events)):
    label_data = {}
    t = events["t"][masks[j]]
    t_min = 0
    t_max = 0.3
    event_rate = []
    for i in range(n_samples):
        t0 = (t_max - t_min) * (i + 1) / n_samples

        num_events = len(t[(t <= t0) & (t > t0 - dt)])
        event_rate += [num_events]
    label_data["rates"] = event_rate

    data[label] = label_data

for label, datum in data.items():
    ax[2].plot(datum["rates"], label=label)

ax[2].legend()

# upsample downsampled with bilinear interpolation
new_image2 = []
new_t2 = []
for t in t3:
    idx_left = np.searchsorted(t2, t)
    r = (t - t2[idx_left-1])/(t2[idx_left] - t2[idx_left-1])
    image = int(idx_left>0)*(1-r)*cv2.imread(images2[idx_left-1]).astype("float32")+(r)*cv2.imread(images2[idx_left]).astype("float32")
    image = np.clip(image,0,255).astype("uint8")
    new_image2.append(image)
    new_t2.append(t)

errs13 = []
errs23 = []
psnrs23 = []
psnrs13 = []
for (im1, im2, im3) in zip(images1, new_image2, images3):
    im1 = cv2.imread(im1).astype("float32")
    im3 = cv2.imread(im3).astype("float32")
    im2 = im2.astype("float32")

    err13 = np.abs(im1-im3).mean()
    err23 = np.abs(im2-im3).mean()

    errs13.append(err13)
    errs23.append(err23)

    mse13 = (np.abs(im1-im3)**2).mean()
    mse23 = (np.abs(im2-im3)**2).mean()
    eps=1e-1
    psnr13 = 20*np.log(np.max(im3)/(eps+np.sqrt(mse13)))
    psnr23 = 20*np.log(np.max(im3)/(eps+np.sqrt(mse23)))
    psnrs13.append(psnr13)
    psnrs23.append(psnr23)

ax[3].plot(errs23, label="gt_downsampled")
ax[3].plot(errs13, label="gt_downsampled_upsampled")
ax[3].legend()

ax[4].plot(psnrs23, label="gt_downsampled")
ax[4].plot(psnrs13, label="gt_downsampled_upsampled")
ax[4].legend()


plt.show()
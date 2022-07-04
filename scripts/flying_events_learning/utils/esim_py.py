"""
Implements event generation on pytorch
"""
import torch
import numpy as np

cuda_timers = {}

class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))


class ESIM(torch.nn.Module):
    def __init__(self, Cp, Cn, refractory_period):
        torch.nn.Module.__init__(self)
        self.Cp = Cp
        self.Cn = Cn
        self.refractory_period = refractory_period

    def forward(self, timestamps, video):
        # video: T x C x H x W
        # outputs events: N x 4
        to_np = False
        if type(video) is np.ndarray:
            to_np = True

            video = torch.from_numpy(video)
            timestamps = torch.from_numpy(timestamps)

        events = torch.zeros(0,4,device=video.device,dtype=torch.float32)

        t0, frame0 = timestamps[0], video[0]
        reference_values = frame0#.clone()
        H, W = frame0.shape
        changed_frame = []

        for i in range(1, len(video)):

            frame = video[i]
            t = timestamps[i]

            # detect when a frame changes completely
            n_ev = (((frame - reference_values)/self.Cp).abs().int()>0).int().sum()

            if n_ev >= 0.7*H*W:
                changed_frame += [[timestamps[i-1].item(), t.item()]]

            new_events = self.generate_events1(frame, frame0, t, t0, reference_values)
            #print(len(new_events))
            events = torch.cat([events, new_events],0)


            t0, frame0 = t, frame

        events = events[events[:,2].argsort()]

        if to_np:
            events = events.numpy()

        return events, changed_frame

    def find_crossings(self, data):
        f0, f1, cl, p = data["frame0"], data["frame1"], data["crossing_level"], data["polarity_map"]
        pos = p>0
        return (pos & (cl > f0) & (cl <= f1)) | ((~pos) & (cl < f0) & (cl >= f1))

    def filter_data(self, data, mask):
        for k in data:
            data[k] = data[k][mask]

    def generate_events1(self, frame1, frame0, t1, t0, reference_values):
        assert (frame1.shape == frame0.shape)
        delta_t = t1 - t0
        assert (delta_t > 0)

        # convert both images to linear image
        H, W = frame1.shape
        frame1 = frame1.view(-1)
        frame0 = frame0.view(-1)
        reference_values = reference_values.view(-1)

        global_index = torch.arange(0,H*W, dtype=torch.long, device=frame0.device)

        p = (frame1 >= frame0).float()  # polarity has value +1, 0
        polarity_map = p * self.Cp - (1 - p) * self.Cn

        intensity_change = (frame1 - reference_values)
        num_events_float = intensity_change / polarity_map
        num_events = (num_events_float+1e-3).int()

        tolerance_map = (frame1 - frame0).abs() > 1e-6

        temp = delta_t/(frame1 - frame0)
        B = t0 + (reference_values - frame0) * temp
        A = polarity_map * temp
        reference_values += (num_events.float() * polarity_map)

        e_list = []
        N_max = num_events.max()

        global_index = global_index[tolerance_map]
        num_events = num_events[tolerance_map]

        for n in range(1, N_max+1):
            coords = global_index[n<=num_events]
            t = A[coords]*n + B[coords]
            y = (coords // W).float()
            x = (coords % W).float()
            new_events = torch.stack([x, y, t, p[coords]])
            e_list += [new_events]

        events = torch.empty(0,4,dtype=frame1.dtype, device=frame1.device) if len(e_list) == 0 else torch.cat(e_list, 1).permute(1, 0)

        return events

    def generate_events(self, frame1, frame0, t1, t0, reference_values, last_timestamps):
        assert (frame1.shape == frame0.shape)
        delta_t = t1 - t0
        assert (delta_t > 0)

        # convert both images to linear image
        C, H, W = frame1.shape
        frame1 = frame1.view(-1)
        frame0 = frame0.view(-1)
        reference_values = reference_values.view(-1)
        last_timestamps = last_timestamps.view(-1)

        tol = 1e-6
        tolerance_mask = (frame1-frame0).abs()>tol
        polarity = (frame1>=frame0).float()  # polarity has value +1, 0

        curr_crossing_level = reference_values.clone()
        events = torch.zeros(0,4,dtype=frame0.dtype, device=frame0.device)

        crossings = torch.arange(len(frame0), dtype=torch.long, device=frame0.device)

        data = {
            "polarity_map": polarity * self.Cp - (1 - polarity) * self.Cn,
            "frame0": frame0,
            "frame1": frame1,
            "crossing_level": curr_crossing_level,
            "global_index": crossings
        }

        data = self.filter_data(data, tolerance_mask)

        while len(data["polarity_map"])>0:
            data["crossing_level"] += data["polarity_map"]
            still_crossings = self.find_crossings(data)  # len N_k

            data = self.filter_data(data, still_crossings)

            # update ref with global index, crossing level has a local index
            reference_values[data["global_index"]] = data["crossing_level"]

            # find where refractory period is exceeded, t of the last events needs to be accessed with the global index
            t_events = t0 + (data["crossing_level"] - data["frame0"]) * delta_t / (data["frame1"] - data["frame0"])
            dt_since_last = t_events - last_timestamps[data["global_index"]]
            triggered = dt_since_last > self.refractory_period

            # only update triggered timestamps and generate triggered events
            last_timestamps[data["global_index"][triggered]] = t_events[triggered]

            # generate events
            y = (data["global_index"][triggered] // W).float()
            x = (data["global_index"][triggered] % W).float()
            t = t_events[triggered]
            p = (data["polarity_map"][triggered] >0).float()

            new_events = torch.stack([x,y,t,p]).permute(1,0)
            events = torch.cat([events, new_events], 0)

        return events, reference_values, last_timestamps

def load_images():
    video_frames_path = "/media/dani/data/tmp/video"
    eps = 1e-3
    fps = 30
    imgs = []
    timestamps = []
    for i in range(700, 751):  # ,751):
        img = cv2.imread("%s/frame/cam0/img_%s.png" % (video_frames_path, str(i).zfill(4)))
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[
            None, ...]#10:10+1,37:37+1]#460:461, 380:381]
        imgs.append(img)
        timestamps.append((i - 699) / fps)
    video = np.stack(imgs)  # .cuda()
    timestamps = np.stack(timestamps)
    video = np.log(video.astype(np.float32)/255+eps)
    return video, timestamps

if __name__ == '__main__':
    import cv2
    import numpy as np
    C = .15
    esim = ESIM(C, C, .00)#.cuda()

    video, timestamps = load_images()

    # convert to log
    video = torch.from_numpy(video)#.cuda()
    timestamps = torch.from_numpy(timestamps)#.cuda()

    with CudaTimer("bla"):
        events = esim.forward(timestamps, video)
    print(events)
    #print(events.cpu().numpy())
    #print(cuda_timers)
    #timestamps = timestamps.cpu()
    #log_video = log_video.cpu()
    #events = events.cpu()
#
    np.savetxt("/media/dani/data/tmp/events_python.txt", events.numpy(), fmt="%i %i %.8f %i")
    #
    #import matplotlib.pyplot as plt
    #plt.plot(timestamps.numpy().ravel(), log_video.numpy().ravel(), "g")
#
    #i1 = log_video.numpy().ravel()[0]
    #for i in range(50):
    #    plt.plot([timestamps[0],timestamps[-1]], [i1-i*C,i1-i*C], 'k')
    #for i in range(50):
    #    plt.plot([timestamps[0],timestamps[-1]], [i1+i*C,i1+i*C], 'k')
#
    #for e in events:
    #    if e[-1] > 0:
    #        plt.plot([e[-2], e[-2]],[-10,10],"b" )
    #    else:
    #        plt.plot([e[-2], e[-2]],[-10,10],"r" )
#
    #plt.ylim([log_video.min().numpy().ravel(), log_video.max().numpy().ravel()])
    #print(events.numpy())
    #plt.show()



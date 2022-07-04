import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    def __init__(self, dataset, flags, sequential=False, voxel_grid=False):
        self.device = flags.device
        split_indices = list(range(len(dataset)))
        if sequential:
            sampler = torch.utils.data.sampler.SequentialSampler(split_indices)
        else:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        collate = default_collate if voxel_grid else collate_events
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, sampler=sampler,
                                                  num_workers=flags.num_workers, pin_memory=flags.pin_memory,
                                                  collate_fn=collate)

    def __iter__(self):
        for data in self.loader:
            d = {}
            for k, v in data.items():
                if type(v) is dict:
                   v_temp = {}
                   for k_, v_ in v.items():
                       v_temp[k_] = v_.to(self.device, non_blocking=True)
                   v = v_temp
                elif not type(v) is list:
                    v = v.to(self.device, non_blocking=True)
                d[k] = v
            yield d

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    event_key=None
    for k in data[0].keys():
        if "event" in k:
            event_key = k
            break

    events = [d.pop(event_key) for d in data]
    events = np.concatenate([np.concatenate([e, i*np.ones((len(e),1), dtype=np.float32)],1) for i, e in enumerate(events)],0)
    data = default_collate(data)

    xy = torch.from_numpy(events[:,[0,1]].astype("int16"))
    b = torch.from_numpy(events[:,[4]].astype("int8"))

    t = torch.from_numpy(events[:,2:3])
    p = torch.from_numpy(events[:,3:4].astype("int8"))
    
    data[event_key] = {"p": p, "xy": xy, "t": t, "b": b}

    return data

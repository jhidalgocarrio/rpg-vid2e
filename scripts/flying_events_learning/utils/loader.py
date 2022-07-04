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
                    v = {k_: v_.to(self.device) for k_, v_ in v.items()}
                elif not type(v) is list:
                    v = v.to(self.device)
                d[k] = v
            yield d

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    event_keys=[]
    for d in data:
        for k in d.keys():
            if "image" in k or "events" in k:
                event_keys.append(k)
    event_keys = sorted(event_keys)
    events = []
    for d in data:
        for k in event_keys:
            if k in d:
                events.append(d.pop(k))

    events = np.concatenate([np.concatenate([e, i*np.ones((len(e),1), dtype=np.float32)],1) for i, e in enumerate(events)],0)
    data = default_collate(data)

    event_dict = {"xypb": events[:,[0,1,3,4]].astype("uint8"), "t": events[:,[2]].astype("float32")}
    event_dict = {k: torch.from_numpy(v) for k,v in event_dict.items()}
    data["events"] = event_dict
    return data

import os
import json
import numpy as np
import pandas as pd
from einops import rearrange

import torch
from torch.utils.data import Dataset, DataLoader
from gp import train_sample_gp 

def wearable_loader(args, mode):
    assert mode in ['train', 'valid', 'test']

    batch_size = args.batch_size
    workers = args.workers

    dataset = WearableDataset(args.data_path, mode, args.n_samples)
    loader = DataLoader(
        dataset, collate_fn=collate_fn, batch_size=batch_size,
        shuffle=True, num_workers=workers,
    )
    return loader

# the same as torch default collate_fn, but removes None (for valid and test)
def collate_fn(batch):
    x, tgt = zip(*batch)
    # remove None
    x = [x_ for x_ in x if x_ is not None]
    tgt = [tgt_ for tgt_ in tgt if tgt_ is not None]

    x = torch.stack(x, dim=0)
    tgt = torch.stack(tgt, dim=0)
    return x, tgt

'''
torch dataloader and dataset for wearable
'''

class WearableDataset(Dataset):
    def __init__(self, data_path, mode, n_samples, num_iters=1000):
        self.mode = mode
        self.n_samples = n_samples
        self.num_iters = num_iters
        self.data_path = data_path
        self.user_ids = json.load(open(os.path.join(data_path, 'user_ids.json'))) 

        # get random users for train-valid / test split
        train_frac = 0.8
        torch.manual_seed(519)
        split = int(train_frac * len(self.user_ids))
        shuffle_idx = torch.randperm(len(self.user_ids))
    
        if mode in ['train', 'valid']: split_idx = shuffle_idx[:split]
        else: split_idx = shuffle_idx[split:]
        self.user_ids = [self.user_ids[i] for i in split_idx]

        # each user has dataframes for: labels, heart_rate, motion
        print(f'loading {mode} data')
        self.user_data = {user_id : self.load_data(user_id) for user_id in self.user_ids}
        print('done')

        # define time intervals that are used in train / valid for each user
        torch.manual_seed(813)
        self.user_t_idx = {}
        if mode in ['train', 'valid']:
            for user_id in self.user_ids:
                T = len(self.user_data[user_id]['labels']) - 1
                t_idx = torch.randperm(T)

                t_split = int(0.9 * T)
                if mode == 'train': t_idx = t_idx[:t_split]
                else: t_idx = t_idx[t_split:]

                self.user_t_idx[user_id] = t_idx

        # follow similar pattern for test
        else:
            for user_id in self.user_ids:
                t_idx = torch.arange(len(self.user_data[user_id]['labels']) - 1)
                self.user_t_idx[user_id] = t_idx

        # create a map from idx : (user_id, t_idx) for valid and test  
        self.idx2t = {}
        idx = 0
        for user_id in self.user_ids:
            for t_idx in self.user_t_idx[user_id]:
                self.idx2t[idx] = (user_id, t_idx.item())
                idx += 1
    
    def __len__(self): return len(self.idx2t)
    
    # load all data dataframes for a user
    def load_data(self, user_id):
        data = []
        folders = ['labels', 'heart_rate', 'motion']

        for folder in folders:
            fname = os.path.join(self.data_path, user_id + '_' + folder + '.parquet')
            df = pd.read_parquet(fname)

            # normalize all columns exept labels and time
            if folder != 'labels':
                normalize = lambda x : (x - np.mean(x, axis=0)) / np.std(x, axis=0)
                df[df.columns[1:]] = df[df.columns[1:]].apply(normalize)
            data.append(df)
        
        return {folders[i] : data[i] for i in range(len(folders))}

    # sample a single valid time point 
    def sample_t_train(self, user_id):
        t_idx = self.user_t_idx[user_id]
        return t_idx[torch.randint(len(t_idx), (1,))].item()

    def __getitem__(self, idx):
        # get user_id and time window
        user_id, t = self.idx2t[idx]
        data = self.user_data[user_id]

        label = data['labels']['label'].iloc[t]
        if label != 5: label += 1

        # get start and end indices
        t_start = data['labels']['time'].iloc[t]
        t_end = data['labels']['time'].iloc[t+1]

        # get heart rate and motion data in window
        heart_rate = data['heart_rate'][(data['heart_rate']['time'] >= t_start) & (data['heart_rate']['time'] < t_end)]
        motion = data['motion'][(data['motion']['time'] >= t_start) & (data['motion']['time'] < t_end)]
        if len(heart_rate) == 0 and len(motion) == 0: return None, None

        # convert to numpy -> torch
        hr_t = torch.tensor( heart_rate['time'].values ).float()
        m_t = torch.tensor( motion['time'].values ).float()
        heart_rate = torch.tensor( heart_rate['heart_rate'].values ).float()[:, None]
        motion = torch.tensor( motion[['x', 'y', 'z']].values ).float()

        if heart_rate.shape[0] <= 2 or motion.shape[0] <= 2: return None, None

        # sample evenly spaced points from gaussian process
        hr_sample = self.sample_gp(hr_t, heart_rate, self.n_samples)
        m_sample = self.sample_gp(m_t, motion, self.n_samples)
        x = torch.cat((hr_sample, m_sample), dim=-1)
        x = rearrange(x, 'n d -> d n')

        return x, torch.tensor(label).long()

    # sample evenly spaced points from gaussian process
    def sample_gp(self, t, y, n_samples):
        # if std dev is 0, return mean for all samples
        if y.std(dim=0).sum() < 1e-6:
            return torch.zeros(n_samples, y.shape[1]) + y.mean(dim=0)

        # if motion, take a subset of the data
        if y.shape[1] == 3:
            rand_idx = torch.randperm(y.shape[0])[:256]
            rand_idx, _ = torch.sort(rand_idx)

            y = y[rand_idx]
            t = t[rand_idx]

        # normalize data before sampling
        mean, std = y.mean(dim=0), y.std(dim=0)
        y_ = (y - mean) / std

        # make x on [0, 1]
        t = (t - t[0]) / (t[-1] - t[0])
        t = torch.stack([t for _ in range(y.shape[1])], dim=1)

        # make x_test
        t_test = torch.linspace(0, 1, n_samples)
        t_test = torch.stack([t_test for _ in range(y.shape[1])], dim=1)

        y_out = train_sample_gp(t, y_, t_test, self.mode)

        # renormalize before outputting
        y_out = y_out * std + mean

        return y_out

from args import get_args
if __name__ == '__main__':
    data_path = 'data'
    args = get_args()
    loader = wearable_loader(args, 'train')

    for i, (x, tgt) in enumerate(loader):
        print(x.shape, tgt.shape)

    print('done')
    quit()
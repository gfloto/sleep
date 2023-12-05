import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import gpytorch
from gp import train_sample_gp 

def wearable_loader(data_path, mode, batch_size, workers):
    assert mode in ['train', 'test']

    dataset = WearableDataset(data_path, mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return loader

'''
torch dataloader and dataset for wearable
'''

class WearableDataset(Dataset):
    def __init__(
        self, data_path, mode, n_samples=64, 
        train_frac=0.8, num_iters=1000
    ):
        self.mode = mode
        self.n_samples = n_samples
        self.num_iters = num_iters
        self.data_path = data_path
        self.user_ids = json.load(open(os.path.join(data_path, 'user_ids.json'))) 

        # get random users for train/test split
        torch.manual_seed(519)
        split = int(train_frac * len(self.user_ids))
        shuffle_idx = torch.randperm(len(self.user_ids))
    
        if mode == 'train': split_idx = shuffle_idx[:split]
        else: split_idx = shuffle_idx[split:]
        self.user_ids = [self.user_ids[i] for i in split_idx]

        # each user has dataframes for: labels, heart_rate, motion
        print('loading data')
        self.user_data = {user_id : self.load_data(user_id) for user_id in self.user_ids}
        print('done')
    
    def __len__(self): return self.num_iters
    
    # load all data dataframes for a user
    def load_data(self, user_id):
        data = []
        folders = ['labels', 'heart_rate', 'motion']

        for folder in folders:
            fname = os.path.join(self.data_path, user_id + '_' + folder + '.parquet')
            df = pd.read_parquet(fname)

            # normalize all columns
            normalize = lambda x : (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            df[df.columns] = df[df.columns].apply(normalize)
            data.append(df)
        
        return {folders[i] : data[i] for i in range(len(folders))}

    def __getitem__(self, _):
        while True:
            idx = np.random.randint(len(self.user_ids))

            user_id = self.user_ids[idx]
            data = self.user_data[user_id]

            # sample window of time (30s)
            t = np.random.randint(len(data['labels']['time']) - 1)

            time = data['labels']['time'][t]
            label = data['labels']['label'][t]

            # get start and end indices
            t_start = data['labels']['time'][t]
            t_end = data['labels']['time'][t + 1]

            # get heart rate and motion data in window
            heart_rate = data['heart_rate'][(data['heart_rate']['time'] >= t_start) & (data['heart_rate']['time'] < t_end)]
            motion = data['motion'][(data['motion']['time'] >= t_start) & (data['motion']['time'] < t_end)]

            if len(heart_rate) == 0 and len(motion) == 0: continue

            # convert to numpy -> torch
            hr_t = torch.tensor( heart_rate['time'].values ).float()
            heart_rate = torch.tensor( heart_rate['heart_rate'].values ).float()[:, None]


            m_t = torch.tensor( motion['time'].values ).float()
            motion = torch.tensor( motion[['x', 'y', 'z']].values ).float()

            if heart_rate.shape[0] <= 2: continue
            if motion.shape[0] <= self.n_samples: continue
            break

        # sample evenly spaced points from gaussian process
        hr_sample = self.sample_gp(hr_t, heart_rate, self.n_samples)
        m_sample = self.sample_gp(m_t, motion, self.n_samples)

        return {
            'label' : torch.tensor(label).long(),
            'heart_rate' : hr_sample,
            'motion' : m_sample,
        }

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

        y_out = train_sample_gp(t, y_, t_test)

        # renormalize before outputting
        y_out = y_out * std + mean

        return y_out

if __name__ == '__main__':
    data_path = 'data'
    loader = wearable_loader(data_path, 'train', batch_size=1, workers=0)

    print(len(loader))
    for i, data in enumerate(loader):
        label = data['label'] 
        heart_rate = data['heart_rate']
        motion = data['motion']

        # print motion
        x = torch.cat((heart_rate, motion), dim=2)
        print(x.shape)

    print('done')
    quit()
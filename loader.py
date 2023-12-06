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

    dataset = WearableDataset(args.data_path, mode)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers,
    )
    return loader

'''
torch dataloader and dataset for wearable
'''

class WearableDataset(Dataset):
    def __init__(self, data_path, mode):
        self.mode = mode
        self.data_path = data_path
        self.user_ids = json.load(open(os.path.join(data_path, 'user_ids.json'))) 

        # get random users for train / valid / test split
        split = {
            'train' : 0.8,
            'valid' : 0.1,
            'test' : 0.1,
        }

        torch.manual_seed(9) # 9 is best (7 too) 
        shuffle_idx = torch.randperm(len(self.user_ids))

        split_1 = int( split['train'] * len(self.user_ids) )
        split_2 = int( (split['train'] + split['valid']) * len(self.user_ids) )

        if mode == 'train': split_idx = shuffle_idx[:split_1]
        elif mode == 'valid': split_idx = shuffle_idx[split_1:]
        #elif mode == 'test': split_idx = shuffle_idx[split_2:]
        else: raise ValueError(f'invalid mode: {mode}')

        self.user_ids = [self.user_ids[i] for i in split_idx]

        # load data
        print(f'loading {mode} data')
        load_path = os.path.join(data_path, 'wearable.parquet')
        self.data = pd.read_parquet(load_path)

        # take subset of users
        self.data = self.data[self.data['user_id'].isin(self.user_ids)]
        self.data.index = range(len(self.data))

        # if label > 0, set to 1
        self.data['label'] = self.data['label'].apply(lambda x: 1 if x > 0 else x)

        # evenly sample label
        self.label2rows = {}
        for label in range(2):
            rows = self.data[self.data['label'] == label].index.values
            self.label2rows[label] = rows
    
    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        # uniformly sample train data across labels
        if self.mode == 'train':
            label = torch.randint(0, 2, (1,)).item()
            r = torch.rand(1).item()
            if r < 0.575: label = 0
            else: label = 1
            rows = self.label2rows[label]

            rand_idx = torch.randint(0, len(rows), (1,)).item()
            idx = rows[rand_idx]

        # get label
        row = self.data.iloc[idx]
        label = torch.tensor(row['label']).float()

        # read in gp data
        gp_path = row['gp-path']
        gp_pred = torch.load(gp_path)
        mean = gp_pred['mean']
        std = gp_pred['std']

        # combine and return 
        x = torch.cat([mean, std], dim=0)
        return x, label

from args import get_args
if __name__ == '__main__':
    data_path = 'data'
    args = get_args()
    loader = wearable_loader(args, 'train')

    for i, (x, tgt) in enumerate(loader):
        print(x.shape, tgt.shape)

    print('done')
    quit()
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange

import torch
from gp import train_sample_gp 

# load all data dataframes for a user
def load_data(user_id, data_path):
    data = []
    folders = ['labels', 'heart_rate', 'motion']

    for folder in folders:
        fname = os.path.join(data_path, user_id + '_' + folder + '.parquet')
        df = pd.read_parquet(fname)

        # normalize all columns exept labels and time
        if folder != 'labels':
            normalize = lambda x : (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            df[df.columns[1:]] = df[df.columns[1:]].apply(normalize)
        data.append(df)
    
    return {folders[i] : data[i] for i in range(len(folders))}

# sample evenly spaced points from gaussian process
def sample_gp(t, y, n_samples):
    # if std dev is 0, return mean for all samples
    if y.std(dim=0).sum() < 1e-6:
        return torch.zeros(n_samples, y.shape[1]) + y.mean(dim=0)

    # if motion, take a subset of the data
    if y.shape[1] == 3:
        rand_idx = torch.randperm(y.shape[0])[:512]
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

    # get gp prediction    
    gp_pred = train_sample_gp(t, y_, t_test)

    if gp_pred is None:
        return None
    else:
        gp_pred['mean'] += mean
        gp_pred['std'] *= std
        return gp_pred

if __name__ == '__main__':
    data_path = '/home/gfloto/bio/wearable/data' 
    n_samples = 128
    user_ids = json.load(open(os.path.join(data_path, 'user_ids.json'))) 

    # each user has dataframes for: labels, heart_rate, motion
    print('loading data')
    user_data = {user_id : load_data(user_id, data_path) for user_id in user_ids}
    print('done')

    dset = {
        'user_id' : [],
        'label' : [],
        'time' : [],
        'gp-path' : [],
    }

    # train gp on each user's data
    for i, user_id in enumerate(user_ids):
        print(f'processing user {i+1}/{len(user_ids)}')

        # gather data in k - 30 second windows
        k = 4
        rows = len(user_data[user_id]['labels'])
        for t in tqdm(range(rows//k - 1)):

            # get start and end time of window
            t_start = user_data[user_id]['labels']['time'].iloc[k*t]
            t_end = user_data[user_id]['labels']['time'].iloc[k*(t+1)]

            # get label in window
            label = user_data[user_id]['labels']['label'].iloc[k*t : k*(t+1)].values
            label = [l for l in label if l != -1]
            if len(label) == 0: continue
            label = np.bincount(label).argmax()

            # get heart rate, motion data and label in window
            heart_rate = user_data[user_id]['heart_rate'][(user_data[user_id]['heart_rate']['time'] >= t_start) & (user_data[user_id]['heart_rate']['time'] < t_end)]
            motion = user_data[user_id]['motion'][(user_data[user_id]['motion']['time'] >= t_start) & (user_data[user_id]['motion']['time'] < t_end)]
            if len(heart_rate) == 0 and len(motion) == 0: continue

            # convert to numpy -> torch
            hr_t = torch.tensor( heart_rate['time'].values ).float()
            m_t = torch.tensor( motion['time'].values ).float()
            heart_rate = torch.tensor( heart_rate['heart_rate'].values ).float()[:, None]
            motion = torch.tensor( motion[['x', 'y', 'z']].values ).float()
            if heart_rate.shape[0] <= 2 or motion.shape[0] <= n_samples: continue


            # sample evenly spaced points from gaussian process
            hr_gp = sample_gp(hr_t, heart_rate, n_samples)
            m_gp = sample_gp(m_t, motion, n_samples)
            if hr_gp is None or m_gp is None: continue

            gp = {
                'mean' : torch.cat((hr_gp['mean'], m_gp['mean']), dim=-1).T,
                'std' : torch.cat((hr_gp['std'], m_gp['std']), dim=-1).T,
            }

            # save gp preditions
            save_path = os.path.join(data_path, 'gp_preds', f'{user_id}-{t_start}.pt')
            torch.save(gp, save_path)

            # save data
            dset['user_id'].append(user_id)
            dset['label'].append(label)
            dset['time'].append(t)
            dset['gp-path'].append(save_path)

    # save dataset as parquet
    dset = pd.DataFrame(dset)
    dset.to_parquet(os.path.join(data_path, 'wearable.parquet'))

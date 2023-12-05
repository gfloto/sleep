import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# map of folder : filename structure to load data
name_map = {
    'labels' : '_labeled_sleep.txt',
    'heart_rate' : '_heartrate.txt',
    'motion' : '_acceleration.txt',
}

# sup value in each folder
load_sep = {
    'labels' : ' ',
    'heart_rate' : ',',
    'motion' : ' ',
}

# map from data type to col name list
col_map = {
    'labels' : ['time', 'label'],
    'heart_rate' : ['time', 'heart_rate'],
    'motion' : ['time', 'x', 'y', 'z'],
}

# select load function based on folder
def load_data(fname, folders):
    assert folders[0] == 'labels'

    data = []
    for folder in folders:
        # load file
        fname = os.path.join(folder, user_id + name_map[folder])
        sep = load_sep[folder]
        df = pd.read_csv(fname, sep=sep, header=None)

        # only take labelled data
        if folder == 'labels':
            time = df[0].values
            t_start = time[0]; t_end = time[-1]

        # filter data
        else: df = df[(df[0] >= t_start) & (df[0] <= t_end)]

        df.columns = col_map[folder]
        data.append(df)
    return data

'''
create pandas df to store all data and labels
'''

if __name__ == '__main__':
    check = None
    folders = ['labels', 'heart_rate', 'motion']
    df_build = {f : [] for f in folders}
    df_build['user_id'] = []

    # get all user ids (assuming folder symmetry)
    files = os.listdir('labels')
    user_ids = [f.split('_')[0] for f in files]

    # save user ids in json
    with open(os.path.join('data', 'user_ids.json'), 'w') as f:
        json.dump(user_ids, f)

    # load each file, convert to numpy, store in df
    for user_id in tqdm(user_ids):
        labels, heart_rate, motion = load_data(user_id, folders)

        # save dataframes as parquet files
        labels.to_parquet(os.path.join('data', user_id + '_labels.parquet'))
        heart_rate.to_parquet(os.path.join('data', user_id + '_heart_rate.parquet'))
        motion.to_parquet(os.path.join('data', user_id + '_motion.parquet'))

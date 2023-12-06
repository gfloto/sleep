import os
import json
import torch

def save_metrics(tgt, pred, test_name, save_name='metrics.json'):
    # get acc for each photo code, tgt and pred are torch tensors
    acc_map = {}

    for idx in range(6):
        code_idx = torch.where(tgt == idx)[0]
        acc = float( torch.sum(tgt[code_idx] == pred[code_idx]) / len(code_idx) )

        acc_map[f'label code: {idx}'] = {
            'acc': acc,
            'count': len(code_idx),
        }

    save_path = os.path.join('results', test_name, save_name)
    with open(save_path, 'w') as f:
        json.dump(acc_map, f, indent=4)
import os
import json
import torch
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def save_metrics(data, test_name, save_name='metrics.json'):
    # use sklearn to get auroc info
    tgt, pred = data
    tgt = tgt.cpu().numpy(); pred = pred.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(tgt, pred)
    auroc = roc_auc_score(tgt, pred)

    # plot roc curve
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUROC: {auroc:.5f}')
    plt.savefig(os.path.join('results', test_name, 'roc.png'))
    plt.close()

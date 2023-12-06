import os
import json
from tqdm import tqdm
import torch
from einops import rearrange

# main test loop
#@torch.no_grad()
def valid(model, loader, device):
    model.eval()

    # main test loop
    tgt_track = None; pred_track = None
    for i, (x, tgt) in enumerate(tqdm(loader)):
        if i == 50: break
        x = x.to(device); tgt = tgt.to(device)

        # forward pass
        pred = model(x, cond=None)

        # track results
        if pred_track is None:
            tgt_track = tgt
            pred_track = pred
        else:
            tgt_track = torch.cat((tgt_track, tgt))
            pred_track = torch.cat((pred_track, pred))
    
    return tgt_track, pred_track

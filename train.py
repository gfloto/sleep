import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

from valid import valid

def train(model, loaders, loss_fn, optim, args):
    train_loader = loaders[0]
    valid_loader = loaders[1]
    device = args.device
    model.train()

    loss_track = []
    for i, (x, tgt) in enumerate(tqdm(train_loader)):
        x = x.to(device); tgt = tgt.to(device)

        # forward pass
        pred = model(x, cond=None)

        # loss
        loss = loss_fn(pred, tgt)

        # backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        # store loss
        loss_track.append(loss.item())
    train_loss = np.mean(loss_track)

    # run on valid
    tgt, pred = valid(model, valid_loader, args.device)
    valid_loss = loss_fn(pred, tgt) 

    return float(train_loss), float(valid_loss.cpu().item()), (tgt, pred)
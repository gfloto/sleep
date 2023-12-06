import os 
import json
import torch
import numpy as np 
from lion_pytorch import Lion

from args import get_args
from train import train
from loader import wearable_loader
from models import Transformer
from plot import plot_loss
from metrics import save_metrics

def main():
    args = get_args()
    save_path = os.path.join('results', args.test_name)

    # dataloader, model, loss, etc
    loaders = []
    modes = ['train', 'valid']
    for mode in modes:
        loaders.append(wearable_loader(args, mode))

    # make transformer model
    model = Transformer(in_size=8, out_size=1, d_model=args.n_samples).to(args.device)

    # add paramters from encoder and model
    #optim = Lion(params=list(model.parameters()), lr=args.lr, weight_decay=1e-2) 
    optim = torch.optim.Adam(params=list(model.parameters()), lr=args.lr, weight_decay=5e-2)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    # binary cross entropy loss
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # load model and optimizer if resuming training
    if args.resume_path is not None:
        print('loading model and optimizer from checkpoint')
        state_dict = torch.load(os.path.join('results', args.resume_path, 'model.pt'))

        model.load_state_dict(state_dict)
        optim.load_state_dict(torch.load(os.path.join('results', args.resume_path, 'optim.pt')))

        # load train and valid loss
        losses = json.load(open(os.path.join('results', args.resume_path, 'loss.json')))
        train_track = losses['train']
        valid_track = losses['valid']
    else:
        train_track, valid_track = [], []

    # main training loop
    for epoch in range(args.epochs):
        train_loss, valid_loss, valid_data = train(model, loaders, loss_fn, optim, args)

        # save training and validation loss
        train_track.append(train_loss); valid_track.append(valid_loss)
        plot_loss(train_track, valid_track, save_path)

        # print epoch info
        print(f'epoch: {epoch} train nll: {train_loss:.5f} valid nll: {valid_loss:.5f} best valid nll: {min(valid_track):.5f}')

        # save model, print images to view
        if valid_loss == min(valid_track):
            print('saving model and optimizer')
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            torch.save(optim.state_dict(), os.path.join(save_path, 'optim.pt'))

            # save aucroc
            save_metrics(valid_data, args.test_name)

        # early stopping
        last_n = valid_track[-args.early_stop:]            
        if np.argmin(last_n) == 0 and epoch > args.early_stop: 
            print('early stopping')
            break

if __name__ == '__main__':
    main()
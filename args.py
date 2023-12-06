import os
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default='dev')
    parser.add_argument('--data_path', default='/home/gfloto/bio/data/wearable', help='train dataset, either odo or damage')

    parser.add_argument('--n_samples', type=int, default=128, help='number of samples to take from gaussian process')

    parser.add_argument('--resume_path', help='resume training')
    parser.add_argument('--early_stop', default=100, help='early stop training')
    parser.add_argument('--device', default='cuda', help='device being used')

    parser.add_argument('--lr', type=float, default=10e-6, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')

    parser.add_argument('--transfer', action='store_true', help='transfer learning mode')
    parser.add_argument('--cond_size', type=int, default=64, help='size of conditioning vector')

    args = parser.parse_args()

    # check args
    assert args.test_name is not None, 'enter a test name'

    # make results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(os.path.join('results', args.test_name)):
        os.makedirs(os.path.join('results', args.test_name))
    else:
        if args.test_name != 'dev' and args.resume_path is None:
            raise ValueError(f'test_name: {args.test_name} already exists')
    
    # save to .json
    save_args(args)

    return args

# save args to .json
def save_args(args):
    save_path = os.path.join('results', args.test_name, 'args.json')
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
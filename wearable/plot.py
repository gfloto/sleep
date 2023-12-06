import os
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# plot loss curves
def plot_loss(train_loss, val_loss, save_path):
    # plot loss info
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('nll')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()

    # save loss info
    loss_dict = {'train': train_loss, 'valid': val_loss}
    with open(os.path.join(save_path, 'loss.json'), 'w') as f:
        json.dump(loss_dict, f)

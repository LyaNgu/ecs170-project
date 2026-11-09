'''
Helper for drawing learning (convergence) curves from a training history
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_learning_curve(history, title, save_path):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(history['epoch'], history['train_loss'], marker='o', ms=3, color='#2a6fdb')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Training loss'); ax[0].set_title(title + ': loss')
    ax[0].grid(alpha=0.3)
    ax[1].plot(history['epoch'], history['train_acc'], marker='o', ms=3, color='#2a6fdb', label='train')
    if history['test_acc'][0] is not None:
        ax[1].plot(history['epoch'], history['test_acc'], marker='s', ms=3, color='#e07b00', label='test')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy'); ax[1].set_title(title + ': accuracy')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('learning curve saved to', save_path)

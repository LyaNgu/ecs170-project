'''
Stage 3, task 3-5: change the CNN configuration (depth, kernel size, padding, stride,
pooling, hidden dimension, loss function, optimizer) and record the impact on performance.

Usage (from script/stage_3_script):
    python script_cnn_ablation.py MNIST            # all variants, default epochs
    python script_cnn_ablation.py ORL --epochs 30
    python script_cnn_ablation.py CIFAR --epochs 8 --only baseline depth_shallow pool_avg

Each variant changes ONE thing relative to the dataset's baseline model, so differences
can be attributed to that change. Results go to result/stage_3_result/ablation_<DATASET>.csv/.md
plus an overlaid test-accuracy plot.
'''
import argparse, csv, os
import _common
from code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = {'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}


def variants(cls):
    base_depth = list(cls.conv_channels)
    k = cls.kernel_size
    return {
        'baseline':          {},
        # model depth
        'depth_shallow':     {'conv_channels': base_depth[:1]},
        'depth_deeper':      {'conv_channels': base_depth + [base_depth[-1] * 2]},
        # kernel size (padding kept "same")
        'kernel_small':      {'kernel_size': 3 if k != 3 else 1, 'padding': (3 if k != 3 else 1) // 2},
        'kernel_large':      {'kernel_size': k + 2, 'padding': (k + 2) // 2},
        # padding
        'padding_none':      {'padding': 0},
        # stride (pooling removed so the stride does the down-sampling)
        'stride2_no_pool':   {'stride': 2, 'pooling': None},
        # pooling layer
        'pool_avg':          {'pooling': 'avg'},
        # hidden layer dimension
        'hidden_small':      {'hidden_dim': max(16, cls.hidden_dim // 4)},
        'hidden_large':      {'hidden_dim': cls.hidden_dim * 4},
        # loss function
        'loss_label_smooth': {'loss_name': 'label_smoothing'},
        'loss_multi_margin': {'loss_name': 'multi_margin'},
        'loss_mse':          {'loss_name': 'mse'},
        # optimizer
        'optim_sgd':         {'optimizer_name': 'sgd', 'learning_rate': 0.01},
        'optim_rmsprop':     {'optimizer_name': 'rmsprop', 'learning_rate': 1e-3},
        # regularization
        'no_batchnorm':      {'use_batchnorm': False},
        'no_dropout':        {'dropout': 0.0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=MODELS.keys())
    ap.add_argument('--epochs', type=int, default=None, help='override max_epoch for every variant')
    ap.add_argument('--only', nargs='*', default=None, help='run only these variant names')
    args = ap.parse_args()

    cls = MODELS[args.dataset]
    all_variants = variants(cls)
    names = args.only or list(all_variants)
    rows, curves = [], {}
    for name in names:
        cfg = dict(all_variants[name])
        if args.epochs:
            cfg['max_epoch'] = args.epochs
        print('\n================ %s / %s : %s ================' % (args.dataset, name, cfg))
        _common.set_seed(2)
        try:
            method_obj = cls('CNN %s %s' % (args.dataset, name), '', **cfg)
        except RuntimeError as e:   # e.g. feature map shrinks to zero with no padding + deep net
            print('skipped (invalid architecture):', e)
            continue
        scores, history = _common.run_experiment(args.dataset, method_obj,
                                                 'ablation/%s_%s' % (args.dataset, name),
                                                 use_single_channel=(args.dataset == 'ORL'))
        curves[name] = history
        rows.append({'variant': name, 'change': str(all_variants[name]) if all_variants[name] else '-',
                     'accuracy': scores['accuracy'], 'f1_macro': scores['f1_macro'],
                     'f1_weighted': scores['f1_weighted'], 'params': method_obj.count_parameters(),
                     'train_time_s': round(method_obj.method_training_time, 1)})

    out = _common.RESULT_DIR + 'ablation_' + args.dataset
    with open(out + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    with open(out + '.md', 'w') as f:
        f.write('| Variant | Change | Accuracy | Macro F1 | Weighted F1 | Params | Train time (s) |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for r in rows:
            f.write('| %s | %s | %.4f | %.4f | %.4f | %d | %.1f |\n' % (
                r['variant'], r['change'], r['accuracy'], r['f1_macro'], r['f1_weighted'],
                r['params'], r['train_time_s']))

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in curves.items():
        ax.plot(h['epoch'], h['test_acc'], label=name, lw=2.2 if name == 'baseline' else 1)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test accuracy'); ax.set_title(args.dataset + ' configuration study')
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out + '_curves.png', dpi=150); plt.close(fig)
    print('\nsaved', out + '.csv', out + '.md', out + '_curves.png')


if __name__ == '__main__':
    main()

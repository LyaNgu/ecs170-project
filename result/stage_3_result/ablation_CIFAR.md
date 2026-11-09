| Variant | Change | Accuracy | Macro F1 | Weighted F1 | Params | Train time (s) |
|---|---|---|---|---|---|---|
| baseline | - | 0.6260 | 0.6201 | 0.6201 | 1423114 | 187.6 |
| depth_shallow | {'conv_channels': [64]} | 0.5151 | 0.5093 | 0.5093 | 4199050 | 109.4 |
| depth_deeper | {'conv_channels': [64, 128, 256, 512]} | 0.6240 | 0.6099 | 0.6099 | 2080010 | 269.9 |
| kernel_large | {'kernel_size': 5, 'padding': 2} | 0.5939 | 0.5842 | 0.5842 | 2081546 | 337.7 |
| padding_none | {'padding': 0} | 0.6948 | 0.6903 | 0.6903 | 636682 | 118.2 |
| stride2_no_pool | {'stride': 2, 'pooling': None} | 0.6253 | 0.6250 | 0.6250 | 1423114 | 54.9 |
| pool_avg | {'pooling': 'avg'} | 0.6637 | 0.6576 | 0.6576 | 1423114 | 186.7 |
| hidden_small | {'hidden_dim': 64} | 0.5961 | 0.5828 | 0.5828 | 634570 | 187.0 |
| loss_mse | {'loss_name': 'mse'} | 0.5611 | 0.5534 | 0.5534 | 1423114 | 193.4 |
| optim_sgd | {'optimizer_name': 'sgd', 'learning_rate': 0.01} | 0.6508 | 0.6492 | 0.6492 | 1423114 | 192.2 |
| no_batchnorm | {'use_batchnorm': False} | 0.6582 | 0.6593 | 0.6593 | 1422218 | 161.1 |
| no_dropout | {'dropout': 0.0} | 0.6903 | 0.6904 | 0.6904 | 1423114 | 193.0 |

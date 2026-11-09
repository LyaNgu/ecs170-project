| Variant | Change | Accuracy | Macro F1 | Weighted F1 | Params | Train time (s) |
|---|---|---|---|---|---|---|
| baseline | - | 0.9750 | 0.9667 | 0.9667 | 2598408 | 78.4 |
| depth_shallow | {'conv_channels': [16]} | 0.9250 | 0.9000 | 0.9000 | 10562280 | 59.0 |
| depth_deeper | {'conv_channels': [16, 32, 64, 128]} | 1.0000 | 1.0000 | 1.0000 | 1427336 | 87.0 |
| kernel_small | {'kernel_size': 3, 'padding': 1} | 0.9750 | 0.9667 | 0.9667 | 2557192 | 60.8 |
| kernel_large | {'kernel_size': 7, 'padding': 3} | 1.0000 | 1.0000 | 1.0000 | 2660232 | 96.5 |
| padding_none | {'padding': 0} | 0.9500 | 0.9333 | 0.9333 | 1385992 | 57.0 |
| stride2_no_pool | {'stride': 2, 'pooling': None} | 0.9250 | 0.9083 | 0.9083 | 2827784 | 20.3 |
| pool_avg | {'pooling': 'avg'} | 0.9500 | 0.9333 | 0.9333 | 2598408 | 68.9 |
| hidden_small | {'hidden_dim': 64} | 0.8250 | 0.7750 | 0.7750 | 698184 | 71.3 |
| hidden_large | {'hidden_dim': 1024} | 1.0000 | 1.0000 | 1.0000 | 10199304 | 90.0 |
| loss_label_smooth | {'loss_name': 'label_smoothing'} | 0.9750 | 0.9667 | 0.9667 | 2598408 | 73.5 |
| loss_multi_margin | {'loss_name': 'multi_margin'} | 1.0000 | 1.0000 | 1.0000 | 2598408 | 72.9 |
| loss_mse | {'loss_name': 'mse'} | 0.9250 | 0.9000 | 0.9000 | 2598408 | 72.9 |
| optim_sgd | {'optimizer_name': 'sgd', 'learning_rate': 0.01} | 0.9500 | 0.9333 | 0.9333 | 2598408 | 70.5 |
| optim_rmsprop | {'optimizer_name': 'rmsprop', 'learning_rate': 0.001} | 0.9750 | 0.9667 | 0.9667 | 2598408 | 78.0 |
| no_batchnorm | {'use_batchnorm': False} | 0.9500 | 0.9333 | 0.9333 | 2598184 | 60.4 |
| no_dropout | {'dropout': 0.0} | 0.9750 | 0.9667 | 0.9667 | 2598408 | 69.8 |

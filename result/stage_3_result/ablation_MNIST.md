| Variant | Change | Accuracy | Macro F1 | Weighted F1 | Params | Train time (s) |
|---|---|---|---|---|---|---|
| baseline | - | 0.9879 | 0.9879 | 0.9879 | 421834 | 65.5 |
| depth_shallow | {'conv_channels': [32]} | 0.9764 | 0.9761 | 0.9764 | 804618 | 45.7 |
| depth_deeper | {'conv_channels': [32, 64, 128]} | 0.9907 | 0.9906 | 0.9907 | 241994 | 592.2 |
| kernel_small | {'kernel_size': 1, 'padding': 0} | 0.9258 | 0.9246 | 0.9256 | 405194 | 61.0 |
| kernel_large | {'kernel_size': 5, 'padding': 2} | 0.9894 | 0.9893 | 0.9894 | 455114 | 88.9 |
| padding_none | {'padding': 0} | 0.9883 | 0.9883 | 0.9883 | 225226 | 48.0 |
| stride2_no_pool | {'stride': 2, 'pooling': None} | 0.9854 | 0.9853 | 0.9854 | 421834 | 16.4 |
| pool_avg | {'pooling': 'avg'} | 0.9896 | 0.9895 | 0.9896 | 421834 | 64.3 |
| hidden_small | {'hidden_dim': 32} | 0.9849 | 0.9847 | 0.9849 | 119722 | 66.8 |
| hidden_large | {'hidden_dim': 512} | 0.9890 | 0.9890 | 0.9890 | 1630282 | 72.4 |
| loss_label_smooth | {'loss_name': 'label_smoothing'} | 0.9895 | 0.9894 | 0.9895 | 421834 | 67.4 |
| loss_multi_margin | {'loss_name': 'multi_margin'} | 0.9854 | 0.9854 | 0.9854 | 421834 | 67.9 |
| loss_mse | {'loss_name': 'mse'} | 0.9859 | 0.9858 | 0.9859 | 421834 | 68.0 |
| optim_sgd | {'optimizer_name': 'sgd', 'learning_rate': 0.01} | 0.9865 | 0.9865 | 0.9865 | 421834 | 66.3 |
| optim_rmsprop | {'optimizer_name': 'rmsprop', 'learning_rate': 0.001} | 0.9867 | 0.9867 | 0.9867 | 421834 | 66.1 |
| no_batchnorm | {'use_batchnorm': False} | 0.9872 | 0.9872 | 0.9872 | 421642 | 53.8 |
| no_dropout | {'dropout': 0.0} | 0.9844 | 0.9845 | 0.9844 | 421834 | 64.4 |

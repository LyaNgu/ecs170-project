'''
CNN for the CIFAR-10 colored object dataset (32x32x3 images, 10 classes)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    # CIFAR is much harder than MNIST: deeper network (3 blocks), wider hidden layer,
    # batch norm, and flip/crop data augmentation to reduce over-fitting.
    # conv(3->64) -> pool -> conv(64->128) -> pool -> conv(128->256) -> pool -> FC(->256) -> FC(->10)
    in_channels = 3
    input_size = (32, 32)
    num_classes = 10
    conv_channels = [64, 128, 256]
    kernel_size = 3
    padding = 1
    pooling = 'max'
    use_batchnorm = True
    hidden_dim = 256
    dropout = 0.3
    optimizer_name = 'adam'
    learning_rate = 1e-3
    weight_decay = 5e-4
    max_epoch = 20
    batch_size = 128
    augment = True

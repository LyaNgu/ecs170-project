'''
CNN for the ORL face dataset (112x92 gray faces, 40 people, 9 train / 1 test image each)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    # Only 360 training images, so: one input channel (R=G=B), a small network,
    # heavy dropout, weight decay, small batches and more epochs.
    # conv(1->16) -> pool -> conv(16->32) -> pool -> conv(32->64) -> pool -> FC(->256) -> FC(->40)
    in_channels = 1
    input_size = (112, 92)
    num_classes = 40
    conv_channels = [16, 32, 64]
    kernel_size = 5
    padding = 2
    pooling = 'max'
    hidden_dim = 256
    dropout = 0.5
    optimizer_name = 'adam'
    learning_rate = 1e-3
    weight_decay = 1e-4
    max_epoch = 40
    batch_size = 32

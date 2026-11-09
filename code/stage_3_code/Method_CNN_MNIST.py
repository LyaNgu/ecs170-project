'''
CNN for the MNIST hand-written digit dataset (28x28 gray images, 10 classes)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    # conv(1->32) -> pool -> conv(32->64) -> pool -> FC(3136->128) -> FC(128->10)
    in_channels = 1
    input_size = (28, 28)
    num_classes = 10
    conv_channels = [32, 64]
    kernel_size = 3
    padding = 1
    pooling = 'max'
    hidden_dim = 128
    dropout = 0.25
    optimizer_name = 'adam'
    learning_rate = 1e-3
    max_epoch = 10
    batch_size = 128

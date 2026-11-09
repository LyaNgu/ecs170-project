'''
Stage 3: CNN on the MNIST dataset
Run from PyCharm (working dir = script/stage_3_script) or:  python script_cnn_mnist.py
'''
import _common
from code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST

#---- Convolutional Neural Network script ----
if 1:
    _common.set_seed(2)
    method_obj = Method_CNN_MNIST('CNN MNIST', '')
    _common.run_experiment('MNIST', method_obj, 'CNN_MNIST', use_single_channel=False)

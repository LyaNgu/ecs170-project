'''
Stage 3: CNN on the CIFAR dataset
Run from PyCharm (working dir = script/stage_3_script) or:  python script_cnn_cifar.py
'''
import _common
from code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR

#---- Convolutional Neural Network script ----
if 1:
    method_obj = Method_CNN_CIFAR('CNN CIFAR', '')
    _common.run_experiment('CIFAR', method_obj, 'CNN_CIFAR', use_single_channel=False)

'''
Stage 3: CNN on the ORL dataset
Run from PyCharm (working dir = script/stage_3_script) or:  python script_cnn_orl.py
'''
import _common
from code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL

#---- Convolutional Neural Network script ----
if 1:
    _common.set_seed(2)
    method_obj = Method_CNN_ORL('CNN ORL', '')
    _common.run_experiment('ORL', method_obj, 'CNN_ORL', use_single_channel=True)

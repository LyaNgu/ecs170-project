'''
Shared runner used by the three stage 3 scripts (keeps them short and identical in structure)
'''
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)   # so `from code....` works when run from a terminal as well as PyCharm

from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test import Setting_Train_Test
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from code.stage_3_code.Plot_Learning_Curve import plot_learning_curve
import numpy as np
import torch

DATA_DIR = os.path.join(ROOT, 'data', 'stage_3_data') + os.sep
RESULT_DIR = os.path.join(ROOT, 'result', 'stage_3_result') + os.sep


def run_experiment(dataset_file, method_obj, tag, use_single_channel=False, seed=2, verbose=True):
    #---- parameter section -------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader(dataset_file, '', use_single_channel=use_single_channel)
    data_obj.dataset_source_folder_path = DATA_DIR
    data_obj.dataset_source_file_name = dataset_file

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR
    result_obj.result_destination_file_name = tag + '_prediction_result'

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '')
    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores, history = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for k, v in scores.items():
        print('%-20s %.4f' % (k, v))
    print('training time: %.1fs, parameters: %d' % (method_obj.method_training_time, method_obj.count_parameters()))
    print('************ Finish ************')

    plot_learning_curve(history, tag, RESULT_DIR + tag + '_learning_curve.png')
    with open(RESULT_DIR + tag + '_scores.json', 'w') as f:
        json.dump({'scores': scores, 'history': history,
                   'training_time_s': method_obj.method_training_time,
                   'num_params': method_obj.count_parameters()}, f, indent=2)
    return scores, history

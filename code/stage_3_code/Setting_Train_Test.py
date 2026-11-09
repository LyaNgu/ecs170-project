'''
Concrete SettingModule class for a pre-partitioned train/test experiment
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting


class Setting_Train_Test(setting):
    '''
    The stage 3 datasets come with their own train/test split,
    so no random split or cross validation is needed here.
    '''

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw ResultModule (predictions + training history for the learning curves)
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), learned_result['history']

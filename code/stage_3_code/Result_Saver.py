'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.result import result
import pickle
import os


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        folder = os.path.dirname(self.result_destination_folder_path + self.result_destination_file_name)
        if folder:
            os.makedirs(folder, exist_ok=True)
        f = open(self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count), 'wb')
        pickle.dump(self.data, f)
        f.close()

    def load(self):
        print('loading results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count), 'rb')
        self.data = pickle.load(f)
        f.close()
        return self.data

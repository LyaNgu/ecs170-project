'''
Concrete Evaluate class for multiclass evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    '''
    Multiclass classification, so binary precision/recall/F1 cannot be used.
    Reports accuracy plus weighted and macro averaged precision / recall / F1.
    '''
    data = None
    verbose = True

    def evaluate(self):
        if self.verbose:
            print('evaluating performance...')
        y_true, y_pred = self.data['true_y'], self.data['pred_y']
        scores = {'accuracy': accuracy_score(y_true, y_pred)}
        for avg in ['weighted', 'macro']:
            scores['precision_' + avg] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            scores['recall_' + avg] = recall_score(y_true, y_pred, average=avg, zero_division=0)
            scores['f1_' + avg] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        return scores

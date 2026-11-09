'''
Concrete MethodModule class for a configurable Convolutional Neural Network.
Method_CNN_MNIST / Method_CNN_ORL / Method_CNN_CIFAR inherit from this class and
only set their own default architecture + hyper-parameters.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import numpy as np
import time


def get_device():
    # 3-7 (optional): use an NVIDIA GPU (CUDA) or Apple-silicon GPU (MPS) when available
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN(method, nn.Module):
    data = None

    # ---- default configuration (overridden by subclasses / constructor kwargs) ----
    in_channels = 1
    input_size = (28, 28)          # (H, W)
    num_classes = 10
    conv_channels = [32, 64]       # one conv block per entry -> controls model depth
    kernel_size = 3
    padding = 1
    stride = 1
    pooling = 'max'                # 'max' | 'avg' | None
    use_batchnorm = True
    hidden_dim = 128               # size of the fully-connected hidden layer
    dropout = 0.25
    loss_name = 'cross_entropy'    # 'cross_entropy' | 'label_smoothing' | 'multi_margin' | 'mse'
    optimizer_name = 'adam'        # 'adam' | 'sgd' | 'adamw' | 'rmsprop'
    learning_rate = 1e-3
    weight_decay = 0.0
    max_epoch = 10
    batch_size = 128
    augment = False                # random horizontal flip + random crop (CIFAR)

    def __init__(self, mName=None, mDescription=None, **config):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        for k, v in config.items():
            if not hasattr(self, k):
                raise ValueError('unknown CNN config option: ' + k)
            setattr(self, k, v)
        self.device = get_device()
        self._build()
        self.to(self.device)

    # ------------------------------------------------------------------ model
    def _build(self):
        layers = []
        c_in = self.in_channels
        for c_out in self.conv_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU())
            if self.pooling == 'max':
                layers.append(nn.MaxPool2d(2))
            elif self.pooling == 'avg':
                layers.append(nn.AvgPool2d(2))
            c_in = c_out
        self.features = nn.Sequential(*layers)

        # work out the flattened feature size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *self.input_size)
            flat_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x):
        '''Forward propagation: returns raw class scores (logits)'''
        return self.classifier(self.features(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------ loss / optim
    def _loss_function(self):
        if self.loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        if self.loss_name == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        if self.loss_name == 'multi_margin':       # multi-class hinge (SVM-style) loss
            return nn.MultiMarginLoss()
        if self.loss_name == 'mse':                # MSE between softmax output and one-hot target
            mse = nn.MSELoss()
            k = self.num_classes
            return lambda logits, y: mse(torch.softmax(logits, dim=1),
                                         nn.functional.one_hot(y, k).float())
        raise ValueError(self.loss_name)

    def _optimizer(self):
        p, lr, wd = self.parameters(), self.learning_rate, self.weight_decay
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(p, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(p, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(p, lr=lr, weight_decay=wd)
        raise ValueError(self.optimizer_name)

    def _augment(self, xb):
        # random horizontal flip
        flip = torch.rand(xb.size(0), device=xb.device) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        # random crop with 4-pixel zero padding
        pad = 4
        h, w = xb.shape[2], xb.shape[3]
        xp = nn.functional.pad(xb, (pad, pad, pad, pad))
        i, j = np.random.randint(0, 2 * pad + 1, size=2)
        return xp[:, :, i:i + h, j:j + w]

    # ---------------------------------------------------------- train / test
    def _predict(self, X):
        self.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), 1000):
                xb = torch.as_tensor(X[start:start + 1000], device=self.device)
                preds.append(self.forward(xb).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def train_model(self, X, y, X_test=None, y_test=None):
        optimizer = self._optimizer()
        loss_function = self._loss_function()
        accuracy_evaluator = Evaluate_Metrics('training evaluator', '')
        accuracy_evaluator.verbose = False
        history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}

        X_t = torch.as_tensor(X)
        y_t = torch.as_tensor(y)
        n = len(X_t)
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            perm = torch.randperm(n)
            epoch_loss, seen = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = X_t[idx].to(self.device)
                yb = y_t[idx].to(self.device)
                if self.augment:
                    xb = self._augment(xb)
                y_pred = self.forward(xb)
                train_loss = loss_function(y_pred, yb)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * len(idx)
                seen += len(idx)

            # per-epoch metrics for the learning curves
            accuracy_evaluator.data = {'true_y': y, 'pred_y': self._predict(X)}
            train_acc = accuracy_evaluator.evaluate()['accuracy']
            test_acc = None
            if X_test is not None:
                accuracy_evaluator.data = {'true_y': y_test, 'pred_y': self._predict(X_test)}
                test_acc = accuracy_evaluator.evaluate()['accuracy']
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(epoch_loss / seen)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            print('Epoch:', epoch + 1, 'Loss: %.4f' % (epoch_loss / seen),
                  'Train Acc: %.4f' % train_acc,
                  ('Test Acc: %.4f' % test_acc) if test_acc is not None else '')
        return history

    def test(self, X):
        return self._predict(X)

    def run(self):
        print('method running...')
        print('device:', self.device, '| trainable parameters:', self.count_parameters())
        print('--start training...')
        t0 = time.time()
        history = self.train_model(self.data['train']['X'], self.data['train']['y'],
                                   self.data['test']['X'], self.data['test']['y'])
        self.method_training_time = time.time() - t0
        print('--start testing...')
        t0 = time.time()
        pred_y = self.test(self.data['test']['X'])
        self.method_testing_time = time.time() - t0
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': history,
                'training_time': self.method_training_time, 'num_params': self.count_parameters()}

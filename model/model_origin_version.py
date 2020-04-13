import torch
import torch.nn as nn
import numpy as np
from utlis.TransferModel import tf2s


class MMANN(nn.Module):
    def __init__(self, in_feature=3, n_out=10):
        '''
        n_out is Q_k
        '''
        super(MMANN, self).__init__()
        self.n_out = n_out
        n_hidden = 2 * in_feature + 1
        self.PR = nn.Sequential(nn.Linear(in_feature, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, self.n_out * 4))

        self.PR_ = nn.Sequential(nn.Linear(in_feature, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden * 3 + 1),
                               nn.BatchNorm1d(n_hidden * 3 + 1),
                               nn.ReLU(),
                               nn.Linear(n_hidden * 3 + 1, self.n_out * 4))
        self.comptfc = tf2s(n_out)


    def forward(self, x, freq=None, ann_out=True):
        # x = nn.ReLU(x)
        pr = self.PR_(x)

        if ann_out:
            return pr
        else:
            s = self.comptfc(pr, freq)
            return s, pr


if __name__ == "__main__":
    from data import load_data

    train_data = load_data(path='./matlab/Training_Data.mat', names=['responses'])

    # model = MMANN()
    # input_ = torch.randn([3, 3])
    # print(model(input_))



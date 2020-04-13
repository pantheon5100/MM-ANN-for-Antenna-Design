import torch
import torch.nn as nn
import numpy as np


class MMANN(nn.Module):
    def __init__(self, in_feature=3, n_out=3):
        '''
        n_out is Q_k
        '''
        super(MMANN, self).__init__()
        self.n_out = n_out
        n_hidden = 2 * in_feature + 1
        self.P = nn.Sequential(nn.Linear(in_feature, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, self.n_out * 2))
        self.R = nn.Sequential(nn.Linear(in_feature, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, self.n_out * 2))
        self.comptfc = tf2s(n_out)


    def forward(self, x, freq=None, ann_out=True):
        # x = nn.ReLU(x)
        p = self.P(x)
        r = self.R(x)
        pr = torch.cat([p, r], 1)

        if ann_out:
            return pr
        else:
            s = self.comptfc(pr, freq)
            return s


class tf2s(nn.Module):
    def __init__(self, n_pr=3):
        super(tf2s, self).__init__()
        self.n_pr = n_pr

    def forward(self, tfc, freq):
        freq = freq + 4.5e2
        freq = torch.tensor(freq).repeat(3, 1).T.cuda()
        tfc = tfc + 100

        res = torch.ones((tfc.size()[0], 1001, 2), requires_grad=True)
        # res = []

        for n, [ars, ais, crs, cis] in enumerate(
                zip(tfc[:, :self.n_pr], tfc[:, self.n_pr:self.n_pr * 2], tfc[:, self.n_pr * 2:self.n_pr * 3],
                    tfc[:, self.n_pr * 3:])):

            d = 2 * np.pi * freq - ais
            c2_d2 = ars ** 2 + d ** 2
            ac_bd = -crs * ars + cis * d
            bc_ad = -cis * ars - crs * d
            r = ac_bd / c2_d2
            i = bc_ad / c2_d2

            tmp_r = torch.sum(r, 1)
            tmp_i = torch.sum(i, 1)

            ars, ais, crs, cis = ars[ais > 0], -ais[ais > 0], crs[ais > 0], -cis[ais > 0]
            d = 2 * np.pi * freq - ais
            c2_d2 = ars ** 2 + d ** 2
            ac_bd = -crs * ars + cis * d
            bc_ad = -cis * ars - crs * d
            r = ac_bd / c2_d2
            i = bc_ad / c2_d2

            tmp_r = torch.sum(r, 1) + tmp_r
            tmp_i = torch.sum(i, 1) + tmp_i
            with torch.no_grad():
                res[n, :, 0] = tmp_r
                res[n, :, 1] = tmp_i
            # res.append()
        return res


if __name__ == "__main__":
    from data import load_data

    train_data = load_data(path='./matlab/Training_Data.mat', names=['responses'])

    # model = MMANN()
    # input_ = torch.randn([3, 3])
    # print(model(input_))



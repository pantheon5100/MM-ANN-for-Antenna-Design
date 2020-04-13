import torch
import torch.nn as nn
import numpy as np


class tf2s(nn.Module):
    def __init__(self, n_pr=3):
        super(tf2s, self).__init__()
        self.n_pr = n_pr

    def forward(self, tfc, freq):
        # freq = freq + 4.5e2
        freq = torch.tensor(freq).repeat(self.n_pr, 1).T.cuda()
        res = []
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

            # ars, ais, crs, cis = ars[ais > 0], -ais[ais > 0], crs[ais > 0], -cis[ais > 0]
            # d = 2 * np.pi * freq - ais
            # c2_d2 = ars ** 2 + d ** 2
            # ac_bd = -crs * ars + cis * d
            # bc_ad = -cis * ars - crs * d
            # r = ac_bd / c2_d2
            # i = bc_ad / c2_d2
            #
            # tmp_r = torch.sum(r, 1) + tmp_r
            # tmp_i = torch.sum(i, 1) + tmp_i

            res.append(torch.stack([tmp_r, tmp_i]).T)
                # res.append([tmp_r, tmp_i])

        # tfc = tfc + 100

        # res = torch.ones((tfc.size()[0], 1001, 2), requires_grad=True)
        # # res = []
        #
        # for n, [ars, ais, crs, cis] in enumerate(
        #         zip(tfc[:, :self.n_pr], tfc[:, self.n_pr:self.n_pr * 2], tfc[:, self.n_pr * 2:self.n_pr * 3],
        #             tfc[:, self.n_pr * 3:])):
        #
        #     d = 2 * np.pi * freq - ais
        #     c2_d2 = ars ** 2 + d ** 2
        #     ac_bd = -crs * ars + cis * d
        #     bc_ad = -cis * ars - crs * d
        #     r = ac_bd / c2_d2
        #     i = bc_ad / c2_d2
        #
        #     tmp_r = torch.sum(r, 1)
        #     tmp_i = torch.sum(i, 1)
        #
        #     # ars, ais, crs, cis = ars[ais > 0], -ais[ais > 0], crs[ais > 0], -cis[ais > 0]
        #     # d = 2 * np.pi * freq - ais
        #     # c2_d2 = ars ** 2 + d ** 2
        #     # ac_bd = -crs * ars + cis * d
        #     # bc_ad = -cis * ars - crs * d
        #     # r = ac_bd / c2_d2
        #     # i = bc_ad / c2_d2
        #     #
        #     # tmp_r = torch.sum(r, 1) + tmp_r
        #     # tmp_i = torch.sum(i, 1) + tmp_i
        #     with torch.no_grad():
        #         res[n, :, 0] = tmp_r
        #         res[n, :, 1] = tmp_i
            # res.append()
        return res

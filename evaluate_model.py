import torch
import math
import numpy as np
import glob

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from model.model_pr_mul import tf2s
from model.model_origin_version import MMANN
from data import load_ann_pr, load_s
from utlis.parser_argparse import get_parser
from utlis.transferfunction import freq_resp, plot_comparision_s11, plot_n_s11


def main():
    parser = get_parser()
    arg_distill = ['--save_dir', 'geo2tfc_s',
                   '--n_pr', '10',
                   '--epochs', '10000',
                   '--batch', '40',
                   '--lr', '1e-1',
                   '--lr_decay_epochs', '10',
                   '--use_gpu', 'True',
                   '--lr_scheduler', 'CosineAnnealingLR',
                   '--s_para', 'True']

    opts = parser.parse_args(arg_distill)

    # load data
    train_data, test_data, responses = load_ann_pr(opts.n_pr, opts.s_para)

    freq = responses[0]
    train_tfc = responses[1]
    test_tfc = responses[2]

    # model
    model = MMANN(n_out=opts.n_pr)
    model_files = glob.glob(r"./torch_save/Ex2/pth/*.pth")

    detail = False
    for model_file in [r"./torch_save/Ex3/geo2tfc_3B/best-epoch_4311-0.0010.pth"]:
        print(model_file)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        net = tf2s(opts.n_pr)

        for loader, tfc, name in  zip([test_data, train_data], [test_tfc, train_tfc], ["test", "train"]):
            print(name)
            model.eval()
            cri = torch.nn.MSELoss()
            with torch.no_grad():
                pr_tf, labels = loader[:]
                pr_tf, labels = pr_tf.float(), labels.float()
                output, pr = model(pr_tf, freq=freq, ann_out=False)
                output = torch.stack(output)
                s11_tfc = net(torch.tensor(tfc), freq)
                loss = cri(output, labels)
                plt.figure()
                # n_rows = math.ceil(len(loader) / 3)
                mape_for_s11tfc_labels = torch.mean(torch.abs((torch.stack(s11_tfc) - labels) / labels)) * 100
                print(mape_for_s11tfc_labels)

                mape = torch.mean(torch.abs((output - labels) / labels)) * 100
                print(name, mape.numpy())
                if detail:
                    mape = torch.mean(torch.abs((output - labels) / labels), (1, 2)) * 100
                    print(name, mape.numpy())

                yp = np.log10(np.abs(output.cpu().detach().numpy()[:, :, 0] + 1j*output.cpu().detach().numpy()[:, :, 1]))
                yl = np.log10(np.abs(labels.cpu().detach().numpy()[:, :, 0] + 1j*labels.cpu().detach().numpy()[:, :, 1]))
                mape_s = np.mean(np.abs((yp - yl) / yl)) * 100
                print(name, mape_s)
                if detail:
                    mape_s = np.mean(np.abs((yp - yl) / yl), 1) * 100
                    print(name, mape_s)

                for n_plot in range(len(loader)):
                    plt.figure()
                    # plt.subplot(n_rows, 3, n_plot+1)
                    fig = plot_n_s11([labels[n_plot].cpu().detach().numpy(), output[n_plot].cpu().detach().numpy(),
                                      s11_tfc[n_plot]], freq)
                    # print(n_plot)
                    plt.savefig("./fig3A/{}_{}.jpg".format(name, n_plot))
                    plt.close(fig)


if __name__ == "__main__":
    main()

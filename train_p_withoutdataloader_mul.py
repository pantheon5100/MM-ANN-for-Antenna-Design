import argparse
import os

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

import matplotlib.pyplot as plt

from model.model_d4 import MMANN
# from model.model_origin_version import MMANN
# from model.model_4 import MMANN
from data import load_ann_pr, load_s
from utlis.parser_argparse import get_parser
from utlis.transferfunction import freq_resp, plot_comparision_s11, plot_n_s11


def main():
    parser = get_parser()
    arg_distill = ['--save_dir', 'geo2tfc_4',
                   '--n_pr', '6',
                   '--epochs', '1000',
                   '--batch', '40',
                   '--lr', '2',
                   '--lr_decay_epochs', '200',
                   '--use_gpu', 'True',
                   '--lr_scheduler', 'CosialingLR',
                   '--s_para', 'False']

    opts = parser.parse_args(arg_distill)
    opts.s_para = True

    # train_data, test_data, responses = load_s(opts.n_pr)
    # loader_train_sample = DataLoader(train_data, batch_size=opts.batch, num_workers=8)
    # loader_train_eval = DataLoader(test_data, shuffle=False, batch_size=opts.batch, drop_last=False, num_workers=8)

    # from model_pr_mul import tf2s
    # net = tf2s(10).cuda()
    # out = net(torch.tensor(train_tfc).cuda(), freq)

    # model
    if opts.use_gpu:
        model = MMANN(n_out=opts.n_pr).cuda()
    else:
        model = MMANN(n_out=opts.n_pr)

    modules = model.named_children()

    def hook_fn_backward(module, grad_input, grad_output):
        print(module)  # 为了区分模块
        # 为了符合反向传播的顺序，我们先打印 grad_output
        print('grad_output',grad_output)
        # 再打印 grad_input
        print('grad_input',grad_input)
        # 保存到全局变量
        ng = []
        for n, g in enumerate(grad_input):
            ng.append( g*10)
        return tuple(ng)

    modules = model.named_children()
    # for name, module in modules:
    #     if name in ['P','R']:
    #         module.register_backward_hook(hook_fn_backward)

    # model.load_state_dict(torch.load(r'./geo2tfc_4/best-epoch_799-0.0431.pth'))
    # model.load_state_dict(torch.load('last-2.pth'))
    if False:
        train_data, test_data, responses = load_ann_pr(opts.n_pr, True)
        freq = responses[0]
        train_tfc = responses[1]
        test_tfc = responses[2]

        model.load_state_dict(torch.load(r'./geo2tfc_4/best-epoch_799-0.0431.pth'))
        model.eval()
        pr_tf, labels = test_data[:]
        pr_tf, labels = pr_tf.cuda().float(), labels.cuda().float()

        freq = freq*0.01 + 10

        output, pr = model(pr_tf, freq=freq, ann_out=False)
        mape = torch.mean(torch.abs((torch.stack(output) - labels) / labels)) * 100
        print("Current test mape: ", mape)

        fig = plot_n_s11([labels[0].cpu().detach().numpy(), output[0].cpu().detach().numpy()], freq)
        plt.show()
        return 0


    # load data
    train_data, test_data, responses = load_ann_pr(opts.n_pr, opts.s_para)
    if opts.s_para:
        freq = responses[0] * 0.01 + 10
        train_tfc = responses[1]
        test_tfc = responses[2]

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)

    if opts.lr_scheduler =='CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 , eta_min=0)
        print("Current Learning Rate Scheduler: CosineAnnealingLR.", lr_scheduler)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500, 700, 900], gamma=opts.lr_decay_gamma, last_epoch=-1)
        print("Default Learning Rate Scheduler: MultiStepLR", lr_scheduler)

    criterion = torch.nn.MSELoss()
    criterion_ann = torch.nn.MSELoss()

    # logger
    writer = SummaryWriter(comment='-exp1')
    # dummy_input = torch.rand(20, 3).cuda()
    # writer.add_graph(model, (dummy_input,))

    def train(loader, ep):
        model.train()

        pr_tf, labels = loader[:]

        if opts.use_gpu:
            pr_tf, labels = pr_tf.cuda().float(), labels.cuda().float()
        else:

            pr_tf, labels = pr_tf.float(), labels.float()

        if opts.s_para:
            outputs, pr = model(pr_tf, freq=freq, ann_out=False)
            outputs = torch.stack(outputs)
        else:
            outputs = model(pr_tf)

        # ori loss
        outputs_abs = torch.log10(torch.sqrt(outputs[:,:,0]**2+outputs[:,:,1]**2))*20
        labels_abs = torch.log10(torch.sqrt(labels[:,:,0]**2+labels[:,:,1]**2))*20
        loss = criterion(outputs_abs.cuda().float(), labels_abs)

        # loss = criterion(outputs.cuda().float(), labels)
        # loss_ann = criterion_ann(pr.cuda(), torch.tensor(train_tfc).cuda().float())
        # # loss_t = 0.8*loss + 0.2*loss_ann
        #
        loss_t = loss

        # use mape as loss
        # loss_t = torch.mean(torch.abs((outputs - labels) / labels)) * 100

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        print('[Epoch %d] Loss: %.5f\n' %(ep, loss_t.cpu().detach().numpy()))
        # mape = torch.mean(torch.abs((outputs - labels) / labels)) * 100
        # print("Current train mape: ", mape)
        # writer.add_scalar('mape/train', mape, ep)
        writer.add_scalar('data/loss_all', loss_t.cpu().detach().numpy(), ep)


    def eval(net, loader, ep):
        net.eval()
        embeddings_all, labels_all = [], []

        with torch.no_grad():
            pr_tf, labels = loader[:]

            if opts.use_gpu:
                pr_tf, labels = pr_tf.cuda().float(), labels.cuda().float()
            else:
                pr_tf, labels = pr_tf.float(), labels.float()

            if opts.s_para:
                output, pr = net(pr_tf, freq=freq, ann_out=False)
                output = torch.stack(output)
            else:
                output = net(pr_tf)

            # output = net(pr_tf)
            embeddings_all.append(output.data)
            labels_all.append(labels.data)

            embeddings_all = torch.cat(embeddings_all).cpu()
            labels_all = torch.cat(labels_all).cpu()
            rec = criterion(embeddings_all, labels_all)

            print('[Epoch %d] Recall: [%.4f]\n' % (ep, rec.item()))
            # mape = torch.mean(torch.abs((output - labels) / labels)) * 100
            # print("Current test mape: ", mape)
            # writer.add_scalar('mape/test', mape, ep)

            if opts.s_para:
                for n_plot in [0, -1]:
                    fig = plot_n_s11([labels[n_plot].cpu().detach().numpy(), output[n_plot].cpu().detach().numpy()], freq)
                    writer.add_figure('Image%d'%n_plot, fig, ep)
                    plt.close(fig)
            else:
                s11_resp = freq_resp(output[0].cpu().detach().numpy(), opts.n_pr, responses[0][0][:, 0])
                plot_comparision_s11(responses[0][-1][:, 1:], s11_resp, responses[0][0][:, 0], title="train_last")
                plt.savefig("test.jpg")
                plt.close()
        # print(np.sum(np.power(embeddings_all-labels_all, 2).numpy(),1))
        # print("Current MSE: ", rec.item())
        return rec.item()

    best_train_rec = 1
    for epoch in range(1, opts.epochs+1):
        train(train_data, epoch)
        writer.add_scalar('hyperparameter/lr', lr_scheduler.get_lr(), epoch)
        lr_scheduler.step()
        train_recall = eval(model, test_data, epoch)

        writer.add_scalar('data/train_recall', train_recall, epoch)

        if best_train_rec > train_recall:
            best_train_rec = train_recall

            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "best-epoch_%d-%.4f.pth"%(epoch, train_recall)))

        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write('Best Train Recall@1: %.4f\n' % (best_train_rec * 100))

        print("Best Train Recall: %.4f" % best_train_rec)


if __name__ == "__main__":
    main()


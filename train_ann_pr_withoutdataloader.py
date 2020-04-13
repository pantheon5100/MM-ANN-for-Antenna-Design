import argparse
import os

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

import matplotlib.pyplot as plt

from model.model_pr import MMANN
from data import load_ann_pr
from utlis.parser_argparse import get_parser
from utlis.transferfunction import freq_resp, plot_comparision_s11

import matlab.engine as meng

def main():
    parser = get_parser()
    arg_distill = ['--save_dir', 'geo2tfc',
                   '--n_pr', '3',
                   '--epochs', '5000',
                   '--batch', '6',
                   '--lr', '1e-1',
                   '--lr_decay_epochs', '10',
                   '--use_gpu', 'True',
                   '--lr_scheduler', 'CosineAnnealingLR']

    opts = parser.parse_args(arg_distill)

    # load data
    train_data, test_data, responses = load_ann_pr(opts.n_pr)
    # loader_train_sample = DataLoader(train_data, batch_size=opts.batch, num_workers=8)
    # loader_train_eval = DataLoader(test_data, shuffle=False, batch_size=opts.batch, drop_last=False, num_workers=8)

    # model
    if opts.use_gpu:
        model = MMANN(n_out=opts.n_pr).cuda()
    else:
        model = MMANN(n_out=opts.n_pr)

    model.load_state_dict(torch.load('last-2.pth'))

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)

    if opts.lr_scheduler =='CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        print("Current Learning Rate Scheduler: CosineAnnealingLR.", lr_scheduler)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)
        print("Default Learning Rate Scheduler: MultiStepLR", lr_scheduler)

    criterion = torch.nn.MSELoss()

    # logger
    writer = SummaryWriter(comment='-exp1')
    dummy_input = torch.rand(20, 3).cuda()
    writer.add_graph(model, (dummy_input,))

    def train(loader, ep):
        model.train()
        loss_all = []
        train_iter = tqdm(loader, ascii=True)
        times = int(len(loader)/opts.batch)
        for batch in range(times+1):
            if batch==times :
                if (len(loader)-opts.batch*batch)>1:
                    pr_tf, labels = loader[opts.batch * batch:]
                else:
                    continue
            else:
                pr_tf, labels = loader[opts.batch*batch:opts.batch*(batch+1)]
            # print(batch, pr_tf, labels, len(pr_tf))
        # for pr_tf, labels in train_iter:
            # pr_tf, labels = pr_tf.cuda(), labels.cuda()

            if opts.use_gpu:
                pr_tf, labels = pr_tf.cuda().float(), labels.cuda().float()
            else:

                pr_tf, labels = pr_tf.float(), labels.float()

            outputs = model(pr_tf)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())

            train_iter.set_description("[Train][Epoch %d] loss: %.5f" %(ep, loss.item()))
        print('[Epoch %d] Loss: %.5f\n' %(ep, torch.Tensor(loss_all).mean()))

        writer.add_scalar('data/loss_all', torch.Tensor(loss_all).mean(), ep)


    def eval(net, loader, ep):
        net.eval()
        test_iter = tqdm(loader, ascii=True)
        embeddings_all, labels_all = [], []

        with torch.no_grad():
            pr_tf, labels = loader[:]

            if opts.use_gpu:
                pr_tf, labels = pr_tf.cuda().float(), labels.cuda().float()
            else:
                pr_tf, labels = pr_tf.float(), labels.float()
            output = net(pr_tf)
            embeddings_all.append(output.data)
            labels_all.append(labels.data)
            test_iter.set_description("[Eval][Epoch %d]" % ep)

            embeddings_all = torch.cat(embeddings_all).cpu()
            labels_all = torch.cat(labels_all).cpu()
            rec = criterion(embeddings_all, labels_all)

            print('[Epoch %d] Recall: [%.4f]\n' % (ep, rec.item()))
            s11_resp = freq_resp(output[0].cpu().detach().numpy(), opts.n_pr, responses[0][0][:, 0])
            plot_comparision_s11(responses[0][-1][:, 1:], s11_resp, responses[0][0][:, 0], title="train_last")
            plt.savefig("test.jpg")
            plt.close()
        print(np.sum(np.power(embeddings_all-labels_all, 2).numpy(),1))
        print("Current MSE: ", rec.item())
        return rec.item()

    best_train_rec = 100
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
                torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))

        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write('Best Train Recall@1: %.4f\n' % (best_train_rec * 100))

        print("Best Train Recall: %.4f" % best_train_rec)


if __name__ == "__main__":
    main()


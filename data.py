from scipy import io as sio
import numpy as np
import torch


def load_data(path, names):
    '''
    path is the directory of .mat file
    names is a list with 2 elements (input data and label)
    '''

    data_s = sio.loadmat(path)
    data = []
    for name in names:
        data.append(np.squeeze(np.array(data_s[name])))
    if len(data)==1:
        return data[0]
    return data


def load_ann_pr(n_pr, load_s_p=False):
    data_dir = './matlab/data4/'
    # train_data
    train_data = load_data(path='./matlab/data/' + 'Training_Data.mat', names=['candidates'])
    train_responses = load_data(path='./matlab/data/' + 'Training_Data.mat', names=['responses'])
    train_index = load_data(path=data_dir + 'train_index.mat', names=['Index'])
    train_label = load_data(path=data_dir + 'train_pr%d.mat' % n_pr, names=['pr_ac'])
    # test_data
    test_data = load_data(path='./matlab/data/' + 'Test_Data.mat', names=['test_candidates'])
    test_responses = load_data(path='./matlab/data/' + 'Test_Data.mat', names=['test_responses'])
    test_label = load_data(path=data_dir + 'test_pr%d.mat' % n_pr, names=['pr_ac'])
    test_index = load_data(path=data_dir + 'test_index.mat', names=['Index'])

    train_d = torch.from_numpy(train_data[train_index == n_pr])
    train_l = torch.tensor([x[:, 1:] for x in train_responses])[train_index == n_pr]
    test_d = torch.from_numpy(test_data[test_index == n_pr])
    test_l = torch.tensor([x[:, 1:] for x in test_responses])[test_index == n_pr]

    print("Number of images in Training Set: %d" % len(train_l))
    print("Number of images in Testing Set: %d" % len(test_l))

    if load_s_p:
        return torch.utils.data.TensorDataset(train_d, train_l), \
               torch.utils.data.TensorDataset(test_d, test_l), \
               [train_responses[0][:, 0], train_label, test_label]
    else:
        return torch.utils.data.TensorDataset(torch.from_numpy(train_data[train_index==n_pr]), torch.tensor(train_label)),\
               torch.utils.data.TensorDataset(torch.from_numpy(test_data[test_index==n_pr]), torch.tensor(test_label)), \
                [train_responses, test_responses]

def load_s(n_pr):
    data_dir = './matlab/data/'
    # train_data
    train_data = load_data(path=data_dir+'Training_Data.mat', names=['candidates'])
    train_responses = load_data(path=data_dir+'Training_Data.mat', names=['responses'])
    train_index = load_data(path=data_dir+'train_index.mat', names=['Index'])
   # test_data
    test_data = load_data(path=data_dir+'Test_Data.mat', names=['test_candidates'])
    test_responses = load_data(path=data_dir+'Test_Data.mat', names=['test_responses'])

    test_index = load_data(path=data_dir+'test_index.mat', names=['Index'])


    train_d = torch.from_numpy(train_data[train_index==n_pr])
    train_l = torch.tensor([x[:, 1:] for x in train_responses])[train_index==n_pr]
    test_d = torch.from_numpy(test_data[test_index==n_pr])
    test_l = torch.tensor([x[:, 1:] for x in test_responses])[test_index==n_pr]

    print("Number of images in Training Set: %d" % len(train_l))
    print("Number of images in Testing Set: %d" % len(test_l))

    return torch.utils.data.TensorDataset(train_d, train_l),\
           torch.utils.data.TensorDataset(test_d, test_l), \
           train_responses[0][:, 0]


if __name__ == "__main__":
    train_data, test_data, responses = load_ann_pr(3)

    import torch
    ac = load_data(path='./matlab/data/train_pr%d.mat' % 3, names=['pr_ac'])
    res, cand = load_data(path='./matlab/data/Training_Data.mat', names=['responses', 'candidates'])
    ac_cal = torch.tensor(ac[-1, :])
    freq = torch.tensor(res[-1][:, 0].copy())
    s_para_o = torch.tensor([x[:, 1:] for x in res]).float()

    from utlis.transferfunction import *
    s_para = freq_resp(ac_cal, 3, freq)
    s_para = torch.tensor(s_para)

    from model.model_pr import MMANN
    model = MMANN()
    out = model(torch.from_numpy(cand).float(), freq=freq, ann_out=False)

    loss = torch.nn.MSELoss()
    l = loss(out, s_para_o)
    l.backward()
    print(l)

    plot_comparision_s11(s_para_o[0], s_para[0], freq)

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--n_pr', default=3, type=int)
    parser.add_argument('--use_gpu', default=False, type=bool)
    parser.add_argument('--lr_scheduler', default=None, type=str)
    parser.add_argument('--s_para', default=False, type=bool)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--data', default='data')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch', default=6, type=int)
    parser.add_argument('--iter_per_epoch', default=100, type=int)
    parser.add_argument('--lr_decay_epochs', type=int, default=[4, 6], nargs='+')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--load', default=None)
    return parser

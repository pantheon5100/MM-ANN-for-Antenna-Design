import numpy as np
import matplotlib.pyplot as plt


def freq_resp(tfc, n_pr, freq):
    """
    Like the function in matlab.

    Compute the frequency response of RATIONALFIT function output(A, C). In our case, A and C are effective parameters
    in paper Parametric Modeling of EM Behavior of Microwave Components Using Combined Neural Networks and
    Pole-Residue-Based Transfer Functions.

    Parameters
    ----------
    tfc : array_like
        Contained the pole(A) and residue(C) coefficients of transfer function(TF).
        For example [[Ar Ai Cr Ci], ...[...]]
    n_pr : int
        The order of TF.
    freq : array_like
        Frequency

    Returns
    -------
    freq resp : ndarray
        The frequency response of effective TF coefficients

    """
    # frequency scaling and shifting
    freq = freq + 4.5e9

    res = []
    for n, f in enumerate(freq):
        tmp_r = 0
        tmp_i = 0
        for ar, ai, cr, ci in zip(tfc[:n_pr], tfc[n_pr:n_pr * 2], tfc[n_pr * 2:n_pr * 3], tfc[n_pr * 3:]):
            d = 2 * np.pi * f - ai
            c2_d2 = ar ** 2 + d ** 2
            ac_bd = -cr * ar + ci * d
            bc_ad = -ci * ar - cr * d
            tmp_r = ac_bd / c2_d2 + tmp_r
            tmp_i = bc_ad / c2_d2 + tmp_i
            if ai != 0:
                ai = -ai
                ci = -ci
                d = 2 * np.pi * f - ai
                c2_d2 = ar ** 2 + d ** 2
                ac_bd = -cr * ar + ci * d
                bc_ad = -ci * ar - cr * d
                tmp_r = ac_bd / c2_d2 + tmp_r
                tmp_i = bc_ad / c2_d2 + tmp_i

        res.append([tmp_r, tmp_i])
    return res


def freq_resps(tfc, n_pr, freq):
    """
    Like the function in matlab.

    Compute the frequency response of RATIONALFIT function output(A, C). In our case, A and C are effective parameters
    in paper Parametric Modeling of EM Behavior of Microwave Components Using Combined Neural Networks and
    Pole-Residue-Based Transfer Functions.

    Parameters
    ----------
    tfc : array_like
        Contained the pole(A) and residue(C) coefficients of transfer function(TF).
        For example [[Ar Ai Cr Ci], ...[...]]
    n_pr : int
        The order of TF.
    freq : array_like
        Frequency

    Returns
    -------
    freq resp : ndarray
        The frequency response of effective TF coefficients

    """
    # frequency scaling and shifting
    freq = freq + 4.5e9

    res = []
    for n, f in enumerate(freq):
        tmp_r = 0
        tmp_i = 0
        for ar, ai, cr, ci in zip(tfc[:n_pr], tfc[n_pr:n_pr * 2], tfc[n_pr * 2:n_pr * 3], tfc[n_pr * 3:]):
            d = 2 * np.pi * f - ai
            c2_d2 = ar ** 2 + d ** 2
            ac_bd = -cr * ar + ci * d
            bc_ad = -ci * ar - cr * d
            tmp_r = ac_bd / c2_d2 + tmp_r
            tmp_i = bc_ad / c2_d2 + tmp_i
            if ai != 0:
                ai = -ai
                ci = -ci
                d = 2 * np.pi * f - ai
                c2_d2 = ar ** 2 + d ** 2
                ac_bd = -cr * ar + ci * d
                bc_ad = -ci * ar - cr * d
                tmp_r = ac_bd / c2_d2 + tmp_r
                tmp_i = bc_ad / c2_d2 + tmp_i

        res.append([tmp_r, tmp_i])
    return res


def plot_comparision_s11(s_1, s_2, freq, label1='EM', label2='Model', title='S11 magnitude'):
    if not isinstance(s_1, np.ndarray):
        s_1 = np.array(s_1)
    if not isinstance(s_2, np.ndarray):
        s_2 = np.array(s_2)
    abs_1 = np.abs(s_1[:, 0]+1j*s_1[:, 1])
    abs_2 = np.abs(s_2[:, 0]+1j*s_2[:, 1])
    freq = 0.01*freq + 10

    fig = plt.figure()
    plt.plot(freq, abs_1, 'b', label=label1)
    plt.plot(freq, abs_2, 'r--', label=label2)
    plt.xlabel('Freq. in GHz')
    plt.ylabel('S11 in dB')
    plt.legend()
    plt.title(title)
    return fig
    # plt.show()

def plot_n_s11(s_para, freq, labels=None, linetypes=None, title='S11 magnitude'):
    fig = plt.figure()
    if linetypes is None:
        linetypes = ['b', 'r--', 'y--']
    if labels == None:
        labels = ['EM', 'Model', 'TFC']
    for s_1, label, linetype in zip(s_para, labels, linetypes):
        if not isinstance(s_1, np.ndarray):
            s_1 = np.array(s_1)
        abs_1 = np.log10(np.abs(s_1[:, 0]+1j*s_1[:, 1]))*20
        plt.plot(freq, abs_1, linetype, label=label)

    plt.xlabel('Freq. in GHz')
    plt.ylabel('S11 in dB')
    plt.legend()
    plt.title(title)
    # plt.show()
    return fig

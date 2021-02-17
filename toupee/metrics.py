#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy as np # type: ignore
from sklearn.calibration import calibration_curve # type: ignore


def ece_binary(prob_true, prob_pred, bin_sizes):
    ece = 0
    for m in np.arange(len(bin_sizes)):
        ece = ece + (bin_sizes[m] / sum(bin_sizes)) * np.abs(prob_true[m] - prob_pred[m])
    return ece


def mce_binary(prob_true, prob_pred, bin_sizes):
    mce = 0
    for m in np.arange(len(bin_sizes)):
        mce = max(mce, np.abs(prob_true[m] - prob_pred[m]))
    return mce


def rmsce_binary(prob_true, prob_pred, bin_sizes):
    rmsce = 0
    for m in np.arange(len(bin_sizes)):
        rmsce = rmsce + (bin_sizes[m] / sum(bin_sizes)) * (prob_true[m] - prob_pred[m]) ** 2
    return np.sqrt(rmsce)


def calibration(y_true, y_pred, n_bins=10):
    ece_bin = []
    mce_bin = []
    rmsce_bin = []
    for a_class in range(y_true.shape[1]):
        prob_true, prob_pred = calibration_curve(y_true[a_class], y_pred[a_class], n_bins)
        bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
        ece_bin.append(ece_binary(prob_true, prob_pred, bin_sizes))
        mce_bin.append(mce_binary(prob_true, prob_pred, bin_sizes))
        rmsce_bin.append(rmsce_binary(prob_true, prob_pred, bin_sizes))
    return {'ece': np.mean(ece_bin),
            'mce': np.mean(mce_bin),
            'rmsce': np.mean(rmsce_bin)
    }
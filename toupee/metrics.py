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
import toupee as tp

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
        prob_true, prob_pred = calibration_curve(y_true=y_true[a_class], y_prob=y_pred[a_class], n_bins=n_bins)
        bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
        ece_bin.append(ece_binary(prob_true, prob_pred, bin_sizes))
        mce_bin.append(mce_binary(prob_true, prob_pred, bin_sizes))
        rmsce_bin.append(rmsce_binary(prob_true, prob_pred, bin_sizes))
    return {'ece': np.mean(ece_bin),
            'mce': np.mean(mce_bin),
            'rmsce': np.mean(rmsce_bin)
    }


def _to_numpy(tensor):
    """Convert a tensor (PyTorch or numpy) to numpy array."""
    if hasattr(tensor, 'numpy'):
        # PyTorch tensor
        return tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
    return np.asarray(tensor)


def evaluate(model, test_data, adversarial_gradient_source=None):
    """ Evaluate model on some test data handle """
    #TODO: update for different data formats
    all_y_pred = []
    all_y_true = []
    all_y_pred_onehot = []
    all_y_true_onehot = []
    all_x = []
    all_adversarial = []
    for (x, y_true) in test_data:
        # Convert to numpy if needed (PyTorch DataLoader returns tensors)
        x_np = _to_numpy(x)
        y_true_np = _to_numpy(y_true)

        # For image data, DataLoader returns NCHW, need NHWC for predict methods
        if x_np.ndim == 4 and x_np.shape[1] in (1, 3, 4):  # Likely NCHW
            x_np = np.transpose(x_np, (0, 2, 3, 1))  # NCHW -> NHWC

        all_x.append(x_np)
        all_y_pred.append(model.predict_classes(x_np))
        all_y_pred_onehot.append(model.predict_proba(x_np))
        all_y_true.append(np.argmax(y_true_np, axis=1))
        all_y_true_onehot.append(y_true_np)
        if adversarial_gradient_source:
            all_adversarial.append(tp.adversarial.FGSM(adversarial_gradient_source, x_np, y_true_np))
    x = np.concatenate(all_x)
    if adversarial_gradient_source:
        adversarial_perturbation = np.concatenate(all_adversarial)
    y_pred = np.concatenate(all_y_pred)
    y_true = np.concatenate(all_y_true)
    y_pred_onehot = np.concatenate(all_y_pred_onehot)
    y_true_onehot = np.concatenate(all_y_true_onehot)
    scores = tp.utils.eval_scores(y_true=y_true, y_pred=y_pred, y_true_onehot=y_true_onehot, y_pred_onehot=y_pred_onehot)
    if adversarial_gradient_source:
        adversarial_scores = {}
        for epsilon in tp.ADVERSARIAL_EPSILONS:
            adversarial_x = x + epsilon * adversarial_perturbation
            y_adv = model.predict_classes(adversarial_x)
            y_adv_onehot = model.predict_proba(adversarial_x)
            adversarial_scores[str(epsilon)] = tp.utils.eval_scores(y_true, y_adv, y_true_onehot, y_adv_onehot)
        scores['adversarial'] = adversarial_scores
    return scores

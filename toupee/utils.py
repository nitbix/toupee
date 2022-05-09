"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import os
import sys
import logging
import sklearn.metrics # type: ignore
import numpy as np # type: ignore
import toupee as tp
from toupee.metrics import calibration

# this is needed because PowerShell does not understand colour escapes
if sys.platform == 'win32': 
    try: 
        import colorama # type: ignore
    except ImportError: 
        pass 
    else: 
        colorama.init() 

def dict_map(dictionary, f):
    return {k: f(v) for k, v in dictionary.items()}


def eval_scores(y_true, y_pred, y_true_onehot=None, y_pred_onehot=None):
    """ Calculate all the eval scores we want and return them in a dict """
    scores = {  'classification_report': sklearn.metrics.classification_report(y_true, y_pred),
                'accuracy_score': sklearn.metrics.accuracy_score(y_true, y_pred),
                'micro_precision_score': sklearn.metrics.precision_score(y_true, y_pred, average="micro"),
                'micro_recall_score': sklearn.metrics.recall_score(y_true, y_pred, average="micro"),
                'micro_f1_score': sklearn.metrics.f1_score(y_true, y_pred, average="micro"),
                'macro_precision_score': sklearn.metrics.precision_score(y_true, y_pred, average="macro"),
                'macro_recall_score': sklearn.metrics.recall_score(y_true, y_pred, average="macro"),
                'macro_f1_score': sklearn.metrics.f1_score(y_true, y_pred, average="macro"),
                'confusion_matrix': sklearn.metrics.confusion_matrix(y_true, y_pred),
                'y_true': y_true,
                'y_pred' : y_pred,
    }
    if y_true_onehot is not None and y_pred_onehot is not None:
        scores['calibration'] = calibration(y_true_onehot, y_pred_onehot)
    return scores


def get_colour_string(string, target_colour, use_colours=True):
    '''
    Returns an ANSI coloured string - the name of this function is short for
        readability, when it is called in a print
    '''
    if not use_colours:
        return string

    output = ''
    if target_colour == 'g':
        #green
        output = '\u001b[38;5;41m' + string
    elif target_colour == 'b':
        #blue
        output = '\u001b[34;1m' + string
    elif target_colour == 'r':
        #red
        output = '\u001b[31;1m' + string
    elif target_colour == 'reverse':
        #reversed colours (background and text colour)
        output = '\u001b[7m' + string
    else:
        logging.warning('Unrecognised colour %s. Printing the plain string', target_colour)
    return output + '\u001b[0m'

def pretty_print_confusion_matrix(conf_matrix, cm_decimals=3):
    """
    Given a string output from confusion_matrix(), print it colour coded
    """
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum()
    for row in range(conf_matrix.shape[0]):
        #for each class (row), if the most likely prediction (column)
        #   matches the true class, modify its colour to green. Otherwise,
        #   modify the colour of the most likely class to red.
        most_likely_class = np.argmax(conf_matrix[row, :])
        for col in range(conf_matrix.shape[1]):
            print_string = ' {0:7.' + str(cm_decimals) + '%} '
            #most_likely_class modifier
            if col == most_likely_class:
                if col == row:
                    print_string = get_colour_string(print_string, 'g')
                else:
                    print_string = get_colour_string(print_string, 'r')
            #diagonal -> reverses colour
            if col == row:
                print_string = get_colour_string(print_string, 'reverse')
            print(print_string.format(conf_matrix[row, col]), end='')
        print("\n", end='')
    print("LEGEND: rows = true class; columns = predicted class;")
    print("        colour = most likely prediction for each row;")

def _log_metrics(metrics) -> None:
    for metric_name in tp.PRINTABLE_METRICS:
        logging.info(f"{metric_name}: {metrics[metric_name]}")

def log_metrics(metrics) -> None:
    logging.info('\n{:*^40}'.format(" Classification Report "))
    logging.info(metrics['classification_report'])
    logging.info('\n{:*^40}'.format(" Confusion Matrix "))
    pretty_print_confusion_matrix(metrics['confusion_matrix'])
    logging.info('\n{:*^40}'.format(" Metrics "))
    _log_metrics(metrics)
    if 'adversarial' in metrics:
        logging.info('\n{:*^40}'.format(" Adversarial Metrics "))
        for epsilon in metrics['adversarial']:
            logging.info(f"** Epsilon = {epsilon}")
            _log_metrics(metrics['adversarial'][epsilon])

def replace_inbound_layer(layer_list, old_value, new_value):
    new_list = []
    for a in layer_list:
        a_acc = []
        for b in a:
            new_b = b
            b[0] = new_value if b[0] == old_value else b[0]
            a_acc.append(new_b)
        new_list.append(a_acc)
    return new_list
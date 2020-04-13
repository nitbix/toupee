"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import os
import sys
import sklearn.metrics
import numpy as np


if sys.platform == 'win32': 
    try: 
        import colorama 
    except ImportError: 
        pass 
    else: 
        colorama.init() 

def eval_scores(y_true, y_pred):
    """ Calculate all the eval scores we want and return them in a dict """
    return {'classification_report': sklearn.metrics.classification_report(y_true, y_pred),
            'accuracy_score': sklearn.metrics.accuracy_score(y_true, y_pred),
            'micro_precision_score': sklearn.metrics.precision_score(y_true, y_pred, average="micro"),
            'micro_recall_score': sklearn.metrics.recall_score(y_true, y_pred, average="micro"),
            'micro_f1_score': sklearn.metrics.f1_score(y_true, y_pred, average="micro"),
            'macro_precision_score': sklearn.metrics.precision_score(y_true, y_pred, average="macro"),
            'macro_recall_score': sklearn.metrics.recall_score(y_true, y_pred, average="macro"),
            'macro_f1_score': sklearn.metrics.f1_score(y_true, y_pred, average="macro"),
            'confusion_matrix': sklearn.metrics.confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
    }

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

# def gauss(x, y, sigma=2.0):
#     Z = 2 * np.pi * sigma**2
#     return  1./Z * np.exp(-(x**2 + y**2) / (2. * sigma**2))
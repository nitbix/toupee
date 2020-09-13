#!/usr/bin/python
"""
Run an ensemble experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import argparse
import toupee as tp
import dill
import logging

PRINTABLE_METRICS = ['accuracy_score',
                     'micro_precision_score',
                     'micro_recall_score',
                     'micro_f1_score',
                     'macro_precision_score',
                     'macro_recall_score',
                     'macro_f1_score',
                     ]

def main(args=None, params=None):
    """ Train a base model as specified """
    if args is None:
        parser = argparse.ArgumentParser(description='Train a single Base Model')
        parser.add_argument('params_file', help='the parameters file')
        parser.add_argument('save_file', nargs='?',
                            help='the file where the trained MLP is to be saved')
        parser.add_argument('--epochs', type=int, nargs='?',
                            help='number of epochs to run')
        args = parser.parse_args()
    logging.info(("using toupee version {0}".format(tp.version)))
    if not params:
        params = tp.config.load_parameters(args.params_file)
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    method = tp.ensembles.create(params, data)
    metrics = method.fit()
    logging.info('\n{:*^40}'.format(' Ensemble trained in %.2fm ' % (metrics['time'] / 60.)))
    logging.info(metrics['ensemble']['classification_report'])
    tp.utils.pretty_print_confusion_matrix(metrics['ensemble']['confusion_matrix'])
    logging.info('\n{:*^40}'.format(" Aggregate Metrics "))
    for metric_name in PRINTABLE_METRICS:
        logging.info('** {}: {:02f}'.format(metric_name, metrics['ensemble'][metric_name]))
    logging.info('\n{:*^40}'.format(" Member Metrics "))
    for metric_name in PRINTABLE_METRICS:
        logging.info('** {}'.format(metric_name))
        logging.info(metrics['members'][metric_name].tolist())
    if args.save_file:
        method.save(args.save_file)
        dill.dump(metrics, args.save_file + '.metrics')
    #save_metadata_etc


if __name__ == '__main__':
    main()
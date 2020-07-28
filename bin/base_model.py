#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import argparse
import toupee as tp

def main(args=None):
    """ Train a base model as specified """
    if args is None:
        parser = argparse.ArgumentParser(description='Train a single Base Model')
        parser.add_argument('params_file', help='the parameters file')
        parser.add_argument('save_file', nargs='?',
                            help='the file where the trained MLP is to be saved')
        parser.add_argument('--epochs', type=int, nargs='?',
                            help='number of epochs to run')
        args = parser.parse_args()
    print(("using toupee version {0}".format(tp.version)))
    params = tp.config.load_parameters(args.params_file)
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    base_model = tp.model.Model(params=params)
    base_model.fit(data=data)
    print(base_model.test_metrics['classification_report'])
    tp.utils.pretty_print_confusion_matrix(base_model.test_metrics['confusion_matrix'])
    if args.save_file:
        base_model.save(args.save_file)

if __name__ == '__main__':
    main()

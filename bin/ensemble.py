#!/usr/bin/python
"""
Run an ensemble experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import os
import argparse
import toupee as tp
import dill
import logging
import wandb


def main(args=None, params=None):
    """ Train a base model as specified """
    parser = argparse.ArgumentParser(description='Train a single Base Model')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('save_file', nargs='?',
                        help='the file where the trained MLP is to be saved')
    parser.add_argument('--epochs', type=int, nargs='?',
                        help='number of epochs to run')
    parser.add_argument('--wandb', action="store_true",
                        help="Send results to Weights and Biases")
    parser.add_argument('--wandb-project', type=str, help="Weights and Biases project name")
    parser.add_argument('--wandb-group', type=str, help="Weights and Biases group name")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logging.info(("using toupee version {0}".format(tp.version)))
    if not params:
        params = tp.config.load_parameters(args.params_file)
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    wandb_params = None
    if args.wandb:
        dataset_name = os.path.basename(os.path.normpath(params.dataset))
        wandb_project = args.wandb_project or f"toupee-{dataset_name}"
        group_id = wandb.util.generate_id()
        wandb_group = args.wandb_group or f"{dataset_name}-{params.ensemble_method['class_name']}-{group_id}"
        wandb_params = {"project": wandb_project, "group": wandb_group}
    method = tp.ensembles.create(params=params, data=data, wandb=wandb_params)
    metrics = method.fit()
    logging.info('\n{:*^40}'.format(' Ensemble trained in %.2fm ' % (metrics['time'] / 60.)))
    logging.info(metrics['ensemble']['classification_report'])
    tp.utils.pretty_print_confusion_matrix(metrics['ensemble']['confusion_matrix'])
    if args.wandb:
        run = wandb.init(project=wandb_project, reinit=True,
                    config={"type": "ensemble", "params": params.__dict__},
                    group=wandb_group,
                    name="finished-ensemble")
        for i, member_metrics in enumerate(metrics['members'].to_dict('records')):
            wandb.log({k: v for k, v in member_metrics.items() if k in tp.PRINTABLE_METRICS}, commit=False)
            wandb.log({f"ensemble_{k}": v for k, v in metrics['round_cumulative'][i].items()
                       if k in tp.PRINTABLE_METRICS}, commit=False)
            wandb.log({'member': i, 'step': i, 'epoch': i})
        for metric, value in metrics['ensemble'].items():
            wandb.run.summary[metric] = value
        wandb.run.summary['total time'] = metrics['time']
        run.finish()
    logging.info('\n{:*^40}'.format(" Aggregate Metrics "))
    for metric_name in tp.PRINTABLE_METRICS:
        logging.info(f"{metric_name}: {metrics['ensemble'][metric_name]}")
    logging.info('\n{:*^40}'.format(" Member Metrics "))
    for metric_name in tp.PRINTABLE_METRICS:
        logging.info(f"{metric_name}: {metrics['members'][metric_name].tolist()}")
    if args.save_file:
        method.save(args.save_file)
        dill.dump(metrics, args.save_file + '.metrics')
    #save_metadata_etc


if __name__ == '__main__':
    main()
#!/usr/bin/python
"""
Run an ensemble experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import numpy
import argparse
import os
import re
import dill
from toupee.common import accuracy, euclidian_distance, relative_distance

from pymongo import MongoClient
import numpy as np
import datetime
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a single MLP')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('save_file', nargs='?',
                        help='the file where the trained MLP is to be saved')
    parser.add_argument('--seed', type=int, nargs='?', default=42,
                        help='random seed to use for this sim')
    parser.add_argument('--epochs', type=int, nargs='?',
                        help='number of epochs to run')
    parser.add_argument('--results-db', nargs='?',
                        help='mongodb db name for storing results')
    parser.add_argument('--results-host', nargs='?',
                        help='mongodb host name for storing results')
    parser.add_argument('--results-table', nargs='?',
                        help='mongodb table name for storing results')
    parser.add_argument('--results-dep', nargs='?',
                        help='mongodb table name for dependencies')
    parser.add_argument('--device', nargs='?',
                        help='gpu/cpu device to use for training')
    parser.add_argument('--dump-shapes-to', type=str, nargs='?', default=42,
                        help='location where to save the shape of the ensemble members')
    parser.add_argument('--dump-to', type=str, nargs='?', default='ensemble.pkl',
                        help='location where to save the ensemble')

    args = parser.parse_args()
    #this needs to come before all the toupee and theano imports
    #because theano starts up with gpu0 and never lets you change it
    if args.device is not None:
        if 'THEANO_FLAGS' in os.environ is not None:
            env = os.environ['THEANO_FLAGS']
            env = re.sub(r'/device=[a-zA-Z0-9]+/',r'/device=' + args.device, env)
        else:
            env = 'device=' + args.device
        os.environ['THEANO_FLAGS'] = env

    arg_param_pairings = [
        (args.results_db, 'results_db'),
        (args.results_host, 'results_host'),
        (args.results_table, 'results_table'),
        (args.results_dep, 'results_dep'),
        (args.epochs, 'n_epochs'),
    ]
    
    if 'seed' in args.__dict__:
        print(("setting random seed to: {0}".format(args.seed)))
        numpy.random.seed(args.seed)
    from toupee import data
    from toupee import config 
    from toupee.mlp import sequential_model

    params = config.load_parameters(args.params_file)

    def arg_params(arg_value,param):
        if arg_value is not None:
            params.__dict__[param] = arg_value

    for arg, param in arg_param_pairings:
        arg_params(arg,param)
    dataset = data.load_data(params.dataset,
                             pickled = params.pickled,
                             one_hot_y = params.one_hot,
                             join_train_and_valid = params.join_train_and_valid,
                             zca_whitening = params.zca_whitening)
    method = params.method
    method.prepare(params,dataset)
    train_set = method.resampler.get_train()
    valid_set = method.resampler.get_valid()
    
    #selects the appropriate intermediate score: classification - accuracy; regression - euclidian_distance
    scorer = []
    scorer_name = []
    if params.classification == True:   
        scorer.append(accuracy)
        scorer_name.append('accuracy')
    else:
        scorer.append(euclidian_distance)
        scorer_name.append('euclidian distance')
        
        scorer.append(relative_distance)
        scorer_name.append('relative distance')
    
    members = []
    intermediate_scores = []
    final_score = None
    for i in range(0,params.ensemble_size):
        print(('\n\ntraining member {0}'.format(i)))
        m = method.create_member()
        members.append(m[:2])
        ensemble = method.create_aggregator(params,members,train_set,valid_set)
        test_set_x, test_set_y = method.resampler.get_test()
        
        test_score = []
        for j in range(len(scorer)):
            test_score.append(scorer[j](ensemble,test_set_x,test_set_y))
            print(('Intermediate test {0}: {1}'.format(scorer_name[j], test_score[j])))
        
        intermediate_scores.append(test_score)
        final_score = test_score
        if len(m) > 2 and not m[2]: #the ensemble method told us to stop
            break
    
    for j in range(len(scorer)): print(('Final test {0}: {1}'.format(scorer_name[j], test_score[j])))
    
    if args.dump_shapes_to is not None:
        dill.dump({'members': members, 'ensemble': ensemble},
                open(args.dump_to,"wb"))
    if args.dump_shapes_to is not None:
        for i in range(len(members)):
            with open("{0}member-{1}.model".format(args.dump_shapes_to, i),"w") as f:
                f.truncate()
                f.write(members[i][0])
                
    if 'results_db' in params.__dict__:
        if 'results_host' in params.__dict__:
            host = params.results_host
        else:
            host = None
        print(("saving results to {0}@{1}".format(params.results_db,host)))
        conn = MongoClient(host=host)
        db = conn[params.results_db]
        if 'results_table' in params.__dict__: 
            table_name = params.results_table
        else:
            table_name = 'results'
        table = db[table_name]
        
        #removes mongodb-buggy "params" entries
        params.method = ''
        
        results = {
                    "params_file": args.params_file,
                    "params": params.__dict__,
                    "intermediate_test_scores" : intermediate_scores,
                    "final_test_score" : final_score,
                    "best_score": np.max(intermediate_scores),
                    "best_score_after_ensemble_#": np.argmax(intermediate_scores).item(),   #without "item()", defaults to np.int64, which is not supported by mongodb
                    "date": datetime.datetime.utcnow(),
                    "code version": subprocess.check_output(["git", "describe","--always"]).strip(),
                  }                
        
        #adds the dependency ID
        if 'results_dep' in params.__dict__:
            depend_col = db[params.results_dep]
            if(depend_col.count() > 0): #if the collection exists
                latest_entry = depend_col.find().sort("_id", -1).limit(1)
                latest_id = latest_entry[0]['_id']
            else:
                latest_id = 'no previous entry!'
            column_name = params.results_dep + '_id'
            results[column_name] = latest_id
        
        
        id = table.insert_one(results).inserted_id
        
        #prints stuff if needed [for now its on, to help with the dbg]
        if True:
            import pprint
            print("PRINTING DB ENTRY:")
            print(("DB NAME = ", params.results_db))
            print(("TABLE NAME = ", table_name))
            print("TABLE ENRTY:")
            latest_entry = table.find().sort("_id", -1).limit(1)
            pprint.pprint(latest_entry[0])
        
        print(("\n\nDone. [Results stored in the DB, ID = {0}]".format(id)))

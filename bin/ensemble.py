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


def check_data_format(args):
    '''
    Checks if the input file is a .npz or a h5
    '''
    
    if (args.trainfile[-4:] == '.npz') and (args.validfile[-4:] == '.npz') and (args.testfile[-4:] == '.npz'):
        is_h5 = False
        print("\nLoading .npz data - the dataset stays on the RAM\n")
    elif (args.trainfile[-3:] == '.h5') and (args.validfile[-3:] == '.h5') and (args.testfile[-3:] == '.h5'):
        is_h5 = True
        print("\nLoading .h5 data - the dataset stays on the hard drive\n")
    else:
        raise ValueError('.npz or .h5 files are required; All sets must have the same format.')
        
    return(is_h5)


def run_ensembles_npz(args, params):

    method = params.method
    dataset = data.load_data(params.dataset,
                             pickled = params.pickled,
                             one_hot_y = params.one_hot,
                             join_train_and_valid = params.join_train_and_valid,
                             zca_whitening = params.zca_whitening,
                             testfile = args.testfile, 
                             validfile = args.validfile, 
                             trainfile = args.trainfile)

    method.prepare(params, dataset)
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
    
    if args.dump_to is not None:
        dill.dump({'members': members, 'ensemble': ensemble},
                # open(args.dump_to,"wb"))
                open(os.path.join(params.dataset, args.dump_to),"wb"))
    if args.dump_shapes_to is not None:
        if args.dump_shapes_to == '':
            dump_shapes_to = args.seed
        else:
            dump_shapes_to = args.dump_shapes_to
        for i in range(len(members)):
            with open("{0}member-{1}.model".format(dump_shapes_to, i),"w") as f:
                f.truncate()
                f.write(members[i][0])
                
    return (intermediate_scores, final_score)

    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a single MLP')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('save_file', nargs='?',
                        help='the file where the trained MLP is to be saved')
    parser.add_argument('--seed', type=int, nargs='?', default=42,
                        help='random seed to use for this sim')
    parser.add_argument('--epochs', type=int, nargs='?',
                        help='number of epochs to run')
    parser.add_argument('--ensemble-size', type=int, nargs='?',
                        help="ensemble size")
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
    parser.add_argument('--dump-shapes-to', type=str, nargs='?', default=None,
                        help='location where to save the shape of the ensemble members. Pass \'\' to use the same number as --seed')
    parser.add_argument('--dump-to', type=str, nargs='?', default='ensemble.pkl',
                        help='location where to save the ensemble')
    parser.add_argument('--testfile', default='test.npz',
                        help='test set npz file name')
    parser.add_argument('--validfile', default='valid.npz',
                        help='valid set npz file name')
    parser.add_argument('--trainfile', default='train.npz',
                        help='training set npz file name')
    parser.add_argument('--dict-number', help="dict_number to use (= dataset location)",
                        default=None)
    parser.add_argument('--latest-experiment', help="uses the latest experiment",
                        action='store_true')
    parser.add_argument('--remove-tmp-files', help="remove the temporary model files at the end.",
                        action='store_true')

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
        (args.ensemble_size, 'ensemble_size'),
        (args.testfile, 'testfile'),
        (args.validfile, 'validfile'),
        (args.trainfile, 'trainfile'),
    ]
    
    if 'seed' in args.__dict__:
        print(("setting random seed to: {0}".format(args.seed)))
        numpy.random.seed(args.seed)
    from toupee import data
    from toupee import config 
    from toupee.mlp import sequential_model

    
    
    #Dataset location: hardcoded (@.yaml) < latest-experiment flag < specific dict number
    
    params = config.load_parameters(args.params_file)

    if args.latest_experiment or args.dict_number:
        conn = MongoClient(host=args.results_host)
        db = conn[args.results_db]
        table = db[args.results_dep]
        
        if args.latest_experiment:
            #gets the most recent ID
            latest_entry = table.find().sort("_id", -1).limit(1)
            if latest_entry.count() == 0:
                raise ValueError('No DB entries for trained NNs')
            else:
                latest_entry = latest_entry[0]
                
            latest_entry_location = latest_entry['file_location']
            params.dataset = latest_entry_location
            
        if args.dict_number is not None:
            #TODO: for now, it assumes it is the local user folder
            target_root = os.path.expanduser("~/data_tmp")
            dict_dir = os.path.join(target_root, 'dict_' + str(args.dict_number))
            
            if os.path.exists(dict_dir):
                params.dataset = dict_dir
            else:
                print("The desired dict_number doesn't exist!")
            


    def arg_params(arg_value,param):
        if arg_value is not None:
            params.__dict__[param] = arg_value

    for arg, param in arg_param_pairings:
        arg_params(arg,param)
        
    
    #Checks for h5/npz data
    #TODO: if any data transform option is true, the h5 version will be incorrect
    is_h5 = check_data_format(args)
    
    
    if is_h5 is False: 
        intermediate_scores, final_score = run_ensembles_npz(args, params)
    #else:
        
                
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
                    "dict_number": args.dict_number,
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

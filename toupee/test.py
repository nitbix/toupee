#!/usr/bin/python

from pprint import pprint
import yaml

import parameters
import update_rules

print yaml.dump(update_rules.RProp())

p = parameters.load_parameters('mnist-vanilla.yaml')

print p.dataset

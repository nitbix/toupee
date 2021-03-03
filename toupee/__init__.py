import toupee.data as data
import toupee.model as model
import toupee.ensembles as ensembles
import toupee.parameters as parameters
import toupee.config as config
import toupee.utils as utils
import toupee.adversarial as adversarial

from toupee.utils import log_metrics

#TODO: use AI Platform
#TODO: use local TPUs

version = "2.0.0 pre-release nightly"

PRINTABLE_METRICS = ['accuracy_score',
                     'micro_precision_score',
                     'micro_recall_score',
                     'micro_f1_score',
                     'macro_precision_score',
                     'macro_recall_score',
                     'macro_f1_score',
                     'calibration'
                     ]

ADVERSARIAL_EPSILONS = [0.1, 0.01, 0.001]

config.setup_yaml()
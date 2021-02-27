from .aggregators import Averaging, MajorityVoting
from .methods import Simple, Bagging, AdaBoost, Incremental, DIB, DIBag

METHOD_MAPPER = {
    'simple': Simple,
    'bagging': Bagging,
    'adaboost': AdaBoost,
    'incremental': Incremental,
    'dib': DIB,
    'dibag': DIBag,
}


AGGREGATOR_MAPPER = {
    'averaging': Averaging,
    'majorityvoting': MajorityVoting
}


def create(params, data, wandb=None, adversarial_testing:bool=False, distil:bool=False, tensorboard:bool=False):
    """ Create an Ensemble from spec """
    spec = params.ensemble_method
    return METHOD_MAPPER[spec['class_name'].lower()](
        data=data,
        model_params=params,
        wandb=wandb,
        adversarial_testing=adversarial_testing,
        distil=distil,
        tensorboard=tensorboard,
        **spec['params']
    )


def get_aggregator(aggregator_type:str):
    """ Get a reference to a new aggregator """
    return AGGREGATOR_MAPPER[aggregator_type.lower()]()
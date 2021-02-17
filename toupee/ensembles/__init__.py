from .aggregators import Averaging
from .methods import Simple, Bagging, AdaBoost, DIB

METHOD_MAPPER = {
    'simple': Simple,
    'bagging': Bagging,
    'adaboost': AdaBoost,
    'dib': DIB,
}


AGGREGATOR_MAPPER = {
    'averaging': Averaging
}

def create(params, data, wandb=None, adversarial_testing:bool=False):
    """ Create an Ensemble from spec """
    spec = params.ensemble_method
    return METHOD_MAPPER[spec['class_name'].lower()](
        data=data,
        model_params=params,
        wandb=wandb,
        adversarial_testing=adversarial_testing,
        **spec['params']
    )


def get_aggregator(aggregator_type):
    """ Get a reference to a new aggregator """
    return AGGREGATOR_MAPPER[aggregator_type.lower()]()
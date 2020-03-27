from .aggregators import Averaging
from .methods import Simple

METHOD_MAPPER = {
    'simple': Simple
}


AGGREGATOR_MAPPER = {
    'averaging': Averaging
}

def create(params, data):
    """ Create an Ensemble from spec """
    spec = params.ensemble_method
    return METHOD_MAPPER[spec['class_name'].lower()](
        data=data,
        params=params,
        **spec['params']
    )


def get_aggregator(aggregator_type):
    """ Get a reference to a new aggregator """
    return AGGREGATOR_MAPPER[aggregator_type.lower()]()
import pytest
import toupee as tp


MNIST_PARAMS_FILE = 'tests/test_mnist.yaml'


METRICS_TO_CHECK = {'accuracy_score': 0.98,
                    'micro_precision_score': 0.98,
                    'micro_recall_score': 0.98,
                    'micro_f1_score': 0.98,
                    'macro_precision_score': 0.98,
                    'macro_recall_score': 0.98,
                    'macro_f1_score': 0.98}


def test_mnist_single() -> None:
    """ Loads parameters to train a single model on MNIST """
    params = tp.config.load_parameters(MNIST_PARAMS_FILE)
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    base_model = tp.model.Model(params=params)
    base_model.fit(data=data)
    for metric, limit in METRICS_TO_CHECK.items():
        assert base_model.test_metrics[metric] > limit


def test_mnist_bagging() -> None:
    """ Loads parameters to train a single model on MNIST """
    params = tp.config.load_parameters(MNIST_PARAMS_FILE)
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    method = tp.ensembles.create(params, data)
    metrics = method.fit()
    print(metrics)
    for metric, limit in METRICS_TO_CHECK.items():
        assert base_model.test_metrics[metric] > limit

if __name__ == "__main__":
    test_mnist_bagging()
    test_mnist_single()
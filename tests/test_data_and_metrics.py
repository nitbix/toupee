import numpy as np
import toupee as tp

METRICS_THAT_SHOULD_BE_1 = ['accuracy_score',
                            'micro_f1_score',
                            'macro_f1_score',
                            'micro_precision_score',
                            'macro_precision_score',
                            'micro_recall_score',
                            'macro_recall_score'
                        ]


def check_scores(scores):
    for metric in METRICS_THAT_SHOULD_BE_1:
        assert scores[metric] == 1.


def test_cifar10():
    params = tp.config.load_parameters("tests/test_params.yaml")
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    for data_slice in data.raw_data.values():
        y_true = np.argmax(data_slice[1], axis=1)
        check_scores(tp.utils.eval_scores(y_true, y_true))
    y_true = np.argmax(data.raw_data['test'][1], axis=1)
    all_y_true = []
    test_handle = data.get_testing_handle()
    for (x, y_true_batch) in test_handle:
        all_y_true.append(np.argmax(y_true_batch.numpy(), axis=1))
    y_true_gen = np.concatenate(all_y_true)
    check_scores(tp.utils.eval_scores(y_true, y_true_gen))
    model = tp.model.Model(params)
    model.fit(data=data)
    y_pred = model.predict_classes(data.raw_data['test'][0])
    assert max(y_pred) == max(y_true)
    assert min(y_pred) == min(y_true)
    metrics = tp.utils.eval_scores(y_true, y_pred)
    for metric in METRICS_THAT_SHOULD_BE_1:
        assert metrics[metric] == model.test_metrics[metric]

if __name__ == "__main__":
    test_cifar10()
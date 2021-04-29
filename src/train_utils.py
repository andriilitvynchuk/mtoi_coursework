import copy
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .membership import build_membership_function
from .nefclass import NefClassModel
from .preprocess_utils import preprocess_with_knn_imputer_minmax_scaler


seed = 8


def train_nefclass(
    model_params: Dict[str, Any],
    train_data: np.ndarray,
    train_targets: np.ndarray,
    test_data: np.ndarray,
    test_targets: np.ndarray,
    universe_max: np.ndarray,
    universe_min: np.ndarray,
    metric_to_maximize: Callable = f1_score,
) -> Tuple[NefClassModel, float, float]:
    model = NefClassModel(
        num_input_units=model_params["num_input_units"],
        num_fuzzy_sets=model_params["num_sets"],
        kmax=model_params["kmax"],
        output_units=model_params["output_units"],
        universe_max=universe_max,
        universe_min=universe_min,
        membership_type=model_params["mf"],
    )

    # linguistic variables
    labels = [f"label_{index}" for index in range(model_params["num_sets"])]
    abcs = [
        build_membership_function(train_data[:, feature_index], labels) for feature_index in range(train_data.shape[1])
    ]
    model.init_fuzzy_sets(abcs)

    # train rules
    for features, target in zip(train_data, train_targets):
        model.learn_rule(features, target)

    best_metric_dict: Dict[str, Any] = dict(value=0, epoch=0, model=None)
    metrics: Dict[str, List[float]] = dict(train=[], test=[])
    # train fuzzy sets
    for epoch in tqdm(range(model_params["num_epoch"])):
        for features, target in zip(train_data, train_targets):
            output = model(features, target)
            delta = [
                1 - output[class_index] if class_index == target else 0 - output[class_index]
                for class_index in range(len(output))
            ]
            model.update_fuzzy_sets(model_params["sigma"], delta)
        metrics["train"].append(
            metric_to_maximize(y_true=train_targets, y_pred=model.predict(train_data, train_targets))
        )
        metrics["test"].append(metric_to_maximize(y_true=test_targets, y_pred=model.predict(test_data, test_targets)))
        this_epoch_test_metric = metrics["test"][-1]
        if this_epoch_test_metric > best_metric_dict["value"]:
            best_metric_dict["value"] = this_epoch_test_metric
            best_metric_dict["epoch"] = epoch
            best_metric_dict["model"] = copy.deepcopy(model)
        # early stopping
        if epoch - best_metric_dict["epoch"] > min(model_params["num_epoch"] / 10, 10):
            break
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: {this_epoch_test_metric:.4f}")

    print("Best metric: ", best_metric_dict["value"], " at epoch # ", best_metric_dict["epoch"])
    return (
        best_metric_dict["model"],
        metrics["train"][best_metric_dict["epoch"]],
        metrics["test"][best_metric_dict["epoch"]],
    )


def cross_validation_train_neflcass(
    data: np.ndarray,
    target: np.ndarray,
    model_params: Dict[str, Any],
    preprocess_func: Callable = preprocess_with_knn_imputer_minmax_scaler,
    metric_to_maximize: Callable = f1_score,
    metrics_to_save: Tuple[Tuple[str, Callable], ...] = (("f1", f1_score),),  # type: ignore
    folds: int = 5,
) -> Dict[str, float]:
    stratified_kfold = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    metrics: Dict[str, Any] = defaultdict(list)
    for train_index, test_index in stratified_kfold.split(data, target):
        train_data, train_targets = data[train_index], target[train_index]
        test_data, test_targets = data[test_index], target[test_index]

        train_data, test_data = preprocess_func(train_data=train_data, test_data=test_data)

        model_params["num_input_units"] = train_data.shape[1]
        model_params["output_units"] = len(np.unique(train_targets))

        model, best_train, best_test = train_nefclass(
            model_params=model_params,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            universe_max=np.max(train_data, axis=0),
            universe_min=np.min(train_data, axis=0),
            metric_to_maximize=metric_to_maximize,
        )

        for (metric_name, metric_func) in metrics_to_save:
            # metric_name, metric_func = metric_tuple
            metrics[metric_name + "_train"].append(
                metric_func(y_true=train_targets, y_pred=model.predict(train_data, train_targets))
            )
            metrics[metric_name + "_test"].append(
                metric_func(y_true=test_targets, y_pred=model.predict(test_data, test_targets))
            )
    metrics = {key: np.mean(value) for key, value in metrics.items()}
    return metrics

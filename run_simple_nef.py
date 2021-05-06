from typing import NoReturn

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess_utils import preprocess_with_knn_imputer_minmax_scaler
from src.train_utils import cross_validation_train_neflcass


def main() -> NoReturn:
    data = pd.read_csv("data/preprocessed_data.csv", header=0, index_col=0)

    model_params = dict(sigma=0.001, num_epoch=50, num_sets=7, kmax=5, mf="gaussian")
    metrics = cross_validation_train_neflcass(
        data=data.iloc[:, 1:].values,
        target=data.iloc[:, 0].values.astype(int),
        model_params=model_params,
        preprocess_func=preprocess_with_knn_imputer_minmax_scaler,
        metric_to_maximize=f1_score,
        metrics_to_save=(("f1", f1_score), ("accuracy", accuracy_score)),
        folds=5,
    )
    print(metrics)


if __name__ == "__main__":
    main()

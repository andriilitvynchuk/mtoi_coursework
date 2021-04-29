from typing import NoReturn

import pandas as pd
from sklearn.metrics import f1_score

from src.preprocess_utils import preprocess_with_knn_imputer_minmax_scaler
from src.train_utils import cross_validation_train_neflcass


def main() -> NoReturn:
    data = pd.read_csv("data/preprocessed_data.csv", header=0, index_col=0)

    model_params = dict(sigma=0.01, num_epoch=5, num_sets=3, kmax=50, mf="tri")
    metrics = cross_validation_train_neflcass(
        data=data.iloc[:, 1:].values,
        target=data.iloc[:, 0].values.astype(int),
        model_params=model_params,
        preprocess_func=preprocess_with_knn_imputer_minmax_scaler,
        metric_to_maximize=f1_score,
        folds=5,
    )
    print(metrics)


if __name__ == "__main__":
    main()

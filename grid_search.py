import multiprocessing as mp
from collections import defaultdict
from functools import partial
from itertools import product
from typing import NoReturn

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.preprocess_utils import preprocess_with_knn_imputer_minmax_scaler
from src.train_utils import cross_validation_train_neflcass


def main() -> NoReturn:
    data = pd.read_csv("data/preprocessed_data.csv", header=0, index_col=0)

    grid_sigma = [0.001, 0.01, 0.1]
    grid_num_epoch = [15, 25, 50]
    grid_num_sets = [3, 5, 7]
    grid_kmax = [5, 10, 25, 50]
    grid_membership = ["tri", "gaussian", "semicircle"]

    all_model_params = []
    for sigma, num_epoch, num_sets, kmax, membership in product(
        grid_sigma, grid_num_epoch, grid_num_sets, grid_kmax, grid_membership
    ):
        all_model_params.append(dict(sigma=sigma, num_epoch=num_epoch, num_sets=num_sets, kmax=kmax, mf=membership))

    with mp.Pool(processes=min(len(all_model_params), 24)) as pool:
        function = partial(
            cross_validation_train_neflcass,
            data=data.iloc[:, 1:].values,
            target=data.iloc[:, 0].values.astype(int),
            preprocess_func=preprocess_with_knn_imputer_minmax_scaler,
            metric_to_maximize=f1_score,
            metrics_to_save=(("f1", f1_score), ("accuracy", accuracy_score)),
            folds=5,
            return_full_dict=True,
            tqdm_learning=False,
        )
        results = list(tqdm(pool.imap(function, all_model_params), total=len(all_model_params)))

    results_for_dataframe = defaultdict(list)
    for result in results:
        for key, value in result.items():
            results_for_dataframe[key].append(value)
    results_df = pd.DataFrame(results_for_dataframe)
    results_df.to_csv("results/baseline.csv", header=True, index=True)


if __name__ == "__main__":
    main()

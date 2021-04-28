import argparse
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.train_utils import train_nefclass


def main() -> NoReturn:
    data = pd.read_csv("data/preprocessed_data.csv", header=0, index_col=0)

    train_data, train_target = data.iloc[:4000, 1:].to_numpy(), data.iloc[:4000, 0].to_numpy().astype(int)
    test_data, test_target = data.iloc[4000:, 1:].to_numpy(), data.iloc[4000:, 0].to_numpy().astype(int)

    model_params = dict(sigma=0.01, num_epoch=100, num_sets=3, kmax=100, mf="tri")
    model_params["num_input_units"] = train_data.shape[1]
    model_params["output_units"] = data["target"].nunique()

    train_nefclass(
        model_params=model_params,
        train_data=train_data,
        train_targets=train_target,
        test_data=test_data,
        test_targets=test_target,
        universe_max=np.max(train_data, axis=0),
        universe_min=np.min(train_data, axis=0),
        metric_to_maximize=f1_score,
    )


if __name__ == "__main__":
    main()

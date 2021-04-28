import argparse
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.train_utils import train_nefclass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEFCLASS")
    parser.add_argument("--sigma", default=0.01, type=float, help="learning rate")
    parser.add_argument("--num_epoch", default=100, type=int, help="number of epoch for fuzzy set learning")
    parser.add_argument("--num_sets", default=3, type=int, help="number of fuzzy sets")
    parser.add_argument("--kmax", default=50, type=int, help="maximum number of rules")
    parser.add_argument(
        "--mf",
        default="tri",
        type=str,
        help="membership function to use. Default: tri. Options: gaussian, semicircle",
    )

    args = parser.parse_args()

    data = pd.read_csv("data/preprocessed_data.csv", header=0, index_col=0)

    train_data, train_target = data.iloc[:4000, 1:].to_numpy(), data.iloc[:4000, 0].to_numpy().astype(int)
    test_data, test_target = data.iloc[4000:, 1:].to_numpy(), data.iloc[4000:, 0].to_numpy().astype(int)

    model_params = dict(sigma=0.01, num_epoch=100, num_sets=3, kmax=50, mf="tri")
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


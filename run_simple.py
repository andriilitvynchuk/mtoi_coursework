import argparse
from typing import NoReturn

import numpy as np
import pandas as pd

from src.train_utils import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEFCLASS")
    parser.add_argument("--sigma", default=0.01, type=float, help="learning rate")
    parser.add_argument("--num_epoch", default=5, type=int, help="number of epoch for fuzzy set learning")
    parser.add_argument("--num_sets", default=5, type=int, help="number of fuzzy sets")
    parser.add_argument("--kmax", default=50, type=int, help="maximum number of rules")
    parser.add_argument("--cv", default=False, action="store_true", help="do 10 fold cross validation?")
    parser.add_argument("--kfold", default=10, type=int, help="number of k fold")
    parser.add_argument("-v", default=False, action="store_true", help="verbosity")
    parser.add_argument(
        "--mf",
        default="tri",
        type=str,
        help="membership function to use. Default: tri. Options: gaussian, semicircle",
    )

    args = parser.parse_args()

    data = pd.read_csv("data/preprocessed_data.csv", header=0, index_col=0)

    train_data, train_target = data.iloc[:4000, 1:].to_numpy(), data.iloc[:4000, 1].to_numpy().astype(int)
    test_data, test_target = data.iloc[4000:, 1:].to_numpy(), data.iloc[4000:, 1].to_numpy().astype(int)
    args.num_input_units = train_data.shape[1]
    args.output_units = data["target"].nunique()
    labels = [f"label_{index}" for index in range(args.num_sets)]

    train(
        args=args,
        labels=labels,
        train_data=train_data,
        train_targets=train_target,
        test_data=test_data,
        test_targets=test_target,
        universe_max=np.max(train_data, axis=0),
        universe_min=np.min(train_data, axis=0),
        verbose=True,
    )


from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


np.random.seed(8)


def main() -> NoReturn:
    raw_data: pd.DataFrame = pd.read_csv("data/raw_data.csv", header=0, index_col=0)
    raw_data = raw_data.rename(columns={"SeriousDlqin2yrs": "target"})
    raw_data["target"] = raw_data["target"].astype(int)

    subset = 5000
    index_subset = np.concatenate(
        [
            np.random.choice(raw_data[raw_data["target"] == 1].index, int(subset / 2)),
            np.random.choice(raw_data[raw_data["target"] == 0].index, int(subset / 2)),
        ]
    )
    np.random.shuffle(index_subset)

    original_subset = raw_data.loc[index_subset]
    original_subset.index = range(len(original_subset.index))
    original_subset.to_csv("data/clear_data.csv", header=True, index=True)

    # input nans with KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    values_without_nans = imputer.fit_transform(original_subset)
    # scale all values to [0, 1] range
    min_max_scaler = MinMaxScaler()
    scaled_values_without_nans = min_max_scaler.fit_transform(values_without_nans)
    preprocessed_subset = pd.DataFrame(
        scaled_values_without_nans, index=original_subset.index, columns=original_subset.columns
    )
    preprocessed_subset.to_csv("data/preprocessed_data.csv", header=True, index=True)


if __name__ == "__main__":
    main()

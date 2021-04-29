from typing import Tuple

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def preprocess_with_knn_imputer_minmax_scaler(
    train_data: np.ndarray, test_data: np.ndarray, n_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    train_data_without_nans = imputer.fit_transform(train_data)
    test_data_without_nans = imputer.transform(test_data)

    min_max_scaler = MinMaxScaler()
    train_data_without_nans_scaled = min_max_scaler.fit_transform(train_data_without_nans)
    test_data_without_nans_scaled = min_max_scaler.transform(test_data_without_nans)

    return train_data_without_nans_scaled, test_data_without_nans_scaled

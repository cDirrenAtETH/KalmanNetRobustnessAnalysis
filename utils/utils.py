from config.configuration import DataSetConfig

import regex

import numpy as np
import pandas as pd

from typing import Any, Optional

import json
import logging.config
import os
import os.path

NAN_LOSS_VALUE = 1000

DEFAULT_ESTIMATOR_NAMES = ['KalmanNet (full, known)', 'KalmanNet (velocity, known)', 'KalmanNet (full, unknown)',
                           'KalmanNet (velocity, unknown)', 'EKF', 'Noise Floor']
BASE_ADD_FILTER = {
    DEFAULT_ESTIMATOR_NAMES[0]: {'training_config': "loss='MSE'", 'train_random_variance': 'None'},
    DEFAULT_ESTIMATOR_NAMES[1]: {'training_config': "loss='velMSE'", 'train_random_variance': 'None'},
    DEFAULT_ESTIMATOR_NAMES[2]: {'training_config': "loss='MSE'", 'train_random_variance': '[4, 4, 1, 1]'},
    DEFAULT_ESTIMATOR_NAMES[3]: {'training_config': "loss='velMSE'", 'train_random_variance': '[4, 4, 1, 1]'}}


def setup_logging(
        default_path='logging/config.json',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration.py

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            log_config = json.load(f)
        logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=default_level)


def get_min_loss(result_path: str, result_name: str, group_by: str, filter_dict: dict[str, Any],
                 **kwargs) -> pd.DataFrame:
    results = pd.read_csv(result_path + '/' + result_name + '.csv')
    results_header = list(results.columns.values)

    filter_dict.update(kwargs)

    # Get additional filter values from kwargs
    for column_name, value in filter_dict.items():
        if column_name in results_header:
            filter_dict[column_name] = value

    # Filter the dataframe with the given values
    for column_name, value in filter_dict.items():
        if type(value) is float:
            results = results[np.isclose(results[column_name], value)]
        elif type(value) is str:
            results = results[results[column_name].str.contains(value, regex=False)]
        else:
            results = results.loc[results[column_name] == value]

    # Fill all nan values with fixed values in order to get all results
    results.fillna(value={'loss_in_dB': NAN_LOSS_VALUE}, inplace=True)

    # Get the min loss of the filtered dataframe
    results = results.loc[results.groupby(group_by).loss_in_dB.idxmin()].sort_values(group_by, ascending=False)

    # Replace NAN_LOSS_VALUES with pd.NA
    results.replace({'loss_in_dB': NAN_LOSS_VALUE}, np.NAN, inplace=True)

    return results


def get_best_model_name(result_path: str, result_name: str, group_by: str, filter_dict: dict[str, Any]) -> tuple[
    str, Optional[str]]:
    results = get_min_loss(result_path, result_name, group_by, filter_dict)
    if results.size == 0:
        raise ValueError("There is no result with filters: " + str(filter_dict))
    else:
        training_config_name = regex.search("config_name='+\K[^']*", results['training_config'].iloc[0])
        if training_config_name is not None:
            training_config_name = training_config_name.group()
        return results['model_name'].iloc[0], training_config_name


def get_add_filters_per_estimator(estimator_names: list[str], train_set_config: DataSetConfig) -> dict[str, dict]:
    data_config = {'data_config': 'train_set=' +str(train_set_config)}
    add_filters = {k: v
                   for (k, v) in BASE_ADD_FILTER.items()
                   if k in estimator_names}
    for (k, v) in add_filters.items():
        v.update(data_config)
    return add_filters

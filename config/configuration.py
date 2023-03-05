from dataclasses import dataclass
from itertools import product
from typing import Optional

import jsons

DATA_CONFIG_JSON_PATH = 'config/data_configs.json'
ESTIMATOR_CONFIG_JSON_PATH = 'config/estimator_configs.json'
TRAINING_CONFIG_JSON_PATH = 'config/training_configs.json'
RESULT_CONFIG_JSON_PATH = 'config/result_configs.json'

NN_ESTIMATOR_TYPES = ['KalmanNetNN']
ALL_ESTIMATOR_TYPES = ['ExtendedKalmanFilter', 'KalmanNetNN']


@dataclass
class DataSetConfig:
    batch_size: int
    trajectory_length: int


@dataclass
class PathConfig:
    name: Optional[str]
    path: str


@dataclass
class ResultConfig:
    config_name: str
    save: bool
    save_info: PathConfig


@dataclass
class DataConfig:
    config_name: str
    generate: bool
    from_config: Optional[str]
    save_info: PathConfig
    process_var: float
    measurement_var: float
    sampling_time: float
    max_taylor_coefficient: int
    train_set: DataSetConfig
    val_set: DataSetConfig
    test_set: DataSetConfig


@dataclass
class EkfConfig:
    config_name: str
    loss: Optional[str]
    type: Optional[str]
    test: bool
    random_variance: Optional[list]
    save_info: PathConfig
    save: Optional[bool]
    save_test_result: Optional[bool]
    single_trajectory: Optional[bool]


@dataclass
class TrainingConfig:
    config_name: str
    n_epochs: int
    loss: str
    train_size: int
    val_size: int
    test_size: int
    learning_rate: float
    weight_decay: float
    random_variance: Optional[list]


@dataclass
class EstimatorConfig:
    config_name: str
    loss: str
    type: str
    train: bool
    train_config: Optional[TrainingConfig]
    test: bool
    random_variance: Optional[list]
    save: bool
    save_pipeline: bool
    save_info: PathConfig


@dataclass
class KnetConfig:
    config_name: str
    train: bool
    test: bool
    loss: str
    random_variance: Optional[list]
    save_info: PathConfig
    type: str  # KalmanNetNN, KalmanNetSophisticated
    save_pipeline: bool
    save_test_result: bool


class MainConfiguration:
    data_config: str
    estimator_config: str
    training_config: Optional[str]
    result_config: str


class Configuration:
    configuration_name: str
    additional_info: str
    data_config: DataConfig
    estimator_config: EstimatorConfig
    result_config: ResultConfig


def get_config_from_subconfig(
        data_name: str,
        estimator_name: str,
        result_name: str,
        training_name: Optional[str] = None) -> Configuration:
    main_config = MainConfiguration()
    main_config.data_config = data_name
    main_config.estimator_config = estimator_name
    main_config.training_config = training_name
    main_config.result_config = result_name

    return get_config_from_main(main_config)


def get_all_possible_configs_from_subconfigs(
        data_names: list[str],
        estimator_names: list[str],
        result_names: list[str],
        training_names: list[str]) -> list[Configuration]:

    all_combinations = product(data_names, estimator_names, training_names, result_names)
    main_configs: list[MainConfiguration] = []

    for config_set in all_combinations:
        config = MainConfiguration()
        config.data_config = config_set[0]
        config.estimator_config = config_set[1]
        config.training_config = config_set[2]
        config.result_config = config_set[3]

        main_configs.append(config)

    return get_configs_from_main(main_configs)


def get_specific_configs_from_subconfigs(
        data_names: list[str],
        estimator_names: list[str],
        result_names: list[str],
        training_names: list[Optional[str]] = [None]) -> list[Configuration]:
    if len(estimator_names)!= len(training_names):
        raise ValueError("Each estimator must have a training name even if its none")
    complete_estimator_names = zip(estimator_names, training_names)
    all_combinations = product(data_names, complete_estimator_names, result_names)
    main_configs: list[MainConfiguration] = []

    for config_set in all_combinations:
        config = MainConfiguration()
        config.data_config = config_set[0]
        config.estimator_config = config_set[1][0]
        config.training_config = config_set[1][1]
        config.result_config = config_set[2]

        main_configs.append(config)

    return get_configs_from_main(main_configs)


# def get_configs_from_path(path: str, config_names: list[str]) -> list[Configuration]:
#     json_data = get_json_from_path(path)
#     main_configs = get_main_configs_from_json(json_data, config_names)
#     return get_configs_from_main(main_configs)


def get_json_from_path(path: str) -> str:
    with open(path, 'r') as json_file:
        json_data = json_file.read()
    return json_data


# def get_main_configs_from_json(json: str, config_names: list[str]) -> list[MainConfiguration]:
#     configs = jsons.loads(json, List[MainConfiguration])
#     filtered_configs = []
#
#     for config in configs:
#         if config.main_config in config_names:
#             filtered_configs.append(config)
#     return filtered_configs

def get_config_from_main(main_config: MainConfiguration) -> Configuration:
    config = Configuration()
    json_data = get_json_from_path(DATA_CONFIG_JSON_PATH)
    json_estimator = get_json_from_path(ESTIMATOR_CONFIG_JSON_PATH)
    json_result = get_json_from_path(RESULT_CONFIG_JSON_PATH)

    config.data_config = get_data_config_from_json(json_data, main_config.data_config)
    config.estimator_config = get_estimator_config_from_json(json_estimator,
                                                             estimator_config_name=main_config.estimator_config,
                                                             train_config_name=main_config.training_config,
                                                             data_config=config.data_config)
    config.result_config = get_result_config_from_json(json_result, main_config.result_config)
    config.configuration_name = f'{config.data_config=}_{config.estimator_config=}_{config.result_config=}'
    return config


def get_configs_from_main(main_configs: list[MainConfiguration]) -> list[Configuration]:
    configs = []
    for main_config in main_configs:
        config = get_config_from_main(main_config)
        configs.append(config)

    return configs


def get_data_config_from_json(data_json: str, config_name: str) -> DataConfig:
    configs = jsons.loads(data_json, list[DataConfig])
    configs = [config for config in configs if config.config_name == config_name]
    if len(configs) == 0:
        raise ValueError("The data config name " + config_name + " does not exist")
    config = configs[0]
    if config.save_info.name is None:
        config.save_info.name = get_data_name(config)
    return config


def get_estimator_config_from_json(data_json: str, estimator_config_name: str,
                                   train_config_name: Optional[str],
                                   data_config: DataConfig) -> EstimatorConfig:
    configs = jsons.loads(data_json, list[EstimatorConfig])
    configs = [config for config in configs if config.config_name == estimator_config_name]
    if len(configs) == 0:
        raise ValueError("The estimator config name " + estimator_config_name + " does not exist")
    config = configs[0]

    if config.type == "KalmanNetNN" and train_config_name is not None:
        json_train = get_json_from_path(TRAINING_CONFIG_JSON_PATH)
        config.train_config = get_training_config_from_json(json_train, train_config_name)
    elif config.type == "KalmanNetNN" and train_config_name is None:
        raise ValueError("The config name " + estimator_config_name + " need a valid training configuration")

    if config.save_info.name is None:
        config.save_info.name = get_default_estimator_name(config, data_config)

    return config


# def get_ekf_config_from_json(data_json: str, config_name: str, data_name: str) -> EkfConfig:
#     configs = jsons.loads(data_json, list[EkfConfig])
#     configs = [config for config in configs if config.config_name == config_name]
#     if len(configs) == 0:
#         raise ValueError("The ekf config name " + config_name + " does not exist")
#     config = configs[0]
#     if config.save_info.name is None:
#         config.save_info.name = get_ekf_name(config, data_name)
#     return config
#
#
# def get_knet_config_from_json(data_json: str, config_name: str) -> KnetConfig:
#     configs = jsons.loads(data_json, list[KnetConfig])
#     configs = [config for config in configs if config.config_name == config_name]
#     if len(configs) == 0:
#         raise ValueError("The knet_old config name " + config_name + " does not exist")
#     return configs[0]


def get_training_config_from_json(data_json: str, config_name: str) -> TrainingConfig:
    configs = jsons.loads(data_json, list[TrainingConfig])
    configs = [config for config in configs if config.config_name == config_name]
    if len(configs) == 0:
        raise ValueError("The training config name " + config_name + " does not exit")
    return configs[0]


def get_result_config_from_json(data_json: str, config_name: str) -> ResultConfig:
    configs = jsons.loads(data_json, list[ResultConfig])
    configs = [config for config in configs if config.config_name == config_name]
    if len(configs) == 0:
        raise ValueError("The result config name " + config_name + " does not exit")
    return configs[0]


def get_data_name(data_config: DataConfig):
    if data_config.save_info.name is None:
        return get_default_data_name(data_config)
    else:
        return data_config.save_info.name


def get_default_data_name(data_config: DataConfig):
    return f'data_sr{data_config.sampling_time:.0e}_' \
           f'q{data_config.process_var:.0e}_' \
           f'r{data_config.measurement_var:.0e}_' \
           f'j{data_config.max_taylor_coefficient}_' \
           f'train[{data_config.train_set.batch_size},{data_config.train_set.trajectory_length}]_' \
           f'val[{data_config.val_set.batch_size},{data_config.val_set.trajectory_length}]_' \
           f'test[{data_config.test_set.batch_size},{data_config.test_set.trajectory_length}]'


def get_default_estimator_name(est_config: EstimatorConfig, data_config: DataConfig) -> str:
    if est_config.type == "ExtendedKalmanFilter" or est_config.type == "IdentityEstimator":
        name = f'{est_config.type}_'
        if est_config.random_variance:
            name += f'ri{est_config.random_variance}_'

        return name + data_config.save_info.name

    elif est_config.type == "KalmanNetNN":
        name = f'{est_config.train_config.loss}_' \
               f'ep{est_config.train_config.n_epochs}_' \
               f'train{est_config.train_config.train_size}_' \
               f'val{est_config.train_config.val_size}_' \
               f'test{est_config.train_config.test_size}_' \
               f'lr{est_config.train_config.learning_rate:.0e}_' \
               f'wd{est_config.train_config.weight_decay:.0e}_'
        if est_config.train_config.random_variance:
            name += f'ri{est_config.train_config.random_variance}_'

        return name + data_config.save_info.name

    else:
        raise ValueError("Not a valid estimator type: " + est_config.type)


def get_filter_dict_from_config(config: Configuration) -> dict[str, str]:
    if config.estimator_config.loss == 'MSE':
        loss_name = 'def' + config.estimator_config.loss
    else:
        loss_name = config.estimator_config.train_config.loss
    filter_dict = {'test_random_variance': str(config.estimator_config.random_variance),
                   'estimator_type': config.estimator_config.type,
                   'process_variance': config.data_config.process_var,
                   'measurement_variance': config.data_config.measurement_var,
                   'trajectory_length': config.data_config.test_set.trajectory_length,
                   'loss_name': loss_name}
    if config.estimator_config.train_config is not None:
        filter_dict['training_config'] = "loss='" + config.estimator_config.train_config.loss + "'"
        filter_dict['train_random_variance'] = str(config.estimator_config.train_config.random_variance)

    # if config.estimator_config.train_config is None:
    #     filter_dict = {'random_variance': str(config.estimator_config.random_variance),
    #                    'estimator_type': config.estimator_config.type,
    #                    'process_variance': config.data_config.process_var,
    #                    'measurement_variance': config.data_config.measurement_var,
    #                    'trajectory_length': config.data_config.test_set.trajectory_length,
    #                    'loss_name': 'defMSE'}
    # else:
    #     if config.estimator_config.train_config.loss == 'MSE':
    #         loss_name = 'def' + config.estimator_config.train_config.loss
    #     else:
    #         loss_name = config.estimator_config.train_config.loss
    #     filter_dict = {'random_variance': str(config.estimator_config.random_variance),
    #                    'estimator_type': config.estimator_config.type,
    #                    'process_variance': config.data_config.process_var,
    #                    'measurement_variance': config.data_config.measurement_var,
    #                    'trajectory_length': config.data_config.test_set.trajectory_length,
    #                    'loss_name': loss_name,
    #                    'training_config': "loss='" + config.estimator_config.train_config.loss + "'"}
    return filter_dict

from utils.utils import setup_logging, get_best_model_name

from config import configuration

from estimators.pipeline import Pipeline

from models.models import GerstnerWaves
from models.simulation import DataGeneration

import torch

import os.path

import logging.config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    setup_logging()

    # Configs for getting best models
    # 'len250_r1', 'len250_r1e-1', 'len250_r1e-2', 'len250_r1e-3'
    data_config_name = ['default_r1', 'default_r1e-1', 'default_r1e-2', 'default_r1e-3', 'len250_r1', 'len250_r1e-1', 'len250_r1e-2', 'len250_r1e-3']
    # ['default', 'default', 'default_velMSE', 'default_velMSE', 'default_ri', 'default_ri', 'default_velMSE_ri', 'default_velMSE_ri']
    training_config_name = ['default_ri']
    # #'knet_notrain', 'knet_notrain_velMSE', 'knet_notrain', 'knet_notrain_velMSE', 'knet_notrain_ri', 'knet_notrain_velMSE_ri'
    estimator_config_name = ['knet_default']
    result_config_name = ['default']

    # Configs for testing
    test_data_config_name = ['default_r1', 'default_r1e-1', 'default_r1e-2', 'default_r1e-3', 'len250_r1', 'len250_r1e-1', 'len250_r1e-2', 'len250_r1e-3']
    # 'knet_velMSE_ri', 'knet_velMSE_ri', 'knet_velMSE_ri', 'knet_velMSE_ri', 'knet_posMSE_ri', 'knet_posMSE_ri', 'knet_posMSE_ri', 'knet_posMSE_ri'
    test_estimator_config_name = ['knet_default']

    add_filter_dict = {'n_features': 1}
    group_by = 'measurement_variance'

    model_configs = configuration.get_specific_configs_from_subconfigs(
        data_names=data_config_name,
        estimator_names=estimator_config_name,
        result_names=result_config_name,
        training_names=training_config_name)

    test_configs = configuration.get_specific_configs_from_subconfigs(
        data_names=test_data_config_name,
        estimator_names=test_estimator_config_name,
        result_names=result_config_name,
        training_names=training_config_name)

    logger.debug("Configuration list with size " + str(len(model_configs)) + " has been loaded")

    for model_config, test_config in zip(model_configs, test_configs):
        logger.info("Model configuration: " + model_config.configuration_name)
        logger.info("Test configuration: " + test_config.configuration_name)

        # Test
        logger.debug("Test started ...")

        # Initialize noise params
        process_covariance = torch.mul(torch.eye(4), test_config.data_config.process_var)
        measurement_covariance = torch.mul(torch.Tensor([[1]]), test_config.data_config.measurement_var)

        # Initialize model
        model = GerstnerWaves()
        model.set_process_noise(process_covariance)
        model.set_measurement_noise(measurement_covariance)

        gen = DataGeneration(test_config.data_config.save_info.path, test_config.data_config.save_info.name, model)

        # Pipeline
        pipeline = Pipeline("", "", device='cpu')

        [train_set, val_set, test_set] = gen.load_data()

        # Get actual filter
        filter_dict = configuration.get_filter_dict_from_config(model_config)
        filter_dict.update(add_filter_dict)

        # Get estimator name with the lowest loss
        estimator_name, training_config_name = get_best_model_name(
            model_config.result_config.save_info.path,
            model_config.result_config.save_info.name,
            group_by=group_by,
            filter_dict=filter_dict)

        test_config = configuration.get_config_from_subconfig(test_config.data_config.config_name,
                                                              test_config.estimator_config.config_name,
                                                              test_config.result_config.config_name,
                                                              training_config_name)
        test_config.estimator_config.save_info.name = estimator_name

        pipeline.set_model_path_complete(model_config.estimator_config.save_info.path,
                                         test_config.estimator_config.save_info.name)
        pipeline.set_loss_function(test_config.estimator_config.loss)

        # Use saved estimator
        if os.path.isfile(
                test_config.estimator_config.save_info.path + '/' + test_config.estimator_config.save_info.name + '.pt'):

            if test_config.estimator_config.type == "KalmanNetNN":
                pipeline.set_device("cuda")

            pipeline.load_estimator()
        else:
            logger.warning(
                "Could not find estimator with name: " + test_config.estimator_config.save_info.path + '/' + test_config.estimator_config.save_info.name + '.pt')

        # Test estimator
        pipeline.test_estimator(test_set[2], test_set[1],
                                random_variance=test_config.estimator_config.random_variance)

        # Save results
        if test_config.result_config.save:
            pipeline.save_test_results(test_config)

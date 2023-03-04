from config import configuration

import estimators.pipeline
from estimators.pipeline import Pipeline

from models.models import GerstnerWaves
from models.simulation import DataGeneration

from utils.utils import setup_logging
from utils.plots import GerstnerWavesPlots

import logging

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    setup_logging()

    # ["default_r1", "default_r1_wd1e-4", "default_r1_wd1e-6", "default_r1_lr1e-4", "default_r1_wd1e-7"]
    # config_names = ['default_r1-2_velMSE_lr5e-4_wd1e-4']
    #
    # configs = configuration.get_configs_from_path("config/main_configs.json", config_names)

    save_plots = True

    data_config_name = ['default_r1']
    ekf_config_name = ['ekf_notrain', 'ekf_notrain_velMSE', 'ekf_notrain', 'ekf_notrain_velMSE']
    knet_config_name = ['knet_notrain', 'knet_notrain_velMSE', 'knet_notrain', 'knet_notrain_velMSE']
    # ['default', 'default_lr5e-4', 'default_wd1e-3', 'default_wd1e-6', 'default_wd1e-4', 'default_lr5e-4_wd1e-6','default_lr5e-4_wd1e-4']
    training_config_name = ['default_wd1e-6', 'default_velMSE_wd1e-6', 'default_wd1e-4_ri', 'default_velMSE_ri']
    result_config_name = ['default']

    knet_configs = configuration.get_specific_configs_from_subconfigs(
        data_names=data_config_name,
        estimator_names=knet_config_name,
        training_names=training_config_name,
        result_names=result_config_name)

    ekf_configs = configuration.get_specific_configs_from_subconfigs(
        data_names=data_config_name,
        estimator_names=ekf_config_name,
        training_names=len(ekf_config_name)*[None],
        result_names=result_config_name)

    logger.debug("Configuration list with size " + str(len(knet_configs)) + " has been loaded")

    for knet_config, ekf_config in zip(knet_configs, ekf_configs):
        logger.info("Plot configuration knet: " + knet_config.configuration_name)
        logger.info("Plot configuration ekf: " + ekf_config.configuration_name)

        # Define name
        data_name = knet_config.data_config.save_info.name
        knet_name = knet_config.estimator_config.save_info.name
        ekf_name = ekf_config.estimator_config.save_info.name
        pipeline_name = 'p_' + knet_name
        plot_name = knet_name.replace('_'+data_name, "")+\
                    f'_data_q{knet_config.data_config.process_var:.0e}_' \
                    f'r{knet_config.data_config.measurement_var:.0e}_' \
                    f't{knet_config.data_config.test_set.trajectory_length}'

        datagen = DataGeneration(knet_config.data_config.save_info.path, data_name, GerstnerWaves())
        [train, val, test] = datagen.load_data()

        pipeline_knet = estimators.pipeline.load_pipeline(knet_config.estimator_config.save_info.path, pipeline_name)
        pipeline_knet.set_device("cuda")
        pipeline_knet.set_loss_function(knet_config.estimator_config.loss)
        pipeline_knet.test_estimator(test[2], test[1], random_variance=knet_config.estimator_config.random_variance)

        pipeline_ekf = Pipeline(ekf_config.estimator_config.save_info.path, ekf_name)
        pipeline_ekf.load_estimator()
        pipeline_ekf.set_loss_function(ekf_config.estimator_config.loss)
        pipeline_ekf.test_estimator(test[2], test[1], random_variance=ekf_config.estimator_config.random_variance)

        plot = GerstnerWavesPlots('data/plots', knet_config.result_config.save_info.path, plot_add_name=plot_name, format='eps')
        plot.set_figsize((11, 6))
        plot.set_fontsize(12)

        # Loss
        train_avg_loss, train_state_loss = pipeline_knet.get_loss_train_per_epoch(in_decibel=False)
        val_avg_loss, val_state_loss = pipeline_knet.get_loss_val_per_epoch(in_decibel=False)
        test_state_loss_knet, test_state_loss_batch_knet = pipeline_knet.get_state_loss_test(in_decibel=False)
        test_state_loss_ekf, test_state_loss_batch_ekf = pipeline_ekf.get_state_loss_test(in_decibel=False)
        test_loss_knet = pipeline_knet.get_mean_loss_test(in_decibel=False)
        test_loss_ekf = pipeline_ekf.get_mean_loss_test(in_decibel=False)

        # Plot
        plot.plot_avg_loss_epoch_compare(train_state_loss, val_state_loss, test_loss_knet[0], test_loss_ekf[0],
                                         in_decibel=True, save_plot=save_plots)
        plot.plot_state_loss_epoch(train_state_loss, data_set='training', in_decibel=True, save_plot=save_plots)
        plot.plot_state_loss_epoch(val_state_loss, data_set='validation', in_decibel=True, save_plot=save_plots)
        plot.plot_state_loss_hist(test_state_loss_batch_knet, data_set='test', in_decibel=True, save_plot=save_plots)
        plot.plot_avg_loss_hist(train_state_loss, val_state_loss, test_state_loss_batch_knet, in_decibel=True,
                                save_plot=save_plots)

        logger.info(f'Mean test loss KalmanNet [db]: {pipeline_knet.get_mean_loss_test()}')
        logger.info(f'State test loss KalmanNet [dB]: {pipeline_knet.get_state_loss_test()[0]}')
        logger.info(f'Mean test loss EKF [dB]: {pipeline_ekf.get_mean_loss_test()}')
        logger.info(f'State test loss EKF [dB]: {pipeline_ekf.get_state_loss_test()[0]}')

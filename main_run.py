import numpy as np

from utils.utils import setup_logging

from config import configuration

from KalmanNet_nn import KalmanNetNN
from KalmanNet_sysmdl import SystemModel

from estimators.estimators import ExtendedKalmanFilter, IdentityEstimator
from estimators.pipeline import Pipeline

from models.models import GerstnerWaves, GerstnerWavesGPU
from models.simulation import DataGeneration

import torch

import os.path

import logging.config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    setup_logging()

    # Specific config
    # ["default_r1e-3_wd1e-6", "default_r1e-3_wd1e-4", "default_r1e-3_lr5e-4",
    #                     "default_r1e-3_lr5e-4_wd1e-6", "default_r1e-4_wd1e-6", "default_r1e-4_wd1e-4",
    #                     "default_r1e-4_lr5e-4", "default_r1e-4_lr5e-4_wd1e-6"]
    # config_names = ["default_r1e-4_velMSE_wd1e-3", "default_r1e-4_velMSE_lr5e-4_wd1e-4"]
    # configs = configuration.get_configs_from_path("config/main_configs.json", config_names)

    # Combinations configs (good for testing purposes)
    data_config_name = ['default_r1']  #'len250_r1', 'len250_r1e-1', 'len250_r1e-2', 'len250_r1e-3'
    # 'knet_notrain', 'knet_notrain', 'knet_notrain', 'knet_notrain', 'knet_notrain', 'knet_notrain', 'knet_notrain', 'knet_notrain'
    estimator_config_name = ['knet_posMSE', 'knet_freqMSE']
    # 'default', 'default_lr5e-4', 'default_wd1e-6', 'default_wd1e-4'
    training_config_name = ['default_posMSE', 'default_freqMSE']  # ['default', 'default_lr5e-4', 'default_wd1e-6', 'default_wd1e-4' ///'default_velMSE_wd1e-4', 'default_velMSE_wd1e-6', 'default_velMSE_lr5e-4','default_wd1e-3', 'default_wd1e-6', 'default_lr5e-4'] # ['default', 'default_lr5e-4', 'default_wd1e-3', 'default_wd1e-6', 'default_wd1e-4', 'default_lr5e-4_wd1e-6','default_lr5e-4_wd1e-4']

    result_config_name = ['save']

    configs = configuration.get_specific_configs_from_subconfigs(
        data_names=data_config_name,
        estimator_names=estimator_config_name,
        training_names=training_config_name,
        result_names=result_config_name)

    logger.debug("Configuration list with size " + str(len(configs)) + " has been loaded")

    for config in configs:
        logger.info("Run started with configuration " + config.configuration_name)

        # Initialize noise params
        process_covariance = torch.mul(torch.eye(4), config.data_config.process_var)
        measurement_covariance = torch.mul(torch.Tensor([[1]]), config.data_config.measurement_var)

        # Initialize model
        model = GerstnerWaves()
        model.set_process_noise(process_covariance)
        model.set_measurement_noise(measurement_covariance)

        gen = DataGeneration(config.data_config.save_info.path, config.data_config.save_info.name, model)

        # Pipeline
        pipeline = Pipeline("", "", device='cpu')

        # Data generation
        if config.data_config.generate:
            logger.debug("Data generation started...")

            data_config = config.data_config

            if data_config.from_config is not None:
                old_data_gen = configuration.get_data_config_from_json(
                    configuration.get_json_from_path(configuration.DATA_CONFIG_JSON_PATH), data_config.from_config)
                gen.set_path(old_data_gen.save_info.path, old_data_gen.save_info.name)
                train, val, test = gen.load_data()
                gen.set_path(data_config.save_info.path, config.data_config.save_info.name)
                gen.generate_observations_from_state_and_save(list(train), list(val), list(test))
            else:
                traj_lengths = [data_config.train_set.trajectory_length, data_config.val_set.trajectory_length,
                                data_config.test_set.trajectory_length]
                gen.generate_and_save_train_val_test_data(
                    traj_lengths=traj_lengths,
                    sampling_time=config.data_config.sampling_time)

        [train_set, val_set, test_set] = gen.load_data()

        if config.estimator_config.type == "ExtendedKalmanFilter":
            if config.estimator_config.test:
                logger.debug("EKF test started...")

                pipeline.set_model_path_complete(config.estimator_config.save_info.path, config.estimator_config.save_info.name)
                pipeline.set_loss_function(config.estimator_config.loss)

                # Perform grid search in order to obtain best estimator
                if config.estimator_config.train:
                    process_vars = list(np.power(10, -(np.linspace(0, 30, 10)) / 10))
                    estimator = ExtendedKalmanFilter(
                        model.get_continuous_state_evolution_callable(),
                        model.get_measurement_matrix(),
                        process_covariance,
                        measurement_covariance,
                        delta_t=config.data_config.sampling_time,
                        n_taylor_coefficients=config.data_config.max_taylor_coefficient)
                    pipeline.set_estimator(estimator)
                    pipeline.test_best_param_for_estimator(test_set[2],
                                                           test_set[1],
                                                           process_vars,
                                                           random_variance=config.estimator_config.random_variance)
                # Use saved estimator
                else:
                    if os.path.isfile(
                            config.estimator_config.save_info.path + '/' + config.estimator_config.save_info.name + '.pt'):
                        pipeline.load_estimator()
                    else:
                        logger.warning(
                            "Could not find estimator with name: " + config.estimator_config.save_info.path + '/' + config.estimator_config.save_info.name + '.pt')

                # Test estimator
                pipeline.test_estimator(test_set[2], test_set[1], random_variance=config.estimator_config.random_variance)

                # Save results
                if config.result_config.save:
                    pipeline.save_test_results(config)

        elif config.estimator_config.type == "IdentityEstimator":
            if config.estimator_config.test:
                logger.debug("Identity test started...")

                pipeline.set_model_path_complete(config.estimator_config.save_info.path, config.estimator_config.save_info.name)
                pipeline.set_loss_function(config.estimator_config.loss)

                estimator = IdentityEstimator(
                    model.get_measurement_matrix().size(dim=1),
                    model.get_measurement_matrix().size(dim=0))
                pipeline.set_estimator(estimator)

                # Test estimator
                pipeline.test_estimator(test_set[2], test_set[1], random_variance=config.estimator_config.random_variance)

                # Save results
                if config.result_config.save:
                    pipeline.save_test_results(config)

        elif config.estimator_config.type == "KalmanNetNN":
            # Knet training
            if config.estimator_config.train:
                model_gpu = GerstnerWavesGPU()
                sys_model = SystemModel(
                    model_gpu.f_discrete,
                    process_covariance,
                    model_gpu.h_cor,
                    measurement_covariance,
                    config.data_config.train_set.trajectory_length,
                    config.data_config.test_set.trajectory_length,
                    prior_Q=torch.zeros((4, 4)),
                    prior_Sigma=torch.zeros((4, 4)),
                    prior_S=torch.Tensor([[0]]))
                sys_model.InitSequence(torch.Tensor([0, 1, 1, 0]), torch.eye(4) * 0 * 0)
                kalman_net = KalmanNetNN()
                kalman_net.Build(sys_model)
                # elif config.estimator_config.type == "KalmanNetSophisticated":
                #     kalman_net = KalmanNetSophisticated(
                #         model.get_continuous_state_evolution_callable(),
                #         model.get_measurement_matrix(),
                #         delta_t=config.data_config.sample_rate,
                #         n_taylor_coefficients=config.data_config.max_taylor_coefficient,
                #         is_nonlinear_evolution=True,
                #         device='cuda')
                # else:
                #     raise ValueError("The NN estimator has no valid type")
                pipeline.set_device("cuda")
                pipeline.set_model_path(config.estimator_config.save_info.path)
                pipeline.set_estimator(kalman_net, estimator_file_name=config.estimator_config.save_info.name)
                pipeline.set_training_params(config.estimator_config.train_config)

                pipeline.train_estimator(
                    train_set[2],
                    train_set[1],
                    val_set[2],
                    val_set[1],
                    train_size=config.estimator_config.train_config.train_size,
                    val_size=config.estimator_config.train_config.val_size,
                    add_increase_timestep=10,
                    random_variance=config.estimator_config.train_config.random_variance)

                # Save pipeline
                if config.estimator_config.save_pipeline:
                    pipeline.save()

            # Knet test
            if config.estimator_config.test:
                # Test if estimator exist
                if os.path.isfile(config.estimator_config.save_info.path + '/' + config.estimator_config.save_info.name + '.pt'):
                    pipeline.set_model_path_complete(config.estimator_config.save_info.path,
                                                     config.estimator_config.save_info.name)
                    pipeline.set_device("cuda")
                    pipeline.load_estimator()
                    pipeline.set_loss_function(config.estimator_config.loss)
                    pipeline.test_estimator(test_set[2], test_set[1], test_size=config.estimator_config.train_config.test_size, random_variance=config.estimator_config.random_variance)

                    if config.result_config.save:
                        pipeline.save_test_results(config)

                else:
                    logger.warning(
                        'Could not find estimator with name: ' + config.estimator_config.save_info.path + '/' + config.estimator_config.save_info.name + '.pt')
        #
        # if config.ekf_config.test:
        #     logger.debug("EKF test started...")
        #
        #     pipeline.set_model_path_complete(config.ekf_config.save_info.path, config.ekf_config.save_info.name)
        #     pipeline.set_loss_function(config.ekf_config.loss)
        #
        #     # Perform grid search in order to obtain best estimator
        #     if config.ekf_config.type == 'ExtendedKalmanFilter' and config.ekf_config.save:
        #         process_vars = list(np.power(10, -(np.linspace(0, 30, 10)) / 10))
        #         estimator = ExtendedKalmanFilter(
        #             model.get_continuous_state_evolution_callable(),
        #             model.get_measurement_matrix(),
        #             process_covariance,
        #             measurement_covariance,
        #             delta_t=config.data_config.sample_rate,
        #             n_taylor_coefficients=config.data_config.max_taylor_coefficient)
        #         pipeline.set_estimator(estimator)
        #         pipeline.test_best_param_for_estimator(test_set[2],
        #                                                test_set[1],
        #                                                process_vars,
        #                                                random_variance=config.ekf_config.random_variance)
        #
        #     # Only for testing purposes
        #     elif config.ekf_config.type == 'IdentityEstimator':
        #         estimator = IdentityEstimator(
        #             model.get_measurement_matrix().size(dim=1),
        #             model.get_measurement_matrix().size(dim=0))
        #         pipeline.set_estimator(estimator)
        #
        #     # Use saved estimator
        #     elif not config.ekf_config.save:
        #         if os.path.isfile(config.ekf_config.save_info.path + '/' + config.ekf_config.save_info.name + '.pt'):
        #             pipeline.load_estimator()
        #         else:
        #             logger.warning(
        #                 "Could not find estimator with name: " + config.ekf_config.save_info.path + '/' + config.ekf_config.save_info.name + '.pt')
        #
        #     else:
        #         raise ValueError("Estimator " + config.ekf_config.type + " is not valid")
        #
        #     # Test estimator
        #     pipeline.test_estimator(test_set[2], test_set[1], random_variance=config.estimator_config.random_variance)
        #
        #     # Save results
        #     if config.result_config.save:
        #         pipeline.save_test_results(config)



    # # Changeable parameters
    # # Define tasks
    # generate_data = False
    # test_kalman_filter = False
    # train_kalman_net = True
    # test_kalman_net = True
    #
    # estimator_name = "KalmanNetNN"
    #
    # # Data specific params (default: [(1e-4, 1), (1e-4, 0.1), (1e-4, 0.01), (1e-4, 0.001), (1e-4, 1e-4)])
    # noise_params = [(1e-4, 1)]  # (process_var, measurement_var)
    # delta_t = 1e-1
    # seq_lengths_per_batch = [100, 100, 100]  # [train, val, test]
    # n_batch = [500, 100, 100]  # [train, val, test]
    # max_taylor_coefficient = 5
    #
    # # Training params (default: 100, 40, 10, 1e-3, 1e-5)
    # n_epochs = 100
    # batch_size = 50
    # val_size = 20
    # learning_rate = 5e-3
    # weight_decay = 1e-6
    # continuous_batches = False
    #
    # # Define folders
    # data_folder_name = 'data/gerstner/single_obs'
    # model_folder_name = 'estimators/saved_models_sophisticated/single_obs'
    # result_path = 'data/gerstner/results.csv'
    #
    # # Additional information
    # add_info_nn = 'last_validation_wd=1e-6'
    # add_info_kf = 'full information'
    # add_info_data = 'cont'
    #
    # add_info_nn_all = f'{add_info_nn}; {n_epochs=}; {val_size=}; {batch_size=}; {learning_rate=:.2e}; {weight_decay=:.2e}; {continuous_batches=}'
    #
    # result_header = pd.read_csv(result_path, nrows=0)
    # result_header = list(result_header.columns)
    #
    # for i, (process_variance, measurement_variance) in enumerate(noise_params):
    #
    #     filename = f'train{n_batch[0]}_val{n_batch[1]}_test{n_batch[2]}_q{process_variance:.0e}_r{measurement_variance:.0e}_{str(seq_lengths_per_batch)}'
    #
    #     result_list = [None] * len(result_header)
    #
    #     measurement_covariance = torch.mul(torch.Tensor([[1]]), measurement_variance).to(dev)
    #     process_covariance = torch.mul(torch.eye(4), process_variance).to(dev)
    #
    #     model = GerstnerWaves()
    #
    #     # Initialize noise params
    #     model.set_process_noise(process_covariance)
    #     model.set_measurement_noise(measurement_covariance)
    #
    #     # Data Generation
    #     data_filename = f'{filename}_sr={delta_t:.0e}_j={5}_{add_info_data}'
    #     datageneration = DataGeneration(data_folder_name, data_filename, model)
    #
    #     # Pipeline
    #     pipeline = Pipeline(model_folder_name, "", device='cpu')
    #     pipeline.set_initial_conditions(torch.Tensor([0, 1, 1, 0]), torch.zeros((4, 4)))
    #
    #     if generate_data:
    #         datageneration.set_data_generation_parameters(n_batch[0], n_batch[1], n_batch[2])
    #         datageneration.generate_and_save_train_val_test_data(random_init=[False, False, False],
    #                                                              variance=[0.5, 0.5, 0.5],
    #                                                              seq_len=seq_lengths_per_batch,
    #                                                              sample_rate=[delta_t, delta_t, delta_t])
    #
    #     # Data Loading
    #     [timevectors, train_input, train_target, val_input, val_target, test_input,
    #      test_target] = datageneration.load_training_data_gpu()
    #
    #     # Test KalmanFilter
    #     if test_kalman_filter:
    #         model_file_name = 'KalmanFilter_gerstner'
    #         kalman_filter = ExtendedKalmanFilter(model.get_continuous_state_evolution_callable(),
    #                                              model.get_measurement_matrix(),
    #                                              process_covariance,
    #                                              measurement_covariance,
    #                                              delta_t=delta_t,
    #                                              n_taylor_coefficients=max_taylor_coefficient)
    #         pipeline.set_estimator(kalman_filter, estimator_file_name=model_file_name)
    #         pipeline.test_estimator(test_input, test_target)
    #

    # Code from paper with slight modifications
    # train_input = torch.permute(train_input, (0, 2, 1))
    # train_target = torch.permute(train_target, (0, 2, 1))[:, :, 1:]
    # val_input = torch.permute(val_input, (0, 2, 1))
    # val_target = torch.permute(val_target, (0, 2, 1))[:, :, 1:]
    # test_input = torch.permute(test_input, (0, 2, 1))
    # test_target = torch.permute(test_target, (0, 2, 1))[:, :, 1:]

    # model_gpu = GerstnerWavesGPU()
    # sys_model = SystemModel(model_gpu.f_discrete, process_covariance, model_gpu.h_cor,
    #                         measurement_covariance, seq_lengths_per_batch[0], seq_lengths_per_batch[2])
    # # prior_Q=torch.zeros((4, 4)), prior_Sigma=torch.zeros((4, 4)),
    # # prior_S=torch.Tensor([[0]]))
    # sys_model.InitSequence(torch.Tensor([0, 1, 1, 0]), torch.eye(4) * 0 * 0)

    # [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model,
    # test_input, test_target[:, :, 1:])

    # k_net_filename = filename + "_j=5_arch2"
    #
    # KNet_Pipeline = Pipeline_EKF(strTime, model_folder_name, k_net_filename)
    # KNet_Pipeline.setssModel(sys_model)
    # KNet_model = KalmanNetNN()
    # KNet_model.Build(sys_model)
    # KNet_Pipeline.setModel(KNet_model)
    # KNet_Pipeline.setTrainingParams(n_Epochs=n_epochs, n_Batch=batch_size, learningRate=learning_rate,
    #                                 weightDecay=weight_decay)
    #
    # kalman_net_name = f'{estimator_name}_{filename}_j={max_taylor_coefficient}_{add_info_nn}'
    # pipeline.set_device('cuda')
    # pipeline.set_model_name(kalman_net_name)
    #
    # if train_kalman_net:
    #     if estimator_name == "KalmanNetNN":
    #         kalman_net = KalmanNetNN()
    #         kalman_net.Build(sys_model)
    #     elif estimator_name == "KalmanNetSophisticated":
    #         kalman_net = KalmanNetSophisticated(model.get_continuous_state_evolution_callable(),
    #                                             model.get_measurement_matrix(),
    #                                             delta_t=delta_t,
    #                                             n_taylor_coefficients=max_taylor_coefficient,
    #                                             is_nonlinear_evolution=True,
    #                                             device='cuda')
    #
    #     pipeline.set_estimator(kalman_net)
    #     pipeline.set_training_params(n_epochs, batch_size, learning_rate, weight_decay)
    #     pipeline.train_estimator(train_input, train_target, val_input, val_target, val_size=val_size,
    #                              add_increase_timestep=10)
    #     # train_loss, train_state_loss_per_epoch = pipeline.get_loss_train_per_epoch()
    #     # val_loss, val_state_loss_per_epoch = pipeline.get_loss_val_per_epoch()
    #     pipeline.save()
    #
    #     # if os.path.isfile(model_folder_name + '/pipeline_' + k_net_filename + '.pt'):
    #     #     KNet_Pipeline = torch.load(model_folder_name + '/pipeline_' + k_net_filename + '.pt')
    #     #     logger.info("Loaded old pipeline")
    #     # KNet_Pipeline.NNTrain(train_input, train_target, val_input, val_target, val_size=val_size)
    #
    # if test_kalman_net:
    # k_net = torch.load(model_folder_name + '/model_'+k_net_filename + ".pt")
    # KNet_Pipeline.setModel(k_net)
    # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg,
    #  KNet_test] = KNet_Pipeline.NNTest(
    #     test_input, test_target)
    # pipeline.load_estimator()
    # # pipeline.test_estimator(test_input, test_target)
    #
    # result_list[0] = type(pipeline._estimator).__name__
    # result_list[1] = model.get_observation_dim()
    # result_list[2] = seq_lengths_per_batch[2] * n_batch[2]
    # result_list[3] = process_variance
    # result_list[4] = measurement_variance
    # result_list[5] = pipeline.get_mean_loss_test()[0][0]
    # result_list[6] = datetime.now().isoformat('T', 'seconds')
    # result_list[7] = data_filename
    # result_list[8] = add_info_nn_all
    # result_list[9] = pipeline.get_mean_loss_test()[1]
    #
    # result_pandas = pd.DataFrame(data=[result_list], columns=result_header)
    # result_pandas.to_csv(result_path, index=False, mode='a', header=False)
    # logger.info("Results saved in: " + result_path)

    # KNet_Pipeline.save()

    # # Train KalmanNet
    # pipeline = Pipeline(model_folder_name, filename, device='cuda')
    #
    # # Initialize pipeline
    # k_net_simple = KalmanNetSimple(model.get_continuous_state_evolution_callable(),
    #                                model.get_measurement_matrix(), n_taylor_coefficients=5,
    #                                is_nonlinear_model=True)
    # pipeline.set_estimator(k_net_simple)
    # pipeline.set_initial_conditions(torch.ones(4), torch.eye(4)*0*0)
    # pipeline.set_training_params(50, 100, 1e-3, 1e-5)
    #
    # # Train KalmanNet
    # pipeline.train_estimator(1000, train_input, train_target, 100, val_input, val_target)
    #
    # # Test KalmanNet
    # pipeline.test_estimator(timevectors[2], test_input, test_target)
    # pipeline.save()

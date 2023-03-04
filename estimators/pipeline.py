from config.configuration import Configuration, TrainingConfig

from estimators.estimators import Estimator

from KalmanNet_nn import KalmanNetNN

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
import pandas as pd

import logging

import math

import random

import os

import time
from datetime import datetime

from typing import Optional

# Module logger
logger = logging.getLogger(__name__)


def load_pipeline(path: str, name: str):
    complete_name = path + '/' + name + '.pt'
    pipeline = torch.load(complete_name)
    if isinstance(pipeline, Pipeline):
        logger.debug("Pipeline loaded from " + complete_name)
        return pipeline
    else:
        raise ValueError("It has to be an object of class Pipeline")


class Pipeline:
    _model_complete_name: str
    _pipeline_name: str

    _dev: torch.device

    def __init__(self, model_path: str, model_name: str, device='cpu'):
        self.set_device(device)

        self._model_path = model_path
        self._model_name = model_name

        # Estimator
        self._estimator = None

        # Training param
        self._train_size = None
        self._val_size = None
        self._weight_decay = None
        self._learning_rate = None
        self._n_epochs = None
        self._optimizer = None
        self._loss_function = None
        self._loss_name = None
        self._loss_start_state = None
        self._loss_end_state = None

        # Current training variables
        self.__loss_train_db_epoch = None
        self.__loss_train_linear_epoch = None
        self.__loss_train_state_epoch = None

        self.__loss_val_db_epoch = None
        self.__loss_val_linear_epoch = None
        self.__loss_val_state_epoch = None

        self.__optimal_loss_db = 1000
        self.__optimal_epoch = 0

        # Current test variables
        self.__loss_test_db_mean = None
        self.__loss_test_std = None
        self.__loss_test_db_std = None
        self.__loss_test_state_batch = None
        self.__loss_test_state = None

        self.__initialize_pipeline()

    def __initialize_pipeline(self):
        self.__build_model_names()

    def __build_model_names(self):
        self._model_complete_name = f'{self._model_path}/{self._model_name}.pt'
        self._pipeline_name = self._model_path + "/p_" + self._model_name + ".pt"
        logger.debug("Update model name to: " + self._model_complete_name)

    def __reset_all_current_params(self):
        self.__loss_train_db_epoch = None
        self.__loss_train_linear_epoch = None
        self.__loss_val_db_epoch = None
        self.__loss_val_linear_epoch = None

        self.__loss_test_db_mean = None
        self.__loss_test_std = None
        self.__loss_test_db_std = None
        self.__loss_test_state_batch = None
        self.__loss_test_state = None

        self.__loss_train_db_epoch = None
        self.__loss_train_linear_epoch = None
        self.__loss_train_state_epoch = None

        self.__loss_val_db_epoch = None
        self.__loss_val_linear_epoch = None
        self.__loss_val_state_epoch = None

        self.__optimal_loss_db = 1000
        self.__optimal_epoch = 0

    def __get_initial_condition(self, true_initial_condition: Tensor,
                                random_variance: Optional[list] = None) -> tuple[Tensor, Tensor]:
        if random_variance is not None:
            random_variance = torch.Tensor(random_variance)
            modified_initial_condition = true_initial_condition + torch.normal(mean=0, std=torch.sqrt(random_variance))
            if modified_initial_condition[2] < 0:
                modified_initial_condition[2] = torch.abs(modified_initial_condition[2])
            initial_covariance = torch.diag(random_variance)
        else:
            modified_initial_condition = true_initial_condition
            initial_covariance = torch.zeros((true_initial_condition.size(dim=0), true_initial_condition.size(dim=0)))

        return modified_initial_condition, initial_covariance

    def save(self):
        torch.save(self, self._pipeline_name)
        logger.info("Saved pipeline at: " + self._pipeline_name)

    def save_estimator(self):
        torch.save(self._estimator, self._model_complete_name)
        logger.info("Saved model at: " + self._model_complete_name)

    def save_test_results(self, config: Configuration):
        result_complete_path = config.result_config.save_info.path + '/' + config.result_config.save_info.name + '.csv'
        result_header = pd.read_csv(result_complete_path, nrows=0)
        result_dict = {key: None for key in list(result_header.columns)}

        result_dict['estimator_type'] = type(self._estimator).__name__

        if isinstance(self._estimator, KalmanNetNN):
            result_dict['n_features'] = self._estimator.n
        else:
            result_dict['n_features'] = self._estimator.get_observation_dim()
        result_dict['trajectory_length'] = config.data_config.test_set.trajectory_length
        result_dict['process_variance'] = config.data_config.process_var
        result_dict['measurement_variance'] = config.data_config.measurement_var

        mean_loss, mean_std = self.get_mean_loss_test()
        (mean_state_loss, mean_state_std), __ = self.get_state_loss_test()

        if self._loss_name == "MSE":
            result_dict['loss_name'] = "def"+self._loss_name
        else:
            result_dict['loss_name'] = self._loss_name

        result_dict['loss_in_dB'] = mean_loss
        result_dict['loss_std_in_dB'] = mean_std
        result_dict['state_loss_in_dB'] = mean_state_loss
        result_dict['state_loss_std_in_dB'] = mean_state_std

        result_dict['time'] = datetime.now().isoformat('T', 'seconds')
        result_dict['data_config'] = str(config.data_config)
        result_dict['training_config'] = str(config.estimator_config.train_config)
        result_dict['data_name'] = config.data_config.save_info.name

        result_dict['model_name'] = str(config.estimator_config.save_info.name)

        if config.estimator_config.train_config is not None:
            result_dict['train_random_variance'] = str(config.estimator_config.train_config.random_variance)
        else:
            result_dict['train_random_variance'] = str(config.estimator_config.train_config)

        result_dict['test_random_variance'] = str(config.estimator_config.random_variance)

        result_pandas = pd.Series(result_dict).to_frame().T
        result_pandas.to_csv(result_complete_path, index=False, mode='a', header=False)
        logger.info("Results saved in: " + result_complete_path)

    def load_estimator(self, estimator_path=None, estimator_name=None):
        if estimator_path is not None:
            self._model_path = estimator_path
            self.__build_model_names()

        if estimator_name is not None:
            self._model_name = estimator_name
            self.__build_model_names()

        estimator = torch.load(self._model_complete_name, map_location=self._dev)

        self.set_estimator(estimator)

    # Setter methods
    def set_device(self, dev):
        if not torch.cuda.is_available() and dev == 'cuda':
            logger.warning("GPU is not available")
            self._dev = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
            logger.debug('Running on the CPU')
        elif torch.cuda.is_available() and dev == 'cuda':
            self._dev = torch.device("cuda:0")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            logger.debug('Running on the GPU')
        else:
            self._dev = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
            logger.debug('Running on the CPU')

    def set_estimator(self, estimator, estimator_file_name=None):
        if isinstance(estimator, KalmanNetNN):
            self._estimator = estimator.to(self._dev)
            self._estimator.dev = self._dev
        elif isinstance(estimator, nn.Module):
            self._estimator = estimator.to(self._dev)
            self._estimator.initialize(self._dev)
        elif isinstance(estimator, Estimator):
            self._estimator = estimator
            self._estimator.initialize(self._dev)
        else:
            raise ValueError("Estimator is not valid is not a subclass of Estimator")

        if estimator_file_name is not None:
            self._model_name = estimator_file_name

        self.__build_model_names()
        self.__reset_all_current_params()

        self._loss_function = nn.MSELoss(reduction='none')

        logger.info("Set new estimator of type: " + type(estimator).__name__)

    def set_model_path(self, model_path: str):
        self._model_path = model_path
        self.__build_model_names()

    def set_model_name(self, model_name: str):
        self._model_name = model_name
        self.__build_model_names()

    def set_model_path_complete(self, model_path: str, model_name: str):
        self._model_path = model_path
        self._model_name = model_name
        self.__build_model_names()

    def set_training_params(self, train_config: TrainingConfig):
        self._n_epochs = train_config.n_epochs
        self._learning_rate = train_config.learning_rate
        self._weight_decay = train_config.weight_decay
        self._loss_name = train_config.loss

        # Loss function (MSE)
        self.set_loss_function(train_config.loss)

        # Define optimizer
        self._optimizer = torch.optim.Adam(self._estimator.parameters(), lr=self._learning_rate,
                                           weight_decay=self._weight_decay)
        logger.debug("Set all training parameters")

    def set_loss_function(self, loss_name: str):
        self._loss_name = loss_name
        if self._loss_name == 'MSE':
            self._loss_start_state = 0
            self._loss_end_state = None
        elif self._loss_name == 'velMSE':
            self._loss_start_state = 1
            self._loss_end_state = self._loss_start_state + 1
        elif self._loss_name == 'posMSE':
            self._loss_start_state = 0
            self._loss_end_state = self._loss_start_state + 1
        elif self._loss_name == 'freqMSE':
            self._loss_start_state = 2
            self._loss_end_state = self._loss_start_state+1
        else:
            raise ValueError(loss_name + " is not a valid loss name")

        self._loss_function = nn.MSELoss(reduction='none')
        logger.debug('Set new loss function: ' + self._loss_name)

    # Getter methods
    def get_estimator(self):
        return self._estimator

    def get_loss_train_per_epoch(self, in_decibel=True) -> tuple[np.ndarray, np.ndarray]:
        if self.__loss_train_db_epoch is None or self.__loss_train_state_epoch is None:
            raise ValueError("There is no loss train result, first train the estimator in order to get the data")
        elif in_decibel:
            return self.__loss_train_db_epoch, 10 * np.log10(self.__loss_train_state_epoch)
        else:
            return self.__loss_train_linear_epoch, self.__loss_train_state_epoch

    def get_loss_val_per_epoch(self, in_decibel=True) -> tuple[np.ndarray, np.ndarray]:
        if self.__loss_val_db_epoch is None or self.__loss_val_state_epoch is None:
            raise ValueError("There is no loss validation result, first train the estimator in order to get the data")
        elif in_decibel:
            return self.__loss_val_db_epoch, 10 * np.log10(self.__loss_val_state_epoch)
        else:
            return self.__loss_val_linear_epoch, self.__loss_val_state_epoch

    def get_mean_loss_test(self, in_decibel=True) -> list[float, float]:
        if self.__loss_test_db_mean is None:
            raise ValueError("There is no loss test result, first test the estimator in order to get the data")
        elif in_decibel:
            return [self.__loss_test_db_mean, self.__loss_test_db_std]
        else:
            return [math.pow(10, self.__loss_test_db_mean / 10), self.__loss_test_std]

    def get_state_loss_test(self, in_decibel=True) -> tuple[list[np.ndarray, np.ndarray], np.ndarray]:
        if self.__loss_test_state is None or self.__loss_test_state_batch is None:
            raise ValueError("There is no loss test result, first test the estimator in order to get the data")
        elif in_decibel:
            test_state_std = np.std(self.__loss_test_state_batch, axis=0)
            return [10 * np.log10(self.__loss_test_state),
                    10 * np.log10(test_state_std + self.__loss_test_state) - 10 * np.log10(self.__loss_test_state)], \
                   10 * np.log10(self.__loss_test_state_batch)
        else:
            test_state_std = np.std(self.__loss_test_state_batch, axis=0)
            return [self.__loss_test_state, test_state_std], self.__loss_test_state_batch

    # Train and test methods
    def train_estimator(self, train_input: Tensor,
                        train_target: Tensor,
                        val_input: Tensor,
                        val_target: Tensor,
                        train_size: Optional[int] = None,
                        val_size: Optional[int] = None,
                        add_increase_timestep=10,
                        mul_decrease_timestep=2,
                        random_variance: Optional[list] = None):
        logger.debug("Training started...")

        # Guarantees reconstructibility
        random.seed(42)

        start_time = time.time()

        if train_size is None:
            self._train_size = train_input.size(dim=0) * train_input.size(dim=1)
        else:
            self._train_size = train_size

        if val_size is None:
            self._val_size = val_input.size(dim=0) * val_input.size(dim=1)
        else:
            self._val_size = val_size

        train_traj_len = train_input.size(dim=1)
        val_traj_len = val_input.size(dim=1)
        train_batch_size = math.ceil(self._train_size / train_traj_len)
        val_batch_size = math.ceil(self._val_size / val_traj_len)

        if train_batch_size > train_input.size(dim=0):
            logger.warning("Desired training size is too big, will continue with using total training set")
            train_batch_size = train_input.size(dim=0)
        if val_batch_size > val_input.size(dim=0):
            logger.warning("Desired validation size is too big, will continue with using total validation set")
            val_batch_size = val_input.size(dim=0)

        if not isinstance(self._estimator, nn.Module):
            raise ValueError("The estimator is not a NN and hence cannot be trained")

        elif isinstance(self._estimator, KalmanNetNN):
            self.__loss_val_linear_epoch = np.empty(self._n_epochs)
            self.__loss_val_db_epoch = np.empty(self._n_epochs)
            self.__loss_val_state_epoch = np.empty((self._n_epochs, self._estimator.m))

            self.__loss_train_linear_epoch = np.empty(self._n_epochs)
            self.__loss_train_db_epoch = np.empty(self._n_epochs)
            self.__loss_train_state_epoch = np.empty((self._n_epochs, self._estimator.m))

            train_input = torch.permute(train_input, (0, 2, 1))
            train_target = torch.permute(train_target, (0, 2, 1))
            val_input = torch.permute(val_input, (0, 2, 1))
            val_target = torch.permute(val_target, (0, 2, 1))

            max_train_seq_len = train_input.size(dim=2)
            train_seq_len = int(0.1 * max_train_seq_len)

            train_input = train_input.to(self._dev, non_blocking=True)
            train_target = train_target.to(self._dev, non_blocking=True)
            val_input = val_input.to(self._dev, non_blocking=True)
            val_target = val_target.to(self._dev, non_blocking=True)

            val_batch_list = random.sample(range(0, val_input.size(dim=0)), val_batch_size)
            val_seq_len = val_input.size(dim=2)

            # Epochs
            for epoch in range(0, self._n_epochs):
                logger.debug("Epoch: " + str(epoch + 1))

                loss_val_batch = torch.empty(val_batch_size)
                loss_train_batch = torch.empty(train_batch_size)

                loss_train_state = torch.empty((train_batch_size, self._estimator.m))
                loss_val_state = torch.empty((val_batch_size, self._estimator.m))

                # Validation
                logger.debug("Validate estimator...")
                self._estimator.eval()

                for i, batch_nr in enumerate(val_batch_list):

                    # Set initial state
                    initial_condition, __ = self.__get_initial_condition(val_target[batch_nr, :, 0],
                                                                         random_variance=random_variance)
                    self._estimator.InitSequence(initial_condition, val_seq_len)

                    # Initialize hidden GRU state
                    self._estimator.init_hidden()

                    y_validation = val_input[batch_nr, :, :]
                    estimated_state_val = torch.empty(self._estimator.m, val_seq_len).to(self._dev, non_blocking=True)

                    for time_step in range(0, val_seq_len):
                        estimated_state_val[:, time_step] = self._estimator(y_validation[:, time_step])

                    # Compute validation loss
                    loss_val = self._loss_function(estimated_state_val,
                                                   val_target[batch_nr, :, :val_seq_len])
                    loss_val_state[i, :] = torch.mean(loss_val, dim=1)
                    loss_val_batch[i] = torch.mean(loss_val[self._loss_start_state: self._loss_end_state, :])

                    # Early stopping if tensor is nan
                    if torch.isnan(loss_val_batch[i]):
                        break

                # Average
                self.__loss_val_linear_epoch[epoch] = np.mean(loss_val_batch.numpy(force=True))
                self.__loss_val_db_epoch[epoch] = 10 * np.log10(self.__loss_val_linear_epoch[epoch])
                self.__loss_val_state_epoch[epoch, :] = np.mean(loss_val_state.numpy(force=True), axis=0)

                logger.info(f'Validation loss: {self.__loss_val_db_epoch[epoch]:0.4f} [dB]')

                if self.__loss_val_db_epoch[epoch] < self.__optimal_loss_db:
                    logger.info(
                        f'Loss improvement: {self.__optimal_loss_db - self.__loss_val_db_epoch[epoch]:0.4f} [dB]')
                    self.__optimal_loss_db = self.__loss_val_db_epoch[epoch]
                    self.__optimal_epoch = epoch + 1

                    self.save_estimator()

                logger.info(f'Best loss so far: {self.__optimal_loss_db:0.4f} [dB] in epoch: {self.__optimal_epoch}')

                # Training
                logger.debug("Train estimator...")
                self._estimator.train()

                batch_optimizing_loss_sum = 0

                for batch_nr in range(0, train_batch_size):

                    estimated_state_train = torch.empty(self._estimator.m, train_seq_len).to(self._dev,
                                                                                             non_blocking=True)

                    actual_batch_nr = random.randint(0, train_input.size(dim=0) - 1)

                    # Set initial condition
                    initial_condition, __ = self.__get_initial_condition(train_target[actual_batch_nr, :, 0],
                                                                         random_variance=random_variance)
                    self._estimator.InitSequence(initial_condition, train_seq_len)

                    # Initialize hidden GRU state
                    self._estimator.init_hidden()

                    y_training = train_input[actual_batch_nr, :, :]

                    for time_step in range(0, train_seq_len):
                        estimated_state_train[:, time_step] = self._estimator(y_training[:, time_step])

                    # Compute Training Loss
                    loss_train = self._loss_function(estimated_state_train,
                                                     train_target[actual_batch_nr, :, :train_seq_len])
                    loss_train_state[batch_nr, :] = torch.mean(loss_train, dim=1)
                    loss_train_batch[batch_nr] = torch.mean(loss_train[self._loss_start_state: self._loss_end_state, :])

                    if train_batch_size < 10 or (batch_nr % math.ceil(train_batch_size / 10)) == 0:
                        logger.debug(
                            f'[{batch_nr}|{train_batch_size}]: Training batch loss: {10 * math.log10(loss_train_batch[batch_nr]):0.4f} [dB]')

                    batch_optimizing_loss_sum = batch_optimizing_loss_sum + torch.mean(
                        loss_train[self._loss_start_state: self._loss_end_state, :])

                # Average
                self.__loss_train_linear_epoch[epoch] = np.mean(loss_train_batch.numpy(force=True))
                self.__loss_train_db_epoch[epoch] = 10 * np.log10(self.__loss_train_linear_epoch[epoch])
                self.__loss_train_state_epoch[epoch, :] = np.mean(loss_train_state.numpy(force=True), axis=0)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                self._optimizer.zero_grad()

                if not np.isnan(self.__loss_train_linear_epoch[epoch]) and not np.isinf(
                        self.__loss_train_linear_epoch[epoch]):
                    if train_seq_len + add_increase_timestep <= max_train_seq_len:
                        train_seq_len += add_increase_timestep
                        logger.debug("Increases train sequence length to: " + str(train_seq_len))
                    else:
                        train_seq_len = max_train_seq_len

                    # Backward pass: compute gradient of the loss with respect to model parameters
                    batch_optimizing_loss_mean = batch_optimizing_loss_sum / train_batch_size
                    batch_optimizing_loss_mean.backward()

                    # Calling the step function on an Optimizer makes an update to its parameters
                    self._optimizer.step()
                else:
                    if train_seq_len / mul_decrease_timestep > 1:
                        train_seq_len = int(train_seq_len / mul_decrease_timestep)
                    else:
                        train_seq_len = 1
                    logger.debug("Decreased train sequence length to: " + str(train_seq_len))

                # Training summary
                logger.info(f'Training loss: {self.__loss_train_db_epoch[epoch]:0.4f} [dB]')

            logger.info(f'Train execution time: {(time.time() - start_time) / 60:0.1f} [m]')

            if os.path.isfile(self._model_complete_name):
                self._estimator = torch.load(self._model_complete_name)
            else:
                logger.warning("Training was not sucessfull, could not find estimator: " + self._model_complete_name)

        # else:
        #     self.__loss_val_linear_epoch = np.empty(self._n_epochs)
        #     self.__loss_val_db_epoch = np.empty(self._n_epochs)
        #     self.__loss_val_state_epoch = np.empty((self._n_epochs, self._estimator.get_state_dim()))
        #
        #     self.__loss_train_linear_epoch = np.empty(self._n_epochs)
        #     self.__loss_train_db_epoch = np.empty(self._n_epochs)
        #     self.__loss_train_state_epoch = np.empty((self._n_epochs, self._estimator.get_state_dim()))
        #
        #     max_train_seq_len = train_input.size(dim=1)
        #     train_seq_len = int(0.1 * max_train_seq_len)
        #
        #     train_input = train_input.to(self._dev, non_blocking=True)
        #     train_target = train_target.to(self._dev, non_blocking=True)
        #     val_input = val_input.to(self._dev, non_blocking=True)
        #     val_target = val_target.to(self._dev, non_blocking=True)
        #
        #     val_starting_batch = 80  # random.randint(0, val_input.size(dim=0)-self.__n_val-1)
        #     val_seq_len = val_input.size(dim=1)
        #
        #     # Epochs
        #     for epoch in range(0, self._n_epochs):
        #         logger.debug("Epoch: " + str(epoch + 1))
        #
        #         loss_val_batch = torch.empty(val_batch_size)
        #         loss_train_batch = torch.empty(train_batch_size)
        #
        #         loss_train_state = torch.empty((train_batch_size, self._estimator.get_state_dim()))
        #         loss_val_state = torch.empty((val_batch_size, self._estimator.get_state_dim()))
        #
        #         # Validation
        #         logger.debug("Validate estimator...")
        #         self._estimator.eval()
        #
        #         for batch_nr in range(0, val_batch_size):
        #
        #             # Set initial state
        #             self._estimator.set_initial_conditions(val_target[val_starting_batch + batch_nr, :, 0])
        #
        #             # Initialize hidden GRU state
        #             self._estimator.initialize_hidden_state()
        #
        #             y_validation = val_input[val_starting_batch + batch_nr, :, :]
        #             estimated_state = torch.empty(val_seq_len, self._estimator.get_state_dim()).to(self._dev,
        #                                                                                            non_blocking=True)
        #
        #             for time_step in range(0, val_seq_len):
        #                 estimated_state[time_step, :] = self._estimator(y_validation[time_step, :])
        #
        #             # Compute validation loss
        #             loss_val = self._loss_function(estimated_state, val_target[batch_nr, 1:, :])
        #             loss_val_state[batch_nr, :] = torch.mean(loss_val, dim=0)
        #             loss_val_batch[batch_nr] = torch.mean(loss_val[:, self._loss_start_state: self._loss_end_state])
        #
        #             # Early stopping if tensor is nan
        #             if torch.isnan(loss_val_batch[batch_nr]):
        #                 break
        #
        #         # Average
        #         self.__loss_val_linear_epoch[epoch] = torch.mean(loss_val_batch)
        #         self.__loss_val_db_epoch[epoch] = 10 * torch.log10(self.__loss_val_linear_epoch[epoch])
        #         self.__loss_val_state_epoch[epoch] = torch.mean(loss_val_state, dim=0)
        #
        #         logger.info(f'Validation loss: {self.__loss_val_db_epoch[epoch]:0.4f} [dB]')
        #
        #         if self.__loss_val_db_epoch[epoch] < self.__optimal_loss_db:
        #             logger.info(
        #                 f'Loss improvement: {self.__optimal_loss_db - self.__loss_val_db_epoch[epoch]:0.4f} [dB]')
        #             self.__optimal_loss_db = self.__loss_val_db_epoch[epoch]
        #             self.__optimal_epoch = epoch + 1
        #
        #             self.save_estimator()
        #
        #         logger.info(f'Best loss so far: {self.__optimal_loss_db:0.4f} [dB] in epoch: {self.__optimal_epoch}')
        #
        #         # Training
        #         logger.debug("Train estimator...")
        #         self._estimator.train()
        #
        #         batch_optimizing_loss_sum = 0
        #
        #         # Initialize hidden GRU state
        #         self._estimator.initialize_hidden_state()
        #
        #         for batch_nr in range(0, train_batch_size):
        #
        #             estimated_state = torch.empty(train_seq_len, self._estimator.get_state_dim()).to(self._dev,
        #                                                                                              non_blocking=True)
        #
        #             actual_batch_nr = random.randint(0, self._train_size - 1)
        #
        #             # Set initial condition
        #             if random_init and init_var is not None:
        #                 random_initial_condition = train_target[batch_nr, :, 0] + torch.normal(mean=0, std=init_var,
        #                                                                                        size=self._estimator.m)
        #                 if random_initial_condition[2] < 0:
        #                     random_initial_condition[2] = torch.abs(random_initial_condition[2])
        #                 self._estimator.InitSequence(random_initial_condition, train_seq_len)
        #             self._estimator.set_initial_conditions(self._initial_state)
        #
        #             y_training = train_input[actual_batch_nr, :, :]
        #
        #             for time_step in range(0, train_seq_len):
        #                 estimated_state[time_step, :] = self._estimator(y_training[time_step, :])
        #
        #             # Compute Training Loss
        #             loss_train = self._loss_function(estimated_state,
        #                                              train_target[actual_batch_nr, 1:train_seq_len + 1, :])
        #             loss_train_state[batch_nr, :] = torch.mean(loss_train, dim=0)
        #             loss_train_batch[batch_nr] = torch.mean(loss_train[:, self._loss_start_state: self._loss_end_state])
        #
        #             if train_batch_size < 10 or (batch_nr % math.ceil(train_batch_size / 10)) == 0:
        #                 logger.debug(
        #                     f'[{batch_nr}|{train_batch_size}]: Training batch loss: {10 * math.log10(loss_train_batch[batch_nr]):0.4f} [dB]')
        #
        #             batch_optimizing_loss_sum = batch_optimizing_loss_sum + torch.mean(
        #                 loss_train[:, self._loss_start_state: self._loss_end_state])
        #
        #         # Average
        #         self.__loss_train_linear_epoch[epoch] = np.mean(loss_train_batch.numpy(force=True))
        #         self.__loss_train_db_epoch[epoch] = 10 * np.log10(self.__loss_train_linear_epoch[epoch])
        #         self.__loss_train_state_epoch[epoch] = np.mean(loss_train_batch.numpy(force=True), axis=0)
        #
        #         # Before the backward pass, use the optimizer object to zero all of the
        #         # gradients for the variables it will update (which are the learnable
        #         # weights of the model). This is because by default, gradients are
        #         # accumulated in buffers( i.e, not overwritten) whenever .backward()
        #         # is called. Checkout docs of torch.autograd.backward for more details.
        #         self._optimizer.zero_grad()
        #
        #         if not torch.isnan(self.__loss_train_linear_epoch[epoch]) and not torch.isinf(
        #                 self.__loss_train_linear_epoch[epoch]):
        #             if train_seq_len + add_increase_timestep <= max_train_seq_len:
        #                 train_seq_len += add_increase_timestep
        #                 logger.debug("Increases train sequence length to: " + str(train_seq_len))
        #             else:
        #                 train_seq_len = max_train_seq_len
        #
        #             # Backward pass: compute gradient of the loss with respect to model parameters
        #             batch_optimizing_loss_mean = batch_optimizing_loss_sum / train_batch_size
        #             batch_optimizing_loss_mean.backward()
        #
        #             # Calling the step function on an Optimizer makes an update to its parameters
        #             self._optimizer.step()
        #         else:
        #             if train_seq_len / mul_decrease_timestep > 1:
        #                 train_seq_len = int(train_seq_len / mul_decrease_timestep)
        #             else:
        #                 train_seq_len = 1
        #             logger.debug("Decreased train sequence length to: " + str(train_seq_len))
        #
        #         # Training summary
        #         logger.info(f'Training loss: {self.__loss_train_db_epoch[epoch]:0.4f} [dB]')
        #
        #         # Log difference w.r.t. to the last epoch
        #         # if epoch > 1:
        #         #     d_train = self.__loss_train_db_epoch[epoch] - self.__loss_train_db_epoch[epoch - 1]
        #         #     d_val = self.__loss_val_db_epoch[epoch] - self.__loss_val_db_epoch[epoch - 1]
        #         #     logger.debug(f'Difference loss training: {d_train:0.4f} [dB]')
        #         #     logger.debug(f'Difference loss validation: {d_val:0.4f} [dB]')
        #
        #     logger.info(f'Train execution time: {(time.time() - start_time) / 60:0.1f} [m]')
        #
        #     self._estimator = torch.load(self._model_complete_name)

    def test_best_param_for_estimator(self,
                                      test_input: Tensor,
                                      test_target: Tensor,
                                      process_variances: list[float],
                                      test_size: Optional[int] = None,
                                      random_variance: Optional[list] = None):
        logger.debug("Test best fitting process variance for estimator...")

        start_time = time.time()

        if isinstance(self._estimator, Estimator):
            for i, process_var in enumerate(process_variances):
                self._estimator.set_covariance_matrices(process_var=process_var)
                self.test_estimator(test_input, test_target, test_size=test_size, random_variance=random_variance)

                if self.__loss_test_db_mean < self.__optimal_loss_db:
                    logger.debug(f'Loss improvement: {self.__optimal_loss_db - self.__loss_test_db_mean:0.4f} [dB]')
                    self.save_estimator()
                    self.__optimal_loss_db = self.__loss_test_db_mean
                    self.__optimal_epoch = i

                logger.debug(
                    f'Best loss so far: {self.__optimal_loss_db:0.4f} [dB] with process var: {process_variances[self.__optimal_epoch]}')

            self._estimator = torch.load(self._model_complete_name)

        logger.info(f'Test params execution time: {time.time() - start_time:0.1f} [s]')

    @torch.no_grad()
    def test_estimator(self, test_input: Tensor,
                       test_target: Tensor,
                       test_size: Optional[int] = None,
                       random_variance: Optional[list] = None):
        logger.debug('Test estimator...')

        if test_size is not None:
            test_traj_len = test_input.size(dim=1)
            test_size = test_input.size(dim=0) * test_input.size(dim=1)
            test_batch_size = math.ceil(test_size / test_traj_len)

            if test_batch_size > test_input.size(dim=0):
                logger.warning("Desired test size is too big, will continue with using total test set")
                test_batch_size = test_input.size(dim=0)
        else:
            test_batch_size = test_input.size(dim=0)

        start_time = time.time()

        torch.manual_seed(42)

        test_input = test_input.to(self._dev)
        test_target = test_target.to(self._dev)

        self.__loss_test_state_batch = np.empty((test_batch_size, test_target.size(dim=2)))

        if isinstance(self._estimator, KalmanNetNN):
            test_input = torch.permute(test_input, (0, 2, 1))
            test_target = torch.permute(test_target, (0, 2, 1))

            test_seq_len = test_input.size(dim=2)

            for j in range(0, test_batch_size):
                estimated_state_test = torch.empty((self._estimator.m, test_seq_len))

                # Set initial state
                initial_state, __ = self.__get_initial_condition(test_target[j, :, 0],
                                                                 random_variance=random_variance)

                self._estimator.InitSequence(initial_state, test_seq_len)
                self._estimator.init_hidden()

                for time_step in range(0, test_seq_len):
                    estimated_state_test[:, time_step] = self._estimator(test_input[j, :, time_step])

                # Calculate loss
                loss = self._loss_function(estimated_state_test, test_target[j, :, 1:])
                self.__loss_test_state_batch[j, :] = np.mean(loss.numpy(force=True), axis=1)

        # elif isinstance(self._estimator, ExtendedKalmanFilter):
        #
        #     for j in range(0, n_test):
        #         self._estimator.set_initial_conditions(test_target[j, 0, :], self._initial_covariance)
        #         estimated_state = self._estimator.estimate_state_from_observations(test_input[j, :, :])
        #
        #         # Calculate loss
        #         loss = self._loss_function(estimated_state[1:, :], test_target[j, 1:, :])
        #         self.__loss_test_state_batch[j, :] = np.mean(loss.numpy(force=True), axis=0)
        #
        #     logger.info(f'Test execution time: {time.time() - start_time:0.1f} [s]')

        elif isinstance(self._estimator, Estimator):

            for j in range(0, test_batch_size):
                # Set initial state
                initial_state, initial_covariance = self.__get_initial_condition(test_target[j, 0, :],
                                                                                 random_variance=random_variance)
                self._estimator.set_initial_conditions(initial_state, initial_covariance)
                estimated_state = self._estimator.estimate_state_from_observations(test_input[j, :, :])

                # Calculate loss
                loss = self._loss_function(estimated_state[1:, :], test_target[j, 1:, :])
                self.__loss_test_state_batch[j, :] = np.mean(loss.numpy(force=True), axis=0)

        else:
            raise ValueError("The estimator has to be a subclass of base class Estimator")

        logger.info(f'Test execution time: {time.time() - start_time:0.1f} [s]')

        # Average
        self.__loss_test_state = np.mean(self.__loss_test_state_batch, axis=0)
        loss_test_mean = np.mean(self.__loss_test_state[self._loss_start_state: self._loss_end_state])
        self.__loss_test_db_mean = 10 * np.log10(loss_test_mean)
        loss_test_batch = np.mean(self.__loss_test_state_batch[:, self._loss_start_state: self._loss_end_state], axis=1)
        self.__loss_test_std = np.std(loss_test_batch)
        self.__loss_test_db_std = 10 * np.log10(self.__loss_test_std + loss_test_mean) - self.__loss_test_db_mean

        torch.seed()

        # Test summary
        logger.info(
            f'{self._model_name}-{self._loss_name} test: {self.get_mean_loss_test()[0]} [db]')
        logger.info(
            f'{self._model_name}-{self._loss_name} test state: {self.get_state_loss_test()[0][0].tolist()} [db]')

    @torch.no_grad()
    def get_estimates(self, observations: Tensor, initial_state: Tensor,
                      random_variance: Optional[list] = None) -> Tensor:

        logger.debug('Calculate estimates...')

        start_time = time.time()

        test_input = observations.to(self._dev)
        initial_state = initial_state.to(self._dev)

        # Set initial state
        new_initial_state, initial_covariance = self.__get_initial_condition(initial_state,
                                                                             random_variance=random_variance)

        if isinstance(self._estimator, Estimator):

            self._estimator.set_initial_conditions(new_initial_state, initial_covariance)

            estimated_state = self._estimator.estimate_state_from_observations(test_input)

        elif isinstance(self._estimator, KalmanNetNN):
            seq_len = test_input.size(dim=0)
            test_input = test_input.permute((1, 0))

            self._estimator.InitSequence(new_initial_state, seq_len)
            self._estimator.init_hidden()

            estimated_state = torch.empty(seq_len + 1, self._estimator.m)
            estimated_state[0, :] = new_initial_state

            for time_step in range(seq_len):
                estimated_state[time_step + 1, :] = self._estimator(test_input[:, time_step])

        else:
            raise ValueError("The estimator has to be a subclass of base class Estimator")

        logger.info(f'Calculating estimates execution time: {time.time() - start_time:0.1f} [s]')

        return estimated_state.cpu()

    @torch.no_grad()
    def get_estimates_complete(self, observations: Tensor, initial_state: Tensor,
                               random_variance: Optional[list] = None) -> Tensor:

        logger.debug('Calculate estimates...')

        start_time = time.time()

        test_input = torch.flatten(observations, start_dim=0, end_dim=1).to(self._dev)

        # Set initial state
        new_initial_state, initial_covariance = self.__get_initial_condition(initial_state,
                                                                             random_variance=random_variance)
        if isinstance(self._estimator, Estimator):

            self._estimator.set_initial_conditions(new_initial_state, initial_covariance)

            estimated_state = self._estimator.estimate_state_from_observations(test_input)

        elif isinstance(self._estimator, KalmanNetNN):
            seq_len = test_input.size(dim=0)
            test_input = test_input.permute((1, 0))

            self._estimator.InitSequence(initial_state, seq_len)

            self._estimator.init_hidden()

            estimated_state = torch.empty(seq_len, self._estimator.m)

            for time_step in range(seq_len):
                estimated_state[time_step, :] = self._estimator(test_input[:, time_step])

        else:
            raise ValueError("The estimator has to be a subclass of base class Estimator")

        logger.info(f'Calculating estimates execution time: {time.time() - start_time:0.1f} [s]')

        return estimated_state.cpu()

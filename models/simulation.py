import logging
from typing import Optional

import torch
from torch import Tensor

from models.params import DefaultSimulationParameters
from models.models import Model

logger = logging.getLogger(__name__)


class DataGeneration:
    _complete_data_name: str

    def __init__(self, path_name: str, file_name: str, model: Model):
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.no_grad()

        self._data_path = path_name
        self._data_name = file_name
        self.__build_complete_name()

        self._model = model

        self._n_train = DefaultSimulationParameters.n_train
        self._n_val = DefaultSimulationParameters.n_validation
        self._n_test = DefaultSimulationParameters.n_test

    def __build_complete_name(self):
        self._complete_data_name = self._data_path + '/' + self._data_name + '.pt'

    def set_data_generation_parameters(self, n_train:int, n_val:int, n_test:int):
        self._n_train = n_train
        self._n_val = n_val
        self._n_test = n_test

    def set_path(self, path_name: str, filename: str):
        self._complete_data_name = path_name + '/' + filename + '.pt'

    def set_system_model(self, system_model: Model):
        self._model = system_model

    def save_and_generate(self):
        logger.debug("Generate single sequence...")
        [t, x, y] = self._model.generate_discrete_noise()
        torch.save([t, x, y], self._complete_data_name)
        logger.info('Single sequence saved in: ' + self._complete_data_name)

    def save_and_generate_batch(self, batch_size, random_init=False, variance=0.1 ** 2):
        logger.debug("Generate batch sequence...")
        [t, x, y] = self._model.generate_batch_discrete_noise(batch_size,
                                                              random_init=random_init,
                                                              variance=variance)
        torch.save([t, x, y], self._complete_data_name)
        logger.info("Batch sequence saved in: " + self._complete_data_name)

    def generate_and_save_train_val_test_data(self,
                                              random_init=False,
                                              variance=0.5,
                                              traj_lengths=[100, 100, 100],
                                              sampling_time=1e-1):

        logger.debug('Generate train sequence...')
        self._model.set_simulation_params(sampling_time, traj_lengths[0])
        train = self._model.generate_batch_discrete_noise(self._n_train, random_init=random_init,
                                                          variance=variance)

        logger.debug('Generate val sequence...')
        self._model.set_simulation_params(sampling_time, traj_lengths[1])
        val = self._model.generate_batch_discrete_noise(self._n_val, random_init=random_init,
                                                        variance=variance)

        logger.debug('Generate test sequence...')
        self._model.set_simulation_params(sampling_time, traj_lengths[2])
        test = self._model.generate_batch_discrete_noise(self._n_test, random_init=random_init,
                                                         variance=variance)

        torch.save([train, val, test], self._complete_data_name)
        logger.info('Train, val and test sequences saved in: ' + self._complete_data_name)

    def generate_observations_from_state_and_save(self, train, val, test):
        train_batch_size = train[2].size(dim=1)
        train[2] = self._model.generate_observation_from_state(train[1],train[2], train_batch_size)

        val_batch_size = val[2].size(dim=1)
        val[2] = self._model.generate_observation_from_state(val[1], val[2], val_batch_size)

        test_batch_size = test[2].size(dim=1)
        test[2] = self._model.generate_observation_from_state(test[1],test[2], test_batch_size)

        torch.save([tuple(train), tuple(val), tuple(test)], self._complete_data_name)
        logger.info('Train, val and test sequences saved in: ' + self._complete_data_name)

    def replace_test_and_save(self, random_init=False,
                                              variance=0.5,
                                              sample_rate=1e-1):
        logger.debug('Replace test data...')
        train, val, test = self.load_data()
        self._model.set_simulation_params(sample_rate, test[2].size(dim=2))
        test = self._model.generate_batch_discrete_noise(self._n_test, random_init, variance)

        torch.save([train, val, test], self._complete_data_name)
        logger.info('Train, val and test sequences saved in: ' + self._complete_data_name)

    def change_seq_lengths_and_save(self, seq_lengths: list[int, int, int], new_data_name: str, data_path: Optional[str] = None):
        logger.debug('Change sequence lengths...')
        train, val, test = self.load_data()

        state_sets = [train[1], val[1], test[1]]
        observation_sets = [train[2], val[2], test[2]]

        data_sets = []

        for i, (state_set, observation_set) in enumerate(zip(state_sets, observation_sets)):
            total_points = state_set.shape[0]*(state_set.shape[1]-1)
            if total_points % seq_lengths[i] == 0:

                new_batch_size = int(total_points/seq_lengths[i])

                # Reshaping states
                starting_conditions = state_set[0, 1, :]
                reshaped_state_set = torch.zeros((new_batch_size, seq_lengths[i]+1, state_set.shape[-1]))
                reshaped_state_set[:, 1:, :] = torch.reshape(state_set[:, 1:, :], (new_batch_size, seq_lengths[i], state_set.shape[-1]))
                reshaped_state_set[0, 0, :] = starting_conditions

                for batch_nbr in range(1, reshaped_state_set.shape[0]):
                    reshaped_state_set[batch_nbr, 0, :] = reshaped_state_set[batch_nbr-1, -1, :]

                # Reshaping observations
                reshaped_observation_set = torch.reshape(observation_set, (new_batch_size, seq_lengths[i], observation_set.shape[-1]))

                self._model.set_simulation_params(n_timesteps=seq_lengths[i])
                data_sets.append((self._model.get_timevector_batch(), reshaped_state_set, reshaped_observation_set))

            else:
                raise ValueError("The sequence length can't be changed, because the new sequence length does not divide the number of samples")

        if data_path is not None:
            self._data_path = data_path
        self._data_name = new_data_name
        self.__build_complete_name()

        torch.save(data_sets, self._complete_data_name)
        logger.info('Reshaped train, val and test sequences saved in: ' + self._complete_data_name)

    def load_data(self):
        data = torch.load(self._complete_data_name)
        logger.debug('Data loaded from: ' + self._complete_data_name)
        return data

    def load_training_data_gpu(self) -> [list, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        train: list
        val: list
        test: list

        if torch.cuda.is_available():
            [train, val, test] = torch.load(self._complete_data_name, map_location='cuda')
            logger.debug("Train, val and test sequences loaded to GPU from: " + self._complete_data_name)
        else:
            logger.warning("GPU is not available, loading data on CPU instead")
            [train, val, test] = torch.load(self._complete_data_name, map_location='cpu')
            logger.debug("Train, val and test sequences loaded to CPU from: " + self._complete_data_name)

        train_input, train_target = train[2], train[1]
        val_input, val_target = val[2], val[1]
        test_input, test_target = test[2], test[1]
        timevectors = [train[0], val[0], test[0]]
        return timevectors, train_input, train_target, val_input, val_target, test_input, test_target

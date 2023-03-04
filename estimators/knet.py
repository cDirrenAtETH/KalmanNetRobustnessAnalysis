import custom_logging
import math
from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as func

# Module logger
logger = custom_logging.getLogger(__name__)


# Load directly via this method (not torch.load)
def load_estimator(path: str):
    estimator = torch.load(path)
    estimator.initialize()
    logger.debug("Estimator loaded: " + path)
    return estimator


class KalmanNetSimple(nn.Module):
    # x_(t|t-1)
    __state_prior: Tensor
    # x_(t-1|t-2)
    __state_prev_prior: Tensor
    # x_(t|t)
    __state_posterior: Tensor

    _delta_t: float

    _dev: torch.device

    def __init__(self, state_evolution_continuous: Callable[[Tensor], Tensor], measurement_matrix: Tensor,
                 delta_t=1e-1, n_taylor_coefficients=1, is_nonlinear_evolution=False):
        super().__init__()

        self._set_device()

        self._delta_t = delta_t

        self._n_taylor_coefficients = n_taylor_coefficients

        self._is_nonlinear_evolution = is_nonlinear_evolution

        self._state_dim = measurement_matrix.size(dim=1)
        self._observation_dim = measurement_matrix.size(dim=0)

        # Model
        self._state_evolution_continuous = state_evolution_continuous
        if not self._is_nonlinear_evolution:
            self._const_discrete_matrix = self.__get_matrix_exp_approx(
                state_evolution_continuous(torch.eye(self._state_dim)))
        self._measurement_matrix = measurement_matrix.to(self._dev)

        # Input features: delta x_(t-1), y_t
        self._input_dim = self._state_dim + self._observation_dim

        # Output dimension
        self._output_dim = self._state_dim * self._observation_dim

        # Number of neurons in the 1st linear layer
        self._hidden_dim_1 = (self._state_dim + self._observation_dim) * 10 * 8

        # Number of neurons in the 2nd linear layer
        self._hidden_dim_2 = (self._state_dim + self._observation_dim) * 1 * 4

        # Input layer
        self._linear_layer_1 = nn.Linear(self._input_dim, self._hidden_dim_1)
        self._activation_1 = nn.ReLU()

        # GRU
        self._input_dim_gru = self._hidden_dim_1
        self._hidden_dim_gru = (self._state_dim ** 2 + self._observation_dim ** 2) * 10
        self._n_layers_gru = 1
        self._batch_size_gru = 1
        self._seq_len_input_gru = 1
        self._gru = nn.GRU(self._input_dim_gru, self._hidden_dim_gru, self._n_layers_gru)

        # Hidden layer
        self._linear_layer_2 = nn.Linear(self._hidden_dim_gru, self._hidden_dim_2)
        self._activation_2 = nn.ReLU()

        # Output layer
        self._linear_layer_3 = nn.Linear(self._hidden_dim_2, self._output_dim)

        # Initialize a Tensor for Hidden State
        self._hidden_state_gru = torch.randn(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_gru).to(
            self._dev, non_blocking=True)

    def __get_matrix_exp_approx(self, matrix: Tensor) -> Tensor:
        # Taylor expansion of exp(A*delta_t)
        discretized_matrix = torch.eye(matrix.size(dim=0)).to(self._dev)
        for i in range(1, self._n_taylor_coefficients + 1):
            discretized_matrix_i = torch.matrix_power(torch.mul(matrix, self._delta_t), i) / math.factorial(i)
            discretized_matrix = torch.add(discretized_matrix, discretized_matrix_i)

        return discretized_matrix

    def _set_device(self):
        if torch.cuda.is_available():
            self._dev = torch.device("cuda:0")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            logger.debug('Running on the GPU')
        else:
            self._dev = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
            logger.debug('Running on the CPU')

    def initialize_after_loading(self):
        self._set_device()

    def __f_discrete(self, x: Tensor) -> Tensor:
        matrix_a = self._state_evolution_continuous(x)
        discretized_f = self.__get_matrix_exp_approx(matrix_a)
        return torch.matmul(discretized_f, x)

    def _prediction_step(self):
        self.__state_prev_prior = self.__state_prior
        if self._is_nonlinear_evolution:
            self.__state_prior = self.__f_discrete(self.__state_posterior)
        else:
            self.__state_prior = torch.matmul(self._const_discrete_matrix, self.__state_posterior)
        self.__observation_prior = torch.matmul(self._measurement_matrix, self.__state_prior)

    def _update_step(self, current_observation: Tensor) -> Tensor:
        # Feature 2: yt - y_t+1|t
        innovation_difference = current_observation - self.__observation_prior
        # innovation_difference_squeezed = torch.squeeze(innovation_difference)
        innovation_difference_norm = func.normalize(innovation_difference, p=2, dim=0, eps=1e-12, out=None)

        # Feature 4: x_t|t - x_t|t-1
        forward_update_difference = self.__state_posterior - self.__state_prev_prior
        # forward_update_difference_squeezed = torch.squeeze(forward_update_difference)
        forward_update_difference_norm = func.normalize(forward_update_difference, p=2, dim=0, eps=1e-12,
                                                        out=None)

        input_tensor = torch.cat([innovation_difference_norm, forward_update_difference_norm], dim=0)

        kalman_gain = self._compute_kalman_gain(input_tensor)

        self.__kalman_gain = torch.reshape(kalman_gain, (self._state_dim, self._observation_dim))

        self.__state_posterior = self.__state_prior + torch.matmul(self.__kalman_gain, innovation_difference)
        return self.__state_posterior

    def _compute_kalman_gain(self, features: Tensor) -> Tensor:
        # Input layer
        layer_1_out = self._linear_layer_1(features)
        layer_1_out_act = self._activation_1(layer_1_out)

        # GRU
        input_gru = torch.empty(self._seq_len_input_gru, self._batch_size_gru, self._input_dim_gru).to(self._dev,
                                                                                                       non_blocking=True)
        input_gru[0, 0, :] = layer_1_out_act
        output_gru, self._hidden_state_gru = self._gru(input_gru, self._hidden_state_gru)
        output_gru_reshape = torch.reshape(output_gru, (1, self._hidden_dim_gru))

        # Hidden layer
        layer_2_out = self._linear_layer_2(output_gru_reshape)
        layer_2_out_act = self._activation_2(layer_2_out)

        # Output layer
        output = self._linear_layer_3(layer_2_out_act)

        return output

    def initialize_hidden_gru_state(self):
        self._hidden_state_gru = torch.randn(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_gru).to(
            self._dev, non_blocking=True)

    def forward(self, feature: Tensor) -> Tensor:
        feature = feature.to(self._dev, non_blocking=True)
        self._prediction_step()
        estimate = self._update_step(feature)
        return estimate.squeeze()

    def set_initial_conditions(self, initial_state: Tensor):
        self.__state_prior = initial_state
        self.__state_prev_prior = self.__state_prior
        self.__state_posterior = self.__state_prior

    def set_hidden_dim_of_linear_layers(self, hidden_dim_1: int, hidden_dim_2: int):
        self._hidden_dim_1 = hidden_dim_1
        self._hidden_dim_2 = hidden_dim_2

    def get_state_dim(self) -> int:
        return self._state_dim

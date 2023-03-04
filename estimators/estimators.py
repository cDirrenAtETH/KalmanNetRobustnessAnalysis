import math
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch import Tensor
from torch.autograd.functional import jacobian
import torch.nn.functional as func

# Module logger
logger = logging.getLogger(__name__)


# Do not load via torch.load
def load_estimator(path: str, dev='cpu'):
    estimator = torch.load(path)
    estimator.initialize(dev)
    logger.debug("Estimator loaded: " + path)
    return estimator


# Base class
def get_quadratic_form(x: Tensor, matrix_a: Tensor) -> Tensor:
    return torch.matmul(matrix_a, torch.matmul(x, matrix_a.T))


class Estimator(ABC):
    _dev: torch.device

    def __init__(self, state_dim: int, observation_dim: int, measurement_matrix: Tensor, delta_t: float,
                 max_taylor_coefficient: int, device: str,
                 additional_arguments: dict[str, Any]):

        self._set_device(device)

        self._state_dim: int = state_dim
        self._observation_dim: int = observation_dim

        self._measurement_matrix: Tensor = measurement_matrix.to(self._dev)

        # Dummy tensors
        self._process_covariance: Optional[Tensor] = None
        self._measurement_covariance: Optional[Tensor] = None

        # Number of Taylor coefficients' for discretizing the system
        self._max_taylor_coefficient: int = max_taylor_coefficient
        self._delta_t: float = delta_t

    def _update_device(self, dev: torch.device):
        if dev.type == self._dev.type:
            logger.debug("Device not changed, already using " + dev.type.swapcase())
        else:
            self._dev = dev
            logger.info("Device changed to " + dev.type.swapcase())

            # Change device of members
            self._measurement_matrix = self._measurement_matrix.to(self._dev)
            self._process_covariance = self._process_covariance.to(self._dev)
            self._measurement_covariance = self._measurement_covariance.to(self._dev)

    def _get_empty_state_trajectory(self, n_timesteps: int) -> Tensor:
        return torch.empty(size=(n_timesteps + 1, self._state_dim))

    def _get_empty_state_covariances(self, n_timesteps: int) -> Tensor:
        return torch.empty(size=(n_timesteps + 1, self._state_dim, self._state_dim))

    def _get_matrix_exp_approx(self, matrix: Tensor) -> Tensor:
        # Taylor expansion of exp(A*delta_t)
        discretized_matrix = torch.eye(matrix.size(dim=0))
        for i in range(1, self._max_taylor_coefficient + 1):
            discretized_matrix_i = torch.matrix_power(torch.mul(matrix, self._delta_t), i) / math.factorial(i)
            discretized_matrix = torch.add(discretized_matrix, discretized_matrix_i)

        return discretized_matrix

    def _set_device(self, dev: str):
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

    def get_state_dim(self) -> int:
        return self._state_dim

    def get_observation_dim(self) -> int:
        return self._observation_dim

    def set_discretization_params(self, sampling_time: int, max_taylor_coefficient: int):
        self._delta_t = sampling_time
        self._max_taylor_coefficient = max_taylor_coefficient

    def set_covariance_matrices(self, process_var: Optional[float] = None, measurement_var: Optional[float] = None):
        if process_var is not None:
            self._process_covariance = torch.eye(self._state_dim)*process_var
            logger.debug("Updated process covariance matrix")

        if measurement_var is not None:
            self._measurement_covariance = torch.eye(self._observation_dim)*measurement_var
            logger.debug("Updated measurement covariance matrix")

    def initialize(self, dev: torch.device):
        self._update_device(dev)

    # Abstract methods
    # @abstractmethod
    # def __reset_all_current_params(self):
    #     pass

    @abstractmethod
    def _prediction_step(self):
        pass

    @abstractmethod
    def _update_step(self, current_observation: Tensor):
        pass

    @abstractmethod
    def set_initial_conditions(self, *args):
        pass

    @abstractmethod
    def estimate_state_from_observations(self, observations: Tensor) -> Tensor:
        pass


# Dummy estimator
class IdentityEstimator(Estimator):
    def __init__(self, state_dim: int, observation_dim: int, delta_t=1e-1, device='cpu', **kwargs):
        Estimator.__init__(self, state_dim, observation_dim, torch.Tensor([]), delta_t, 0, device, kwargs)

        self._initial_state = None

    # Protected methods
    def _prediction_step(self):
        pass

    def _update_step(self, current_observation: Tensor):
        pass

    # Public methods
    def set_initial_conditions(self, *args):
        self._initial_state = args[0]

    def estimate_state_from_observations(self, observations: Tensor) -> Tensor:
        n_timesteps = observations.size(dim=0)
        estimated_states = self._get_empty_state_trajectory(n_timesteps)

        estimated_states[0, :] = self._initial_state
        estimated_states[1:, :] = observations
        return estimated_states


class KalmanFilter(Estimator):

    def __init__(self, continuous_evolution_matrix: Tensor, measurement_matrix: Tensor,
                 process_covariance: Tensor,
                 measurement_covariance: Tensor, delta_t=1e-1, max_taylor_coefficient=1, device='cpu', **kwargs):
        Estimator.__init__(self, measurement_matrix.size(dim=1), measurement_matrix.size(dim=0),
                           measurement_matrix, delta_t, max_taylor_coefficient, device, kwargs)

        self._continuous_evolution_matrix = continuous_evolution_matrix

        self._process_covariance = process_covariance.to(self._dev)
        self._measurement_covariance = measurement_covariance.to(self._dev)

        # Current testing variables
        self.__linear_state_evolution_discrete = None
        self.__prev_state_posterior = None
        self.__state_prior = None
        self.__state_posterior = None
        self.__prev_covariance_posterior = None
        self.__covariance_prior = None
        self.__covariance_posterior = None

    def __reset_all_current_params(self):
        self.__linear_state_evolution_discrete = None
        self.__prev_state_posterior = None
        self.__state_prior = None
        self.__state_posterior = None
        self.__prev_covariance_posterior = None
        self.__covariance_prior = None
        self.__covariance_posterior = None

    # Protected methods
    def _prediction_step(self):
        # x_hat(k) = F*x_hat(k-1)
        self.__state_prior = torch.matmul(self.__linear_state_evolution_discrete, self.__prev_state_posterior)
        # P_hat(k) = F*P_hat(k-1)*F^T+Q
        self.__covariance_prior = get_quadratic_form(self.__prev_covariance_posterior,
                                                     self.__linear_state_evolution_discrete) + self._process_covariance

    def _update_step(self, current_observation: Tensor):
        innovation = (current_observation - torch.matmul(self._measurement_matrix, self.__state_prior))
        innovation_covariance = get_quadratic_form(self.__covariance_prior,
                                                   self._measurement_matrix) + self._measurement_covariance
        kalman_gain = torch.matmul(self.__covariance_prior,
                                   torch.matmul(self._measurement_matrix.T, torch.inverse(innovation_covariance)))

        self.__state_posterior = self.__state_prior + torch.matmul(kalman_gain, innovation)
        self.__covariance_posterior = torch.matmul(
            torch.eye(self._state_dim) - torch.matmul(kalman_gain, self._measurement_matrix), self.__covariance_prior)

    # Public methods
    def set_initial_conditions(self, *args):
        if len(args) == 2:
            self.__prev_state_posterior = args[0].to(self._dev)
            self.__state_prior = self.__prev_state_posterior
            self.__prev_covariance_posterior = args[1].to(self._dev)
            self.__covariance_prior = self.__prev_covariance_posterior

        else:
            logger.warning(
                f'{len(args)} initial condition(s) is (were) given, but only 2 are used, initialize with zero instead')
            self.__prev_state_posterior = torch.zeros(self._state_dim)
            self.__state_prior = self.__prev_state_posterior
            self.__prev_covariance_posterior = torch.zeros((self._state_dim, self._state_dim))
            self.__covariance_prior = self.__prev_covariance_posterior
            self.__state_posterior = self.__state_prior

    def estimate_state_from_observations(self, observations: Tensor) -> Tensor:
        # x.size=[n_timesteps+1, n_features]
        n_timesteps = observations.size(dim=0)

        self.__linear_state_evolution_discrete = self._get_matrix_exp_approx(
            self._continuous_evolution_matrix)

        estimated_states = self._get_empty_state_trajectory(n_timesteps)
        estimated_covariances = self._get_empty_state_covariances(n_timesteps)

        estimated_states[0, :] = self.__prev_state_posterior
        estimated_covariances[0, :, :] = self.__prev_covariance_posterior

        observations = observations.to(self._dev)

        for i in range(1, n_timesteps + 1):
            actual_observation = observations[i - 1, :]

            self._prediction_step()
            self._update_step(actual_observation)

            estimated_states[i, :] = self.__state_posterior
            estimated_covariances[i, :, :] = self.__covariance_posterior

            self.__prev_state_posterior = self.__state_posterior
            self.__prev_covariance_posterior = self.__covariance_posterior

        self.__reset_all_current_params()

        return estimated_states


class ExtendedKalmanFilter(Estimator):

    def __init__(self, nonlinear_state_evolution_continuous: Callable[[Tensor], Tensor],
                 measurement_matrix: Tensor, process_covariance: Tensor, measurement_covariance: Tensor,
                 delta_t=1e-1, max_taylor_coefficient=1, device='cpu', **kwargs):
        Estimator.__init__(self, measurement_matrix.size(dim=1), measurement_matrix.size(dim=0),
                           measurement_matrix, delta_t, max_taylor_coefficient, device, kwargs)

        # x_dot = A(x)*x = (nonlinear_state_evolution_continuous(x))*x
        self._nonlinear_state_evolution_continuous = nonlinear_state_evolution_continuous

        self._process_covariance = process_covariance.to(self._dev)
        self._measurement_covariance = measurement_covariance.to(self._dev)

        # Current testing variables
        self.__prev_state_posterior: Optional[Tensor] = None
        self.__state_prior: Optional[Tensor] = None
        self.__state_posterior: Optional[Tensor] = None
        self.__prev_covariance_posterior: Optional[Tensor] = None
        self.__covariance_prior: Optional[Tensor] = None
        self.__covariance_posterior: Optional[Tensor] = None

    def __reset_all_current_params(self):
        self.__prev_state_posterior = None
        self.__state_prior = None
        self.__state_posterior = None
        self.__prev_covariance_posterior = None
        self.__covariance_prior = None
        self.__covariance_posterior = None

    def __get_nonlinear_dynamics_at_x(self, x: Tensor) -> Tensor:
        return self._nonlinear_state_evolution_continuous(x)

    def __f_discrete(self, x: Tensor) -> Tensor:
        matrix_a = self.__get_nonlinear_dynamics_at_x(x)
        discretized_f = self._get_matrix_exp_approx(matrix_a)
        return torch.matmul(discretized_f, x)

    # Protected methods
    def _prediction_step(self):
        # x_hat(k) = f(x_hat(k-1))
        self.__state_prior = self.__f_discrete(self.__prev_state_posterior)

        # Linearize nonlinear discrete model
        linearized_matrix = jacobian(self.__f_discrete, self.__prev_state_posterior)

        # P_hat(k) = F*P_hat(k-1)*F^T+Q
        self.__covariance_prior = get_quadratic_form(self.__prev_covariance_posterior,
                                                     linearized_matrix) + self._process_covariance

    def _update_step(self, current_observation: Tensor):
        innovation = (current_observation - torch.matmul(self._measurement_matrix, self.__state_prior))
        innovation_covariance = get_quadratic_form(self.__covariance_prior,
                                                   self._measurement_matrix) + self._measurement_covariance
        kalman_gain = torch.matmul(self.__covariance_prior,
                                   torch.matmul(self._measurement_matrix.T, torch.inverse(innovation_covariance)))

        self.__state_posterior = self.__state_prior + torch.matmul(kalman_gain, innovation)
        self.__covariance_posterior = torch.matmul(
            torch.eye(self._state_dim) - torch.matmul(kalman_gain, self._measurement_matrix), self.__covariance_prior)

    def set_initial_conditions(self, *args):
        if len(args) == 2:
            self.__prev_state_posterior = args[0].to(self._dev)
            self.__state_prior = self.__prev_state_posterior
            self.__prev_covariance_posterior = args[1].to(self._dev)
            self.__covariance_prior = self.__prev_covariance_posterior

        else:
            logger.warning(
                f'{len(args)} initial conditions were given, but only 1 is used, initialize with zero instead')
            self.__prev_state_posterior = torch.zeros(self._state_dim)
            self.__state_prior = self.__prev_state_posterior
            self.__prev_covariance_posterior = torch.zeros((self._state_dim, self._state_dim))
            self.__covariance_prior = self.__prev_covariance_posterior

    def estimate_state_from_observations(self, observations: Tensor) -> Tensor:
        # x.size=[n_timesteps+1, n_features]
        n_timesteps = observations.size(dim=0)

        estimated_states = self._get_empty_state_trajectory(n_timesteps)
        estimated_covariances = self._get_empty_state_covariances(n_timesteps)

        estimated_states[0, :] = self.__prev_state_posterior
        estimated_covariances[0, :, :] = self.__prev_covariance_posterior

        observations = observations.to(self._dev)

        for i in range(1, n_timesteps + 1):
            actual_observation = observations[i - 1, :]

            self._prediction_step()
            self._update_step(actual_observation)

            estimated_states[i, :] = self.__state_posterior
            estimated_covariances[i, :, :] = self.__covariance_posterior

            self.__prev_state_posterior = self.__state_posterior
            self.__prev_covariance_posterior = self.__covariance_posterior

        return estimated_states


class KalmanNetSimple(nn.Module, Estimator):

    def __init__(self, state_evolution_continuous: Callable[[Tensor], Tensor], measurement_matrix: Tensor,
                 delta_t=1e-1, max_taylor_coefficient=1, is_nonlinear_evolution=False, device='cpu', **kwargs):

        nn.Module.__init__(self)
        Estimator.__init__(self, measurement_matrix.size(dim=1), measurement_matrix.size(dim=0),
                           measurement_matrix, delta_t, max_taylor_coefficient, device, kwargs)

        self._is_nonlinear_evolution = is_nonlinear_evolution

        # Model
        self._state_evolution_continuous = state_evolution_continuous
        if not self._is_nonlinear_evolution:
            self._const_discrete_matrix = self._get_matrix_exp_approx(
                state_evolution_continuous(torch.eye(self._state_dim)))

        # Input features: delta x_(t-1), y_t
        self._input_dim = self._state_dim + self._observation_dim

        # Output dimension
        self._output_dim = self._state_dim * self._observation_dim

        # Number of neurons in the 1st linear layer
        self._hidden_dim_1 = (self._state_dim + self._observation_dim) * 10 * 8

        # Number of neurons in the 2nd linear layer
        self._hidden_dim_2 = (self._state_dim + self._observation_dim) * 1 * 4

        # Sequence length and batch size
        self._batch_size_gru = 1
        self._seq_len_input_gru = 1

        # Build network
        self.__build_network()

        # Current testing variables
        self.__state_prev_posterior = None
        self.__state_prior = None
        self.__observation_prior = None
        self.__state_prior = None
        self.__state_prev_prior = None
        self.__state_posterior = None

        self.__hidden_state = None

    def __reset_all_current_params(self):
        self.__state_prev_posterior = None
        self.__state_prior = None
        self.__observation_prior = None
        self.__state_prior = None
        self.__state_prev_prior = None
        self.__state_posterior = None

        self.__hidden_state = None

    def __build_network(self):
        # Input layer
        self._linear_layer_1 = nn.Linear(self._input_dim, self._hidden_dim_1)
        self._activation_1 = nn.ReLU()

        # GRU
        self._input_dim_gru = self._hidden_dim_1
        self._hidden_dim_gru = (self._state_dim ** 2 + self._observation_dim ** 2) * 10
        self._gru = nn.GRU(self._input_dim_gru, self._hidden_dim_gru)
        self.__hidden_state = torch.randn(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_gru).to(
            self._dev, non_blocking=True)

        # Hidden layer
        self._linear_layer_2 = nn.Linear(self._hidden_dim_gru, self._hidden_dim_2)
        self._activation_2 = nn.ReLU()

        # Output layer
        self._linear_layer_3 = nn.Linear(self._hidden_dim_2, self._output_dim)

        self.initialize_hidden_state()

        logger.debug("Network built")

    def __f_discrete(self, x: Tensor) -> Tensor:
        matrix_a = self._state_evolution_continuous(x)
        discretized_f = self._get_matrix_exp_approx(matrix_a)
        return torch.matmul(discretized_f, x)

    def _prediction_step(self):
        self.__state_prev_prior = self.__state_prior
        if self._is_nonlinear_evolution:
            self.__state_prior = self.__f_discrete(self.__state_posterior)
        else:
            self.__state_prior = torch.matmul(self._const_discrete_matrix, self.__state_posterior)
        self.__observation_prior = torch.matmul(self._measurement_matrix, self.__state_prior)

    def _update_step(self, current_observation: Tensor):
        # Feature 2: y_t - y_t|t-1
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

    def _compute_kalman_gain(self, features: Tensor) -> Tensor:
        # Input layer
        layer_1_out = self._linear_layer_1(features)
        layer_1_out_act = self._activation_1(layer_1_out)

        # GRU
        input_gru = torch.empty(self._seq_len_input_gru, self._batch_size_gru, self._input_dim_gru).to(self._dev,
                                                                                                       non_blocking=True)
        input_gru[0, 0, :] = layer_1_out_act
        output_gru, self.__hidden_state = self._gru(input_gru, self.__hidden_state)
        output_gru_reshape = torch.reshape(output_gru, (1, self._hidden_dim_gru))

        # Hidden layer
        layer_2_out = self._linear_layer_2(output_gru_reshape)
        layer_2_out_act = self._activation_2(layer_2_out)

        # Output layer
        output = self._linear_layer_3(layer_2_out_act)

        return output

    def initialize_hidden_state(self):
        self.__hidden_state = torch.randn(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_gru).to(
            self._dev, non_blocking=True)

    def forward(self, feature: Tensor) -> Tensor:
        feature = feature.to(self._dev, non_blocking=True)
        self._prediction_step()
        self._update_step(feature)
        return self.__state_posterior

    def set_initial_conditions(self, *args):
        if len(args) == 1:
            self.__state_prior = args[0]
            self.__state_prev_prior = self.__state_prior
            self.__state_posterior = self.__state_prior
        else:
            logger.warning(
                f'{len(args)} initial conditions were given, but only 1 is used, initialize with zero instead')
            self.__state_prior = torch.zeros(self._state_dim)
            self.__state_prev_prior = self.__state_prior
            self.__state_posterior = self.__state_prior

    def set_hidden_dim_of_linear_layers(self, hidden_dim_1: int, hidden_dim_2: int):
        self._hidden_dim_1 = hidden_dim_1
        self._hidden_dim_2 = hidden_dim_2
        self.__build_network()

    @torch.no_grad()
    def estimate_state_from_observations(self, observations: Tensor) -> Tensor:
        # x.size=[n_timesteps+1, n_features]
        n_timesteps = observations.size(dim=0)

        estimated_states = self._get_empty_state_trajectory(n_timesteps)

        estimated_states[0, :] = self.__state_prev_posterior

        observations = observations.to(self._dev)

        self.eval()

        self.initialize_hidden_state()

        for i in range(1, n_timesteps + 1):
            actual_observation = observations[i - 1, :]
            estimated_states[i, :] = self.forward(actual_observation)

        return estimated_states


class KalmanNetSophisticated(nn.Module, Estimator):

    def __init__(self, state_evolution_continuous: Callable[[Tensor], Tensor], measurement_matrix: Tensor,
                 delta_t=1e-1, max_taylor_coefficient=1, is_nonlinear_evolution=False, device='cpu', **kwargs):
        nn.Module.__init__(self)
        Estimator.__init__(self, measurement_matrix.size(dim=1), measurement_matrix.size(dim=0),
                           measurement_matrix, delta_t, max_taylor_coefficient, device, kwargs)

        self._is_nonlinear_evolution = is_nonlinear_evolution

        # Model
        self._state_evolution_continuous = state_evolution_continuous
        if not self._is_nonlinear_evolution:
            self._const_discrete_matrix = self._get_matrix_exp_approx(
                state_evolution_continuous(torch.eye(self._state_dim)))

        # GRU input multiplier
        self._gru_input_mult = 5

        # GRU output multiplier
        self._gru_output_mult = 40

        # Sequence length and batch size
        self._batch_size_gru = 1
        self._seq_len_input_gru = 1

        # Current variables
        self.__state_prev_posterior = None
        self.__state_posterior = None
        self.__state_prev_prior = None
        self.__state_prior = None
        self.__observation_prior = None
        self.__observation_prev_posterior = None

        self.__hidden_state_q = None
        self.__hidden_state_sigma = None
        self.__hidden_state_s = None

        self.__kalman_gain = torch.zeros((self._state_dim, self._observation_dim))

        # Build network
        self.__build_network(torch.eye(self._state_dim), torch.eye(self._state_dim), torch.eye(self._observation_dim))

    def __reset_all_current_params(self):
        self.__state_prev_posterior = None
        self.__state_posterior = None
        self.__state_prev_prior = None
        self.__state_prior = None
        self.__observation_prior = None
        self.__observation_prev_posterior = None

        self.__kalman_gain = torch.zeros(())

    def __build_network(self, initial_q: Tensor, initial_sigma: Tensor, initial_s: Tensor):
        # Initial covariances (hidden states of GRU's)
        self._initial_q = initial_q.to(self._dev)
        self._initial_sigma = initial_sigma.to(self._dev)
        self._initial_s = initial_s.to(self._dev)

        # GRU to track Q
        self._input_dim_q = self._state_dim * self._gru_input_mult
        self._hidden_dim_q = self._state_dim ** 2
        self._gru_q = nn.GRU(self._input_dim_q, self._hidden_dim_q)
        self.__hidden_state_q = torch.randn(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_q).to(
            self._dev, non_blocking=True)

        # GRU to track sigma
        self._input_dim_sigma = self._hidden_dim_q + self._state_dim * self._gru_input_mult
        self._hidden_dim_sigma = self._state_dim ** 2
        self._gru_sigma = nn.GRU(self._input_dim_sigma, self._hidden_dim_sigma)
        self.__hidden_state_sigma = torch.randn(self._seq_len_input_gru, self._batch_size_gru,
                                                self._hidden_dim_sigma).to(
            self._dev, non_blocking=True)

        # GRU to track s
        self._input_dim_s = self._observation_dim ** 2 + 2 * self._observation_dim * self._gru_input_mult
        self._hidden_dim_s = self._observation_dim ** 2
        self._gru_s = nn.GRU(self._input_dim_s, self._hidden_dim_s)
        self.__hidden_state_s = torch.randn(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_s).to(
            self._dev,
            non_blocking=True)

        # Fully connected 1
        self._input_dim_fc1 = self._hidden_dim_sigma
        self._output_dim_fc1 = self._observation_dim ** 2
        self._fc1_layer = nn.Sequential(
            nn.Linear(self._input_dim_fc1, self._output_dim_fc1),
            nn.ReLU())

        # Fully connected 2
        self._input_dim_fc2 = self._hidden_dim_s + self._hidden_dim_sigma
        self._output_dim_fc2 = self._observation_dim * self._state_dim
        self._hidden_dim_fc2 = self._input_dim_fc2 * self._gru_output_mult
        self._fc2_layer = nn.Sequential(
            nn.Linear(self._input_dim_fc2, self._hidden_dim_fc2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim_fc2, self._output_dim_fc2))

        # Fully connected 3
        self._input_dim_fc3 = self._hidden_dim_s + self._output_dim_fc2
        self._output_dim_fc3 = self._state_dim ** 2
        self._fc3_layer = nn.Sequential(
            nn.Linear(self._input_dim_fc3, self._output_dim_fc3),
            nn.ReLU())

        # Fully connected 4
        self._input_dim_fc4 = self._hidden_dim_sigma + self._output_dim_fc3
        self._output_dim_fc4 = self._hidden_dim_sigma
        self._fc4_layer = nn.Sequential(
            nn.Linear(self._input_dim_fc4, self._output_dim_fc4),
            nn.ReLU())

        # Fully connected 5
        self._input_dim_fc5 = self._state_dim
        self._output_dim_fc5 = self._state_dim * self._gru_input_mult
        self._fc5_layer = nn.Sequential(
            nn.Linear(self._input_dim_fc5, self._output_dim_fc5),
            nn.ReLU())

        # Fully connected 6
        self._input_dim_fc6 = self._state_dim
        self._output_dim_fc6 = self._state_dim * self._gru_input_mult
        self._fc6_layer = nn.Sequential(
            nn.Linear(self._input_dim_fc6, self._output_dim_fc6),
            nn.ReLU())

        # Fully connected 7
        self._input_dim_fc7 = 2 * self._observation_dim
        self._output_dim_fc7 = 2 * self._observation_dim * self._gru_input_mult
        self._fc7_layer = nn.Sequential(
            nn.Linear(self._input_dim_fc7, self._output_dim_fc7),
            nn.ReLU())

        logger.debug("Network built")

    def __f_discrete(self, x: Tensor) -> Tensor:
        matrix_a = self._state_evolution_continuous(x)
        discretized_f = self._get_matrix_exp_approx(matrix_a)
        return torch.matmul(discretized_f, x)

    def __expand_dim(self, x: Tensor) -> Tensor:
        expanded_tensor = torch.empty(self._seq_len_input_gru, self._batch_size_gru, x.shape[-1])
        expanded_tensor[0, 0, :] = x
        return expanded_tensor

    def _prediction_step(self):
        self.__state_prev_prior = self.__state_prior
        if self._is_nonlinear_evolution:
            self.__state_prior = self.__f_discrete(self.__state_posterior)
        else:
            self.__state_prior = torch.matmul(self._const_discrete_matrix, self.__state_posterior)
        self.__observation_prior = torch.matmul(self._measurement_matrix, self.__state_prior)

    def _update_step(self, current_observation: Tensor):
        # Feature 1: y_t-y_t-1
        observation_difference = current_observation - self.__observation_prev_posterior

        # Feature 2: y_t - y_t|t-1
        innovation_difference = current_observation - self.__observation_prior

        # Feature 3: x_t|t - x_t|t
        forward_evolution_difference = self.__state_posterior - self.__state_prev_posterior

        # Feature 4: x_t|t - x_t|t-1
        forward_update_difference = self.__state_posterior - self.__state_prev_prior

        observation_difference_norm = func.normalize(observation_difference, p=2, dim=0, eps=1e-12, out=None)
        innovation_difference_norm = func.normalize(innovation_difference, p=2, dim=0, eps=1e-12, out=None)
        forward_evolution_difference_norm = func.normalize(forward_evolution_difference, p=2, dim=0, eps=1e-12,
                                                           out=None)
        forward_update_difference_norm = func.normalize(forward_update_difference, p=2, dim=0, eps=1e-12, out=None)

        kalman_gain = self._compute_kalman_gain(observation_difference_norm, innovation_difference_norm,
                                                forward_evolution_difference_norm, forward_update_difference_norm)

        self.__kalman_gain = torch.reshape(kalman_gain, (self._state_dim, self._observation_dim))

        self.__state_prev_posterior = self.__state_posterior
        self.__state_posterior = self.__state_prior + torch.matmul(self.__kalman_gain, innovation_difference)

        self.__observation_prev_posterior = current_observation

    def _compute_kalman_gain(self, observation_diff: Tensor, innovation_diff: Tensor, forward_evolution_diff: Tensor,
                             forward_update_diff: Tensor) -> Tensor:

        observation_diff = self.__expand_dim(observation_diff)
        innovation_diff = self.__expand_dim(innovation_diff)
        forward_evolution_diff = self.__expand_dim(forward_evolution_diff)
        forward_update_diff = self.__expand_dim(forward_update_diff)

        # Forward flow
        # Layer 5
        input_fc5_layer = forward_evolution_diff
        output_fc5_layer = self._fc5_layer(input_fc5_layer)

        # Q-GRU
        input_q = output_fc5_layer
        output_q, self.__hidden_state_q = self._gru_q(input_q, self.__hidden_state_q)

        # Layer 6
        input_fc6_layer = forward_update_diff
        output_fc6_layer = self._fc6_layer(input_fc6_layer)

        # Sigma-GRU
        input_sigma = torch.cat((output_q, output_fc6_layer), 2)
        output_sigma, self.__hidden_state_sigma = self._gru_sigma(input_sigma, self.__hidden_state_sigma)

        # Layer 1
        input_fc1_layer = output_sigma
        output_fc1_layer = self._fc1_layer(input_fc1_layer)

        # Layer 7
        input_fc7_layer = torch.cat((observation_diff, innovation_diff), 2)
        output_fc7_layer = self._fc7_layer(input_fc7_layer)

        # S-GRU
        input_s = torch.cat((output_fc1_layer, output_fc7_layer), 2)
        output_s, self.__hidden_state_s = self._gru_s(input_s, self.__hidden_state_s)

        # Layer 2
        input_fc2_layer = torch.cat((output_sigma, output_s), 2)
        output_fc2_layer = self._fc2_layer(input_fc2_layer)

        # Backward flow
        # Layer 3
        input_fc3_layer = torch.cat((output_s, output_fc2_layer), 2)
        output_fc3_layer = self._fc3_layer(input_fc3_layer)

        # Layer 4
        input_fc4_layer = torch.cat((output_sigma, output_fc3_layer), 2)
        output_fc4_layer = self._fc4_layer(input_fc4_layer)

        # Updating hidden state of the Sigma-GRU
        self.__hidden_state_sigma = output_fc4_layer

        return output_fc2_layer

    # These functions must be called in order to train/test the estimator
    def set_initial_conditions(self, *args):
        if len(args) == 1:
            self.__state_prior = args[0].to(self._dev, non_blocking=True)
        elif len(args) > 1:
            logger.warning(
                f'{len(args)} initial conditions were given, but only 1 is used')
            self.__state_prior = args[0].to(self._dev, non_blocking=True)

        self.__state_prev_prior = self.__state_prior.to(self._dev, non_blocking=True)
        self.__state_posterior = self.__state_prior.to(self._dev, non_blocking=True)
        self.__state_prev_posterior = self.__state_prior.to(self._dev, non_blocking=True)
        self.__observation_prev_posterior = torch.matmul(self._measurement_matrix, self.__state_prior).to(self._dev,
                                                                                                          non_blocking=True)

    def initialize_hidden_state(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_s).zero_()
        self.__hidden_state_s = hidden.data
        self.__hidden_state_s[0, 0, :] = self._initial_s.flatten()
        hidden = weight.new(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_sigma).zero_()
        self.__hidden_state_sigma = hidden.data
        self.__hidden_state_sigma[0, 0, :] = self._initial_sigma.flatten()
        hidden = weight.new(self._seq_len_input_gru, self._batch_size_gru, self._hidden_dim_q).zero_()
        self.__hidden_state_q = hidden.data
        self.__hidden_state_q[0, 0, :] = self._initial_q.flatten()

    # The tensor has to be on the current device (no additional tests --> boost performance)
    def forward(self, feature: Tensor) -> Tensor:
        self._prediction_step()
        self._update_step(feature)
        return self.__state_posterior

    @torch.no_grad()
    def estimate_state_from_observations(self, observations: Tensor) -> Tensor:
        # x.size=[n_timesteps+1, n_features]
        n_timesteps = observations.size(dim=0)

        estimated_states = self._get_empty_state_trajectory(n_timesteps)

        estimated_states[0, :] = self.__state_prev_posterior

        observations = observations.to(self._dev)

        self.eval()

        for i in range(1, n_timesteps + 1):
            actual_observation = observations[i - 1, :]
            estimated_states[i, :] = self.forward(actual_observation)

        return estimated_states

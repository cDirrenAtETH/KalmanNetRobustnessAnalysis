import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions import MultivariateNormal

from models.params import GerstnerWaveParameters
from models.params import LinearPosVelParameters

# Module logger
logger = logging.getLogger(__name__)


# Base class
class Model(ABC):
    # Protected fields which have to be initialized
    _state_dim: int
    _observation_dim: int

    _state_evolution_continuous: Callable[[Tensor], Tensor]
    _measurement_matrix: Tensor

    _t_max_batch: float
    _n_timesteps_batch: int
    _delta_t: float

    _initial_state: Tensor
    _initial_state_random = None

    _max_taylor_coefficient = 1

    _process_noise_distr: Distribution
    _measurement_noise_distr: Distribution

    def __init__(self, state_dim: int, observation_dim: int, measurement_matrix: Tensor):
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.no_grad()

        self._state_dim = state_dim
        self._observation_dim = observation_dim

        self._measurement_matrix = measurement_matrix.cpu()

    # Protected methods
    def _get_t_max(self) -> float:
        return (self._n_timesteps_batch - 1) * self._delta_t

    def _get_matrix_exp_approx(self, matrix: Tensor) -> Tensor:
        # Taylor expansion of exp(A*delta_t)
        discretized_matrix = torch.eye(self._state_dim)
        for i in range(1, self._max_taylor_coefficient + 1):
            discretized_matrix_i = torch.matrix_power(torch.mul(matrix, self._delta_t), i) / math.factorial(i)
            discretized_matrix = torch.add(discretized_matrix, discretized_matrix_i)

        return discretized_matrix

    # Public methods
    def get_continuous_state_evolution(self) -> Callable[[Tensor], Tensor]:
        return self._state_evolution_continuous

    def get_discrete_state_evolution(self) -> Callable[[Tensor], Tensor]:
        return self.f_discrete

    def get_measurement_matrix(self) -> Tensor:
        return self._measurement_matrix.T

    def get_max_taylor_coefficient(self) -> int:
        return self._max_taylor_coefficient

    def get_delta_t(self) -> float:
        return self._delta_t

    def get_timevector_batch(self):
        return torch.from_numpy(np.linspace(self._delta_t, self._t_max_batch, self._n_timesteps_batch))

    def get_initial_condition(self):
        return self._initial_state

    def get_empty_state_trajectory(self):
        return torch.empty(size=(self._n_timesteps_batch + 1, self._state_dim))

    def get_empty_state_batch_trajectory(self, batch_size):
        return torch.empty(size=(batch_size, self._n_timesteps_batch + 1, self._state_dim))

    def get_empty_observation_trajectory(self):
        return torch.empty(size=(self._n_timesteps_batch, self._observation_dim))

    def get_empty_observation_batch_trajectory(self, batch_size: int):
        return torch.empty(size=(batch_size, self._n_timesteps_batch, self._observation_dim))

    def get_initial_condition_random(self):
        if self._initial_state_random is not None:
            return self._initial_state_random
        logger.warning("Random initial condition was not set, return default")
        return self._initial_state

    def get_observation_dim(self) -> int:
        return self._observation_dim

    # Setter methods
    def set_initial_state(self, initial_state: Tensor):
        if initial_state.shape == self._initial_state.shape:
            self._initial_state = initial_state
        else:
            raise ValueError("The given initial state must have the same shape as the state dimension")

    def set_simulation_params(self, delta_t: Optional[float] = None, n_timesteps: Optional[int] = None):
        if delta_t is not None:
            self._delta_t = delta_t
        if n_timesteps is not None:
            self._n_timesteps_batch = n_timesteps
        self._t_max_batch = self._get_t_max()

    def set_process_noise(self, covariance: Tensor):
        self._process_noise_distr = MultivariateNormal(torch.zeros(self._state_dim),
                                                       covariance_matrix=covariance.cpu())

    def set_measurement_noise(self, covariance: Tensor):
        self._measurement_noise_distr = MultivariateNormal(torch.zeros(self._observation_dim),
                                                           covariance_matrix=covariance.cpu())

    # Sequence generation methods
    def generate_discrete(self) -> tuple[Tensor, Tensor, Tensor]:
        # Allocate array for solution
        x = self.get_empty_state_trajectory()
        y = self.get_empty_observation_trajectory()

        # Get timevector
        t = self.get_timevector_batch()

        # Initialize
        x_prev = self.get_initial_condition()
        x[0, :] = x_prev

        # Generate sequence
        for i, actual_time in enumerate(t):
            x_actual = self.f_discrete(x_prev)
            y_actual = self.h(x_actual)

            # Save state and observation
            x[i + 1, :] = x_actual
            y[i, :] = y_actual

            # Save current to previous
            x_prev = x_actual

        return t, x, y

    def generate_discrete_measurement_noise(self) -> tuple[Tensor, Tensor, Tensor]:
        # Allocate array for solution
        x = self.get_empty_state_trajectory()
        y = self.get_empty_observation_trajectory()

        # Get timevector
        t = self.get_timevector_batch()

        # Initialize
        x_prev = self.get_initial_condition()
        x[0, :] = x_prev

        # Generate sequence
        for i, actual_time in enumerate(t):
            x_actual = self.f_discrete(x_prev)
            y_actual = self.h_noise(x_actual, all_timesteps=False)

            # Save state and observation
            x[i + 1, :] = x_actual
            y[i, :] = y_actual

            # Save current to previous
            x_prev = x_actual

        return t, x, y

    def generate_discrete_process_noise(self) -> tuple[Tensor, Tensor, Tensor]:
        # Allocate array for solution
        x = self.get_empty_state_trajectory()
        y = self.get_empty_observation_trajectory()

        # Get timevector
        t = self.get_timevector_batch()

        # Initialize
        x_prev = self.get_initial_condition()
        x[0, :] = x_prev

        # Generate sequence
        for i, actual_time in enumerate(t):
            x_actual = self.f_discrete_noise(x_prev)
            y_actual = self.h(x_actual)

            # Save state and observation
            x[i + 1, :] = x_actual
            y[i, :] = y_actual

            # Save current to previous
            x_prev = x_actual

        return t, x, y

    def generate_discrete_noise(self) -> [Tensor, Tensor, Tensor]:
        # Allocate array for solution
        x = self.get_empty_state_trajectory()
        y = self.get_empty_observation_trajectory()

        # Get timevector
        t = self.get_timevector_batch()

        # Initialize
        x_prev = self.get_initial_condition()
        x[0, :] = x_prev

        # Generate sequence
        for i, actual_time in enumerate(t):
            x_actual = self.f_discrete_noise(x_prev)
            y_actual = self.h_noise(x_actual, all_timesteps=False)

            # Save state and observation
            x[i + 1, :] = x_actual
            y[i, :] = y_actual

            # Save current to previous
            x_prev = x_actual

        return t, x, y

    def generate_batch_discrete_noise(self, batch_size: int, random_init=False, variance=0.1 ** 2) -> tuple[
        Tensor, Tensor, Tensor]:
        x = self.get_empty_state_batch_trajectory(batch_size)
        y = self.get_empty_observation_batch_trajectory(batch_size)

        initial_state_old = self.get_initial_condition()
        initial_state_batch = initial_state_old

        for i in range(batch_size):
            if random_init:
                self.set_initial_state_random(variance)
            else:
                self.set_initial_state(initial_state_batch)
            [t, x_batch, y_batch] = self.generate_discrete_noise()
            x[i, :, :] = x_batch
            y[i, :, :] = y_batch
            initial_state_batch = x_batch[-1, :]

        self.set_initial_state(initial_state_old)

        return self.get_timevector_batch(), x, y

    def generate_observation_from_state(self, states: Tensor, observations: Tensor, batch_size: int) -> Tensor:
        for batch in range(0, states.size(dim=0)):
            observations[batch, :, :] = self.h_noise(states[batch, 1:, :], all_timesteps=True, batch_size=batch_size)
        return observations

    def get_continuous_state_evolution_callable(self) -> Callable[[Tensor], Tensor]:
        return self.get_continuous_state_evolution()

    def get_discrete_state_evolution_callable(self) -> Callable[[Tensor], Tensor]:
        return self.f_discrete

    # Abstract methods
    @abstractmethod
    def _state_evolution_continuous(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def f_discrete(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def f_discrete_noise(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def h(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def h_noise(self, x: Tensor, all_timesteps=False, batch_size=100) -> Tensor:
        pass

    @abstractmethod
    def set_initial_state_random(self, variance: float) -> None:
        pass


class LinearPosVelModel(Model):

    def __init__(self):

        super().__init__(LinearPosVelParameters.n, LinearPosVelParameters.m, Tensor([[1],
                                                                                     [0]]))

        self._state_dim = LinearPosVelParameters.n
        self._observation_dim = LinearPosVelParameters.m

        self._measurement_matrix = Tensor([[1],
                                           [0]])

        self._x_0 = LinearPosVelParameters.x_0

        self._delta_t = LinearPosVelParameters.delta_t
        self._n_timesteps_batch = LinearPosVelParameters.n_timesteps
        self._t_max_batch = self._get_t_max()

        self._max_taylor_coefficient = LinearPosVelParameters.n_taylor_coefficients

        self._process_noise_distr = LinearPosVelParameters.process_noise_distr
        self._measurement_noise_distr = LinearPosVelParameters.measurement_noise_distr

        self._state_evolution_discretized = self._get_matrix_exp_approx(
            self._state_evolution_continuous(torch.eye(self._state_dim)))

    # Private methods
    def _state_evolution_continuous(self, x) -> Tensor:
        return torch.matmul(Tensor([[0, 1], [0, 0]]), x)

    # Public methods
    # Evolution model (x_dot=A*x)
    # x = [position, velocity]
    # x.size = [n_features]
    def f_discrete(self, x) -> Tensor:
        return torch.matmul(self._state_evolution_discretized, x)

    def f_discrete_noise(self, x) -> Tensor:
        return self.f_discrete(x) + self._process_noise_distr.sample()

    # Measurement model
    # y = [pos]
    # x.size = [n_timesteps, n_features]
    def h(self, x):
        return torch.matmul(x, self._measurement_matrix)

    def h_noise(self, x, all_timesteps=False, batch_size=100):
        if not all_timesteps:
            return self.h(x) + self._measurement_noise_distr.sample()
        else:
            return self.h(x) + self._measurement_noise_distr.sample(sample_shape=[batch_size])

    def set_initial_state_random(self, variance: float) -> None:
        self._initial_state_random = torch.rand_like(self._x_0) * variance


class GerstnerWaves(Model):
    def __init__(self):

        super().__init__(GerstnerWaveParameters.state_dim, GerstnerWaveParameters.observation_dim,
                         GerstnerWaveParameters.measurement_matrix)

        self._initial_state = GerstnerWaveParameters.x_0

        self._delta_t = GerstnerWaveParameters.delta_t

        self._max_taylor_coefficient = GerstnerWaveParameters.n_taylor_coefficients

        self._process_noise_distr = GerstnerWaveParameters.process_noise_distr
        self._measurement_noise_distr = GerstnerWaveParameters.measurement_noise_distr

        self._n_timesteps_batch = GerstnerWaveParameters.n_timesteps_batch

    # Evolution model (x_dot=A(x)*x)
    # x = [u, u_dot, omega, u_resting]
    # x.size = [n_features]
    def _state_evolution_continuous(self, x: Tensor):
        matrix_a = torch.zeros((self._state_dim, self._state_dim))
        omega_squared = x[2] ** 2
        matrix_a[0, 1] = 1
        matrix_a[1, 0] = -omega_squared
        matrix_a[1, 3] = omega_squared
        return matrix_a

    # Public methods
    def f_discrete(self, x):
        matrix_a = self._state_evolution_continuous(x)
        state_evolution_discrete = self._get_matrix_exp_approx(matrix_a)
        return torch.matmul(state_evolution_discrete, x)

    def f_discrete_noise(self, x):
        x_next_timestep = self.f_discrete(x) + self._process_noise_distr.sample()

        if x_next_timestep[2].item() < 0.25:
            x_next_timestep[2] = 0.5 - x_next_timestep[2]

        # if x_next_timestep[3].item() > 1:
        #     x_next_timestep[3] = 2 - x_next_timestep[3]
        # elif x_next_timestep[3].item() < -1:
        #     x_next_timestep[3] = -2-x_next_timestep[3]

        return x_next_timestep

    # Measurement model
    # y = [y_u, y_u_dot]
    # x.size = [n_timesteps, n_features]
    def h(self, x):
        return torch.matmul(x, self._measurement_matrix)

    def h_cor(self, x):
        return torch.matmul(self._measurement_matrix.T, x)

    def h_noise(self, x, all_timesteps=False, batch_size=100):
        if not all_timesteps:
            return self.h(x) + self._measurement_noise_distr.sample()
        else:
            return self.h(x) + self._measurement_noise_distr.sample(sample_shape=[batch_size])

    def set_initial_state_random(self, variance: float) -> None:
        initial_state_random = torch.mul(torch.rand_like(self._initial_state), variance)
        # Only positive frequencies
        self._initial_state_random[2] = abs(initial_state_random[2])


class GerstnerWavesGPU(Model):
    _dev: torch.device

    def __init__(self):

        self.__set_dev()

        self._measurement_matrix_dev = GerstnerWaveParameters.measurement_matrix.to(self._dev)

        super().__init__(GerstnerWaveParameters.state_dim, GerstnerWaveParameters.observation_dim,
                         self._measurement_matrix_dev)

        self._initial_state = GerstnerWaveParameters.x_0.to(self._dev)

        self._delta_t = GerstnerWaveParameters.delta_t

        self._max_taylor_coefficient = GerstnerWaveParameters.n_taylor_coefficients

        self._process_noise_distr = GerstnerWaveParameters.process_noise_distr
        self._measurement_noise_distr = GerstnerWaveParameters.measurement_noise_distr

        self._n_timesteps_batch = GerstnerWaveParameters.n_timesteps_batch

    # Protected methods
    def __set_dev(self):
        if torch.cuda.is_available():
            self._dev = torch.device("cuda:0")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            logger.debug('Running on the GPU')
        else:
            self._dev = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
            logger.debug('Running on the CPU')

    # Evolution model (x_dot=A(x)*x)
    # x = [u, u_dot, omega, u_resting]
    # x.size = [n_features]
    def _state_evolution_continuous(self, x: Tensor):
        matrix_a = torch.zeros((self._state_dim, self._state_dim))
        omega = x[2]
        matrix_a[0, 1] = 1
        matrix_a[1, 0] = -omega ** 2
        matrix_a[1, 3] = omega ** 2
        return matrix_a

    def _get_matrix_exp_approx(self, matrix: Tensor) -> Tensor:
        # Taylor expansion of exp(A*delta_t)
        discretized_matrix = torch.eye(self._state_dim)
        for i in range(1, self._max_taylor_coefficient + 1):
            discretized_matrix_i = torch.matrix_power(torch.mul(matrix, self._delta_t), i) / math.factorial(i)
            discretized_matrix = torch.add(discretized_matrix, discretized_matrix_i)

        return discretized_matrix

    # Public methods
    def f_discrete(self, x):
        matrix_a = self._state_evolution_continuous(x)
        state_evolution_discrete = self._get_matrix_exp_approx(matrix_a).to(self._dev)
        return torch.matmul(state_evolution_discrete, x)

    def f_discrete_noise(self, x):
        x_next_timestep = self.f_discrete(x) + self._process_noise_distr.sample()
        if x_next_timestep[2].item() < 0.5:
            x_next_timestep[2] = 1 - x_next_timestep[2]
        return x_next_timestep

    # Measurement model
    # y = [y_u, y_u_dot]
    # x.size = [n_timesteps, n_features]
    def h(self, x):
        return torch.matmul(x, self._measurement_matrix)

    def h_cor(self, x):
        return torch.matmul(self._measurement_matrix_dev.T, x)

    def h_noise(self, x, all_timesteps=False, batch_size=100):
        if not all_timesteps:
            return self.h(x) + self._measurement_noise_distr.sample()
        else:
            return self.h(x) + self._measurement_noise_distr.sample(sample_shape=[batch_size])

    def set_initial_state_random(self, variance: float) -> None:
        initial_state_random = torch.mul(torch.rand_like(self._initial_state), variance)
        # Only positive frequencies
        self._initial_state_random[2] = abs(initial_state_random[2])

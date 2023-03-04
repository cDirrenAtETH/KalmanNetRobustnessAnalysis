import torch
from torch.distributions import MultivariateNormal

torch.set_default_tensor_type('torch.FloatTensor')


class DefaultSimulationParameters:
    # Training default params
    n_train = 1000

    # Validation default params
    n_validation = 100

    # Test default params
    n_test = 100


class LinearPosVelParameters:
    # Basic Parameters
    n = 2
    m = 1

    # Default initial conditions
    pos_0 = 0
    vel_0 = 1

    x_0 = torch.Tensor([pos_0, vel_0])

    # Simulation parameters (in order to generate some data)
    n_timesteps = 10
    delta_t = 1e-1
    n_taylor_coefficients = 1

    # Noise parameters
    variance_process = 0.01 ** 2
    variance_measurement = 0.1 ** 2
    process_noise_distr = MultivariateNormal(torch.zeros(n),
                                             covariance_matrix=variance_process * torch.eye(n))
    measurement_noise_distr = MultivariateNormal(torch.zeros(m),
                                                 covariance_matrix=variance_measurement * torch.eye(m))


class GerstnerWaveParameters:
    # Basic Parameters
    state_dim = 4
    observation_dim = 1

    # Default initial conditions
    u_0 = 0
    u_0_dot = 1
    omega = 1
    u_resting = 0

    x_0 = torch.Tensor([u_0, u_0_dot, omega, u_resting])

    # Model params
    measurement_matrix = torch.Tensor([[1, 0, 0, 0]]).T

    # Simulation parameters (in order to generate some data)
    n_timesteps_batch = 100
    delta_t = 1e-1
    n_taylor_coefficients = 5

    # Noise parameters
    variance_process = 0.01 ** 2
    variance_measurement = 0.1 ** 2
    process_noise_distr = MultivariateNormal(torch.zeros(state_dim),
                                             covariance_matrix=variance_process * torch.eye(state_dim))
    measurement_noise_distr = MultivariateNormal(torch.zeros(observation_dim),
                                                 covariance_matrix=variance_measurement * torch.eye(observation_dim))

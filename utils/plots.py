from utils.parameters import DefaultGerstnerPlotParameters, DefaultLinearPosVelParameters
from utils.utils import get_min_loss

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
from numpy.typing import ArrayLike

import torch

from typing import Optional, Any

import math

import logging

# Module logger
logger = logging.getLogger(__name__)


class LinearPosVelPlots:
    @staticmethod
    def plot_states_default(t, x, title=None):
        fig, ax = plt.subplots(DefaultLinearPosVelParameters.n_rows_state, sharex=True)
        if title is None:
            fig.suptitle(DefaultLinearPosVelParameters.title_state)
        else:
            fig.suptitle(title)
        for i, axis in enumerate(ax):
            axis.plot(t, x[1:, i])
            axis.set_ylabel(DefaultLinearPosVelParameters.y_label_state[i])
        ax[-1].set_xlabel(DefaultLinearPosVelParameters.x_label)
        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_observations_default(t, y, title=None):
        fig, ax = plt.subplots(DefaultLinearPosVelParameters.n_rows_observation, sharex=True)
        if title is None:
            fig.suptitle(DefaultLinearPosVelParameters.title_observation)
        else:
            fig.suptitle(title)
        ax.plot(t, y[:])
        ax.set_ylabel(DefaultLinearPosVelParameters.y_label_observation)
        ax.set_xlabel(DefaultLinearPosVelParameters.x_label)
        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_estimated_vs_true_state_default(t, x_true, x_estimated, title=None):
        fig, ax = plt.subplots(DefaultLinearPosVelParameters.n_rows_state, sharex=True)
        if title is None:
            fig.suptitle(DefaultLinearPosVelParameters.title_state)
        else:
            fig.suptitle(title)
        for i, axis in enumerate(ax):
            axis.plot(t, x_true[1:, i])
            axis.plot(t, x_estimated[1:, i])
            axis.set_ylabel(DefaultLinearPosVelParameters.y_label_state[i])
        ax[-1].set_xlabel(DefaultLinearPosVelParameters.x_label)
        fig.tight_layout()
        plt.show()


class GerstnerWavesPlots:

    def __init__(self, plot_path: str, result_path: str, result_name='results', plot_add_name=None,
                 sampling_time=1e-1, fontsize=10, figsize=(11, 6), format='eps', dpi=1000):
        self._plot_path = plot_path
        self._plot_add_name = plot_add_name
        self._result_path = result_path
        self._result_name = result_name

        self._format = format
        self._dpi = dpi

        self._time_in_seconds_label = r'$\mathrm{t}$ [s]'
        self._time_in_minutes_label = r'$\mathrm{t}$ [min]'
        self._observation_label = [r'$\mathrm{u}$']
        self._state_label = self._observation_label + [r'$\mathrm{\dot{u}}$', r'$\mathrm{\omega}$', r'$\mathrm{u_0}$']
        self._mse_db_label = 'MSE [dB]'
        self._mse_linear_label = 'MSE [linear]'

        self._knet_training_legend = ['KNet - Train', 'KNet - Validation', 'KNet - Test', 'Kalman Filter']

        self._style_train_val_test_kf = ['o-', 'o-', '--', '--']

        default_estimator_colors = DefaultGerstnerPlotParameters.default_estimator_colors
        default_estimator_names = DefaultGerstnerPlotParameters.default_estimator_names
        default_estimator_styles = DefaultGerstnerPlotParameters.default_estimator_styles

        self._default_estimator_colors = dict(zip(default_estimator_names, default_estimator_colors))
        self._default_estimator_styles = dict(zip(default_estimator_names, default_estimator_styles))

        self._sampling_time = sampling_time

        self._fontsize = fontsize
        self._figsize = figsize

    # def __plot_data(self, data:np.ndarray, color_dict:dict[str, str], style_dict: dict[str, str], axis: Optional[plt.Axes]=None):
    #     if axis is None:

    def __save_figure(self, plot_name: str):
        if self._plot_add_name is not None:
            filename = f'{self._plot_path}/{plot_name}_{self._plot_add_name}.{self._format}'
        else:
            filename = f'{self._plot_path}/{plot_name}.{self._format}'
        plt.savefig(filename, format=self._format, dpi=self._dpi)

    def __get_time_vector_in_minutes(self, n_samples: int) -> ArrayLike:
        return np.linspace(self._sampling_time, self._sampling_time * n_samples, n_samples) / 60

    def __plot_on_axis(self, x: np.ndarray, y: np.ndarray, axis: plt.Axes, label: str, color: Optional[str] = None,
                       style: Optional[str] = None):
        if style is not None and color is not None:
            axis.plot(x, y, style, label=label, color=color)
        elif style is not None:
            axis.plot(x, y, style, label=label)
        elif color is not None:
            axis.plot(x, y, label=label, color=color)
        else:
            axis.plot(x, y, '--o', label=label)

    def set_add_plot_name(self, add_name: str):
        self._plot_add_name = add_name

    def set_fontsize(self, fontsize: int):
        self._fontsize = fontsize

    def set_figsize(self, figsize: tuple[int, int]):
        self._figsize = figsize

    def reset_add_plot_name(self):
        self._plot_add_name = None

    def plot_state_trajectory(self, state: ArrayLike, save_plot=False, title: Optional[str] = None):
        fig, ax = plt.subplots(4, sharex=True, figsize=self._figsize)

        if title is not None:
            fig.suptitle(title, fontsize=self._fontsize)

        timevector = self.__get_time_vector_in_minutes(np.array(state).shape[0] - 1)

        for i, axis in enumerate(ax):
            axis.plot(timevector, state[1:, i])
            axis.set_ylabel(self._state_label[i], fontsize=self._fontsize)

        ax[-1].set_xlabel(self._time_in_minutes_label, fontsize=self._fontsize)

        plt.tight_layout()
        plt.grid()

        if save_plot:
            self.__save_figure('state_trajectory')

        plt.show()

    def plot_observation_trajectory(self, observations: np.ndarray, save_plot=False, title: Optional[str] = None):

        plt.figure(figsize=self._figsize)

        if title is not None:
            plt.title(title)

        timevector = self.__get_time_vector_in_minutes(observations.shape[0])
        plt.plot(timevector, observations)
        plt.ylabel(self._observation_label[0], fontsize=self._fontsize)
        plt.xlabel(self._time_in_minutes_label, fontsize=self._fontsize)

        plt.tight_layout()
        plt.grid()

        if save_plot:
            if save_plot:
                self.__save_figure('observation')

        plt.show()

    def plot_state_estimates(self, estimates: dict[str, np.ndarray], style_dict: Optional[dict[str, str]] = None,
                             color_dict: Optional[dict[str, str]] = None, printable_states: Optional[list[int]] = None,
                             save_plot=False,
                             title: Optional[str] = None,
                             legend_loc: str = 'upper right'):
        if style_dict is None:
            style_dict = self._default_estimator_styles
        if color_dict is None:
            color_dict = self._default_estimator_colors
        if printable_states is None:
            printable_states = list(range(4))

        fig, ax = plt.subplots(len(printable_states), sharex=True, figsize=self._figsize, squeeze=False)
        ax = ax.reshape(-1)

        if title is not None:
            fig.suptitle(title, fontsize=self._fontsize)

        timevector = self.__get_time_vector_in_minutes(list(estimates.values())[0].shape[0])

        for i, axis in enumerate(ax):
            for estimator_name, estimate in estimates.items():
                style = style_dict.get(estimator_name)
                color = color_dict.get(estimator_name)
                if style is not None and color is not None:
                    axis.plot(timevector, estimate[:, printable_states[i]], style_dict.get(estimator_name),
                              label=estimator_name, color=color)
                elif style is not None:
                    axis.plot(timevector, estimate[:, printable_states[i]], style_dict.get(estimator_name),
                              label=estimator_name)
                elif color is not None:
                    axis.plot(timevector, estimate[:, printable_states[i]],
                              label=estimator_name, color=color)
                else:
                    axis.plot(timevector, estimate[:, printable_states[i]], label=estimator_name)

            axis.legend(fontsize=self._fontsize, loc=legend_loc)
            axis.grid()
            axis.set_ylabel(self._state_label[printable_states[i]], fontsize=self._fontsize)

        ax[-1].set_xlabel(self._time_in_minutes_label, fontsize=self._fontsize)

        plt.tight_layout()

        if save_plot:
            self.__save_figure('state_estimates')

        plt.show()

    # def plot_complete_state_estimates(self, true_state: np.ndarray, estimated_state: np.ndarray, save_plot=False,
    #                                   title: Optional[str] = None):
    #     fig, ax = plt.subplots(4, sharex=True, figsize=self._figsize)
    #
    #     if title is not None:
    #         fig.suptitle(title, fontsize=self._fontsize)
    #
    #     timevector = self.__get_time_vector_in_minutes(np.array(true_state).shape[0] - 1)
    #
    #     if len(ax) == 1:
    #         ax = list[ax]
    #
    #     for i, axis in enumerate(ax):
    #         axis.plot(timevector, true_state[1:, i], 'g', label='Ground truth')
    #         axis.plot(timevector, estimated_state[1:, i], 'r', label='Estimated state')
    #         axis.legend(fontsize=self._fontsize)
    #         axis.set_ylabel(self._state_label[i], fontsize=self._fontsize)
    #
    #     ax[-1].set_xlabel(self._time_in_minutes_label, fontsize=self._fontsize)
    #
    #     plt.tight_layout()
    #     plt.grid()
    #
    #     if save_plot:
    #         self.__save_figure('estimates')
    #
    #     plt.show()

    def plot_avg_loss_epoch_compare(self, train_loss_epoch_linear: np.ndarray, val_loss_epoch_linear: np.ndarray,
                                    test_avg_loss_knet: Optional[float]=None,
                                    test_avg_loss_kf: Optional[float]=None,
                                    in_decibel=True,
                                    save_plot=False,
                                    plot_title=False,
                                    zoom_bounds: Optional[list] = None,
                                    zoom_location: Optional[list[float, float, float, float]] = None):
        n_epochs = train_loss_epoch_linear.shape[0]

        if test_avg_loss_knet is not None:
            test_avg_loss_knet = test_avg_loss_knet * np.ones(n_epochs)
            test_avg_loss_knet_db = 10 * np.log10(test_avg_loss_knet)

        if test_avg_loss_kf is not None:
            test_avg_loss_kf = test_avg_loss_kf * np.ones(n_epochs)
            test_avg_loss_kf_db = 10 * np.log10(test_avg_loss_kf)

        figure, axis = plt.subplots(figsize=self._figsize)

        epochs_vector = np.array(range(1, n_epochs + 1))

        # Create inset axes
        if zoom_bounds is not None and zoom_location is not None:
            axin = axis.inset_axes(zoom_location)

            axin.set_xlim(zoom_bounds[0], zoom_bounds[1])
            axin.set_ylim(zoom_bounds[2], zoom_bounds[3])

            axis.indicate_inset_zoom(axin)
            axin.grid()
        else:
            axin = None

        if in_decibel:
            train_avg_loss_epoch_db = 10 * np.log10(train_loss_epoch_linear)
            val_avg_loss_epoch_db = 10 * np.log10(val_loss_epoch_linear)

            self.__plot_on_axis(epochs_vector, train_avg_loss_epoch_db, axis=axis, label=self._knet_training_legend[0], style=self._style_train_val_test_kf[0])
            self.__plot_on_axis(epochs_vector, val_avg_loss_epoch_db, axis=axis, label=self._knet_training_legend[1], style=self._style_train_val_test_kf[1])

            if test_avg_loss_knet is not None:
                self.__plot_on_axis(epochs_vector, test_avg_loss_knet_db, axis=axis,
                                    label=self._knet_training_legend[2], style=self._style_train_val_test_kf[2])

            if test_avg_loss_kf is not None:
                self.__plot_on_axis(epochs_vector, test_avg_loss_kf_db, axis=axis,
                                    label=self._knet_training_legend[3], style=self._style_train_val_test_kf[3])

            if axin is not None:
                self.__plot_on_axis(epochs_vector, train_avg_loss_epoch_db, axis=axin,
                                    label=self._knet_training_legend[0], style=self._style_train_val_test_kf[0])
                self.__plot_on_axis(epochs_vector, val_avg_loss_epoch_db, axis=axin,
                                    label=self._knet_training_legend[1], style=self._style_train_val_test_kf[1])

                if test_avg_loss_knet is not None:
                    self.__plot_on_axis(epochs_vector, test_avg_loss_knet_db, axis=axin,
                                        label=self._knet_training_legend[2], style=self._style_train_val_test_kf[2])

                if test_avg_loss_kf is not None:
                    self.__plot_on_axis(epochs_vector, test_avg_loss_kf_db, axis=axin,
                                        label=self._knet_training_legend[3], style=self._style_train_val_test_kf[3])

            axis.set_ylabel(self._mse_db_label, fontsize=self._fontsize)
            if plot_title:
                axis.set_title(f'{self._mse_db_label} - per Epoch', fontsize=self._fontsize)
            type_loss = 'db'

        else:
            self.__plot_on_axis(epochs_vector, train_loss_epoch_linear, axis=axis, label=self._knet_training_legend[0],
                                style=self._style_train_val_test_kf[0])
            self.__plot_on_axis(epochs_vector, val_loss_epoch_linear, axis=axis, label=self._knet_training_legend[1],
                                style=self._style_train_val_test_kf[1])

            if test_avg_loss_knet is not None:
                self.__plot_on_axis(epochs_vector, test_avg_loss_knet, axis=axis,
                                    label=self._knet_training_legend[2], style=self._style_train_val_test_kf[2])

            if test_avg_loss_kf is not None:
                self.__plot_on_axis(epochs_vector, test_avg_loss_kf, axis=axis,
                                    label=self._knet_training_legend[3], style=self._style_train_val_test_kf[3])

            if axin is not None:
                self.__plot_on_axis(epochs_vector, train_loss_epoch_linear, axis=axin,
                                    label=self._knet_training_legend[0], style=self._style_train_val_test_kf[0])
                self.__plot_on_axis(epochs_vector, val_loss_epoch_linear, axis=axin,
                                    label=self._knet_training_legend[1], style=self._style_train_val_test_kf[1])

                if test_avg_loss_knet is not None:
                    self.__plot_on_axis(epochs_vector, test_avg_loss_knet, axis=axin,
                                        label=self._knet_training_legend[2], style=self._style_train_val_test_kf[2])

                if test_avg_loss_kf is not None:
                    self.__plot_on_axis(epochs_vector, test_avg_loss_kf, axis=axin,
                                        label=self._knet_training_legend[3], style=self._style_train_val_test_kf[3])

            axis.set_ylabel(self._mse_linear_label, fontsize=self._fontsize)
            if plot_title:
                axis.set_title(f'{self._mse_linear_label} - per Epoch', fontsize=self._fontsize)
            type_loss = 'linear'

        axis.legend(fontsize=self._fontsize)
        axis.set_xlabel('Number of Training Epochs', fontsize=self._fontsize)

        axis.grid()
        plt.tight_layout()

        if save_plot:
            self.__save_figure(f'avg_loss_per_epoch_comp_{type_loss}')

        plt.show()

    def plot_state_loss_epoch(self, state_loss_epoch: np.ndarray, data_set: str, in_decibel=True, save_plot=False,
                              model_name: Optional[str] = None,
                              zoom_bounds: Optional[list] = None,
                              zoom_location: Optional[list[float, float, float, float]] = None):

        n_epochs = state_loss_epoch.shape[0]
        n_states = state_loss_epoch.shape[1]

        fig, axis = plt.subplots(figsize=self._figsize)

        epochs_vector = range(1, n_epochs + 1)

        if in_decibel:
            state_loss_epoch_db = 10 * np.log10(state_loss_epoch)
            for i in range(n_states):
                axis.plot(epochs_vector, state_loss_epoch_db[:, i], 'o-', label=self._state_label[i])

            if zoom_bounds is not None and zoom_location is not None:
                axin = axis.inset_axes(zoom_location)

                axin.set_xlim(zoom_bounds[0], zoom_bounds[1])
                axin.set_ylim(zoom_bounds[2], zoom_bounds[3])

                for i in range(n_states):
                    axin.plot(epochs_vector, state_loss_epoch_db[:, i], 'o-')

                axis.indicate_inset_zoom(axin)
                axin.grid()

            axis.set_ylabel(self._mse_db_label, fontsize=self._fontsize)
            if model_name is not None:
                axis.set_title(f'{model_name}: State {data_set} {self._mse_db_label} - per Epoch',
                               fontsize=self._fontsize)
            loss_type = 'db'

        else:
            for i in range(n_states):
                axis.plot(epochs_vector, state_loss_epoch[:, i], 'o-', label=self._state_label[i])

            if zoom_bounds is not None and zoom_location is not None:
                axin = axis.inset_axes(zoom_location)

                axin.set_xlim(zoom_bounds[0], zoom_bounds[1])
                axin.set_ylim(zoom_bounds[2], zoom_bounds[3])

                for i in range(n_states):
                    axin.plot(epochs_vector, state_loss_epoch[:, i], 'o-')

                axis.indicate_inset_zoom(axin)
                axin.grid()

            axis.set_ylabel(self._mse_linear_label, fontsize=self._fontsize)
            if model_name is not None:
                axis.set_title(f'{model_name}: State {data_set} {self._mse_linear_label} - per Epoch',
                               fontsize=self._fontsize)
            loss_type = 'linear'

        axis.legend(fontsize=self._fontsize, loc='upper right')
        axis.set_xlabel('Number of Training Epochs', fontsize=self._fontsize)

        plt.tight_layout()
        axis.grid()

        if save_plot:
            self.__save_figure(f'state_{data_set}_{loss_type}_loss')

        plt.show()

    def plot_state_loss_hist(self, state_loss_epoch: np.ndarray, data_set: str, in_decibel=True, save_plot=False,
                             model_name: Optional[str] = None):

        n_states = state_loss_epoch.shape[1]

        plt.figure(figsize=self._figsize)

        if in_decibel:
            state_loss_epoch_db = 10 * np.log10(state_loss_epoch)
            for i in range(n_states):
                sns.kdeplot(state_loss_epoch_db[:, i], label=self._state_label[i], common_grid=True)

            plt.xlabel(self._mse_db_label, fontsize=self._fontsize)
            if model_name is not None:
                plt.title(f'{model_name}: State {data_set} loss - Histogram [dB]', fontsize=self._fontsize)
            data_set += '_db'

        else:
            for i in range(n_states):
                sns.kdeplot(state_loss_epoch[:, i], label=self._state_label[i], common_grid=True)

            plt.xlabel(self._mse_linear_label, fontsize=self._fontsize)
            if model_name is not None:
                plt.title(f'{model_name}: State {data_set} loss - Histogram [linear]', fontsize=self._fontsize)
            data_set += '_linear'

        plt.legend(fontsize=self._fontsize)
        plt.ylabel('Percentage', fontsize=self._fontsize)

        plt.tight_layout()
        plt.grid()

        if save_plot:
            self.__save_figure(f'state_{data_set}_loss_hist')

        plt.show()

    def plot_avg_loss_hist(self, train_state_loss_epoch: np.ndarray, val_state_loss_epoch: np.ndarray,
                           test_state_loss_batch: float, in_decibel=True, save_plot=False,
                           model_name: Optional[str] = None):

        train_avg_loss_epoch = np.mean(train_state_loss_epoch, axis=1)
        val_avg_loss_epoch = np.mean(val_state_loss_epoch, axis=1)
        test_avg_loss_batch = np.mean(test_state_loss_batch, axis=1)

        plt.figure(figsize=self._figsize)

        if in_decibel:
            train_avg_loss_epoch_db = 10 * np.log10(train_avg_loss_epoch)
            val_avg_loss_epoch_db = 10 * np.log10(val_avg_loss_epoch)
            test_avg_loss_batch_db = 10 * np.log10(test_avg_loss_batch)

            sns.kdeplot(train_avg_loss_epoch_db, label=self._knet_training_legend[0], common_norm=False)
            sns.kdeplot(val_avg_loss_epoch_db, label=self._knet_training_legend[1], common_norm=False)
            sns.kdeplot(test_avg_loss_batch_db, label=self._knet_training_legend[2], common_norm=False)

            plt.xlabel(self._mse_db_label, fontsize=self._fontsize)
            if model_name is not None:
                plt.title(f'{model_name}: {self._mse_db_label} - Histogram', fontsize=self._fontsize)
            type_loss = 'db'

        else:
            sns.kdeplot(train_avg_loss_epoch, label=self._knet_training_legend[0], common_grid=True)
            sns.kdeplot(val_avg_loss_epoch, label=self._knet_training_legend[1], common_grid=True)
            sns.kdeplot(test_avg_loss_batch, label=self._knet_training_legend[2], common_grid=True)

            plt.xlabel(self._mse_linear_label, fontsize=self._fontsize)
            if model_name is not None:
                plt.title(f'{model_name}: {self._mse_linear_label} - Histogram', fontsize=self._fontsize)
            type_loss = 'linear'

        plt.legend(fontsize=self._fontsize)
        plt.ylabel('Percentage', fontsize=self._fontsize)

        plt.tight_layout()
        plt.grid()

        if save_plot:
            self.__save_figure(f'avg_loss_hist_{type_loss}')

        plt.show()

    def plot_results_different_process_noise(self, process_noise_list: list, save_plot=False,
                                             title: Optional[str] = None, **kwargs):

        plt.figure(figsize=self._figsize)

        for process_noise in process_noise_list:
            kwargs['process_variance'] = process_noise
            result = get_min_loss(self._result_path + '/' + self._result_name, 'measurement_variance', kwargs)
            t = result["measurement_variance"]
            t = t.apply(lambda x: 10 * math.log10(1 / x))
            plt.plot(t, result["loss_in_dB"], '--o', label=r'$\mathrm{q^2=}$ ' + f'{process_noise:0.0e}')

        plt.legend(fontsize=self._fontsize)
        plt.ylabel(self._mse_db_label, fontsize=self._fontsize)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=self._fontsize)

        if title is not None:
            plt.title(title, fontsize=self._fontsize)

        plt.tight_layout()
        plt.grid()

        if save_plot:
            self.__save_figure(f'different_process_noise')

        plt.show()

    def plot_results_different_estimators(self,
                                          estimator_types: list[str],
                                          add_filter_per_estimator: Optional[dict[str, dict[str, Any]]] = None,
                                          estimator_label_names: Optional[list[str]] = None,
                                          style_dict: Optional[dict[str, str]] = None,
                                          color_dict: Optional[dict[str, str]] = None,
                                          save_plot=False,
                                          title: Optional[str] = None,
                                          y_label: Optional[str] = None,
                                          zoom_bounds: Optional[list] = None,
                                          zoom_location: Optional[list[float, float, float, float]] = None,
                                          **filter_dict):

        fig, axis = plt.subplots(figsize=self._figsize)

        estimator_names = []

        if style_dict is None:
            style_dict = self._default_estimator_styles
        if color_dict is None:
            color_dict = self._default_estimator_colors

        # Construct names from types or arguments
        if estimator_label_names is not None and len(estimator_types) == len(estimator_label_names):
            estimator_names = estimator_label_names
        else:
            last_name = ''
            counter = 1
            for estimator_name in estimator_types:
                if estimator_name == last_name:
                    if counter == 1:
                        estimator_names[-1] = f'{estimator_name} {counter}'
                        counter += 1
                    estimator_names.append(f'{estimator_name} {counter}')
                    counter += counter
                else:
                    estimator_names.append(estimator_name)
                    last_name = estimator_name
                    counter = 1

        # Create inset axes
        if zoom_bounds is not None and zoom_location is not None:
            axin = axis.inset_axes(zoom_location)

            axin.set_xlim(zoom_bounds[0], zoom_bounds[1])
            axin.set_ylim(zoom_bounds[2], zoom_bounds[3])

            axis.indicate_inset_zoom(axin)
            axin.grid()
        else:
            axin = None

        for i, estimator_name in enumerate(estimator_names):

            actual_filter_dict = filter_dict.copy()
            actual_filter_dict['estimator_type'] = estimator_types[i]

            # Get estimator specific filters
            if add_filter_per_estimator is not None and add_filter_per_estimator.get(estimator_name) is not None:
                actual_filter_dict.update(add_filter_per_estimator.get(estimator_name))

            result = get_min_loss(self._result_path, self._result_name, 'measurement_variance',
                                  actual_filter_dict)
            t = result["measurement_variance"]
            t = t.apply(lambda x: 10 * math.log10(1 / x))

            style = style_dict.get(estimator_name)
            color = color_dict.get(estimator_name)

            self.__plot_on_axis(t, result["loss_in_dB"], axis, estimator_name, color=color, style=style)

            if axin is not None:
                self.__plot_on_axis(t, result["loss_in_dB"], axin, estimator_name, color=color, style=style)

        axis.legend(fontsize=self._fontsize)
        if y_label is None:
            axis.set_ylabel(self._mse_db_label, fontsize=self._fontsize)
        else:
            axis.set_ylabel(y_label, fontsize=self._fontsize)
        axis.set_xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=self._fontsize)

        if title is not None:
            axis.set_title(title, fontsize=self._fontsize)

        plt.tight_layout()
        axis.grid()

        if save_plot:
            self.__save_figure(f'different_est')

        plt.show()


@staticmethod
def plot_states_complete(delta_t, x, title=None):
    x = torch.flatten(x[:, 1:, :], start_dim=0, end_dim=1)
    t_max = (x.shape[0] - 1) * (delta_t)
    t = np.linspace(delta_t, t_max, x.shape[0])

    fig, ax = plt.subplots(DefaultGerstnerPlotParameters.n_rows_state, sharex=True)
    if title is None:
        fig.suptitle(DefaultGerstnerPlotParameters.title_state)
    else:
        fig.suptitle(title)
    for i, axis in enumerate(ax):
        axis.plot(t, x[:, i])
        axis.set_ylabel(DefaultGerstnerPlotParameters.y_label_state[i])
    ax[-1].set_xlabel(DefaultGerstnerPlotParameters.x_label)
    fig.tight_layout()
    plt.show()

# @staticmethod
# def plot_estimates(t, x_true, x_estimated, title=None, figsize=(6.4, 4.8)):
#     fig, ax = plt.subplots(DefaultGerstnerPlotParameters.n_rows_state, sharex=True, figsize=figsize)
#
#     if title is None:
#         fig.suptitle(DefaultGerstnerPlotParameters.title_state)
#     else:
#         fig.suptitle(title)
#     for i, axis in enumerate(ax):
#         axis.plot(t, x_true[1:, i], color='g', label="Ground truth")
#         axis.plot(t, x_estimated[1:, i], color='r', label="Estimated values")
#         axis.legend()
#         axis.set_ylabel(DefaultGerstnerPlotParameters.y_label_state[i])
#     ax[-1].set_xlabel(DefaultGerstnerPlotParameters.x_label)
#     fig.tight_layout()
#     plt.show()

# @staticmethod
# def plot_subplots_nrow(t, x, n_rows, **kwargs):
#     fig, ax = plt.subplots(n_rows, sharex=True)
#
#     for i, axis in enumerate(ax):
#         axis.plot(t, x[:, i])
#         if kwargs.get('ylabel') is not None:
#             axis.set_ylabel(kwargs.get('ylabel')[i])
#
#     if kwargs.get('xlabel') is not None:
#         ax[-1].set_xlabel(kwargs.get('xlabel')[-1])
#
#     if kwargs.get('title') is not None:
#         fig.suptitle(kwargs.get('title'))
#     fig.tight_layout()
#     plt.show()

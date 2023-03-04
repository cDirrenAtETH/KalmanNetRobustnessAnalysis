class DefaultGerstnerPlotParameters:
    n_rows_observation = 2
    n_rows_state = 4
    x_label = '$t$'
    y_label_observation = ['$u(t)$', '$\dot{u}$']
    y_label_state = y_label_observation + ["$\omega$", "$u_0$"]
    title_observation = "Observations $y(t)$"
    title_state = "State $x(t)$"
    default_estimator_names = ['KalmanNet (full, known)', 'KalmanNet (velocity, known)', 'KalmanNet (full, unknown)', 'KalmanNet (velocity, unknown)', 'EKF', 'Noise Floor', 'Ground Truth']
    default_estimator_colors = ['blue', 'saddlebrown', 'dodgerblue', 'sandybrown', 'red', 'green', 'green']
    default_estimator_styles = ['o--', 'o--', 'D-.', 'D-.', 'o--', '-', '-']

class DefaultLinearPosVelParameters:
    n_rows_observation = 1
    n_rows_state = 2
    x_label = '$t$'
    y_label_observation = '$p(t)$'
    y_label_state = [y_label_observation] + ["$\dot{p}(t)$"]
    title_observation = "Observations $y(t)$"
    title_state = "State $x(t)$"

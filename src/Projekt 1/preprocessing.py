import numpy as np
from sklearn.preprocessing import StandardScaler


def load_eeg_file(path):

    data = np.loadtxt(path)

    return data


def segment_windows(data, window_size, step_size):

    windows = []

    for start in range(0, len(data) - window_size, step_size):

        window = data[start:start + window_size]

        windows.append(window)

    return np.array(windows)



def fit_scaler(X):

    n_samples, n_channels, n_time = X.shape

    scaler = StandardScaler()

    scaler.fit(
        X.transpose(0, 2, 1).reshape(-1, n_channels)
    )

    return scaler



def apply_scaler(X, scaler):

    n_samples, n_channels, n_time = X.shape

    reshaped = X.transpose(0, 2, 1).reshape(-1, n_channels)

    scaled = scaler.transform(reshaped)

    scaled = scaled.reshape(n_samples, n_time, n_channels)

    return scaled.transpose(0, 2, 1)
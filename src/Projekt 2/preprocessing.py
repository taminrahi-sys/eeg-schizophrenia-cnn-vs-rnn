import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_eeg_file(path):

    """
    Load EEG file from disk.
    """

    data = np.loadtxt(path)

    return data


def segment_signal(signal, window_size, step_size):

    """
    Segment EEG signal into overlapping windows.
    """

    segments = []

    for start in range(
        0,
        len(signal) - window_size,
        step_size
    ):

        segment = signal[start:start + window_size]

        segments.append(segment)

    return np.array(segments)


def create_labels(n_samples, label):

    """
    Create labels for classification.
    """

    return np.full(n_samples, label)


def flatten_windows(windows):

    """
    Flatten EEG windows for ML models.
    """

    n_samples = windows.shape[0]

    return windows.reshape(n_samples, -1)


def scale_features(X_train, X_test):

    """
    Standardize feature vectors.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42
):

    """
    Split dataset into train and test sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def preprocess_pipeline(
    healthy_signal,
    schizophrenia_signal,
    window_size=256,
    step_size=128
):

    """
    Full preprocessing pipeline.
    """

    healthy_windows = segment_signal(
        healthy_signal,
        window_size,
        step_size
    )

    schizophrenia_windows = segment_signal(
        schizophrenia_signal,
        window_size,
        step_size
    )

    X = np.concatenate([
        healthy_windows,
        schizophrenia_windows
    ])

    y_healthy = create_labels(
        len(healthy_windows),
        0
    )

    y_schizophrenia = create_labels(
        len(schizophrenia_windows),
        1
    )

    y = np.concatenate([
        y_healthy,
        y_schizophrenia
    ])

    X = flatten_windows(X)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )
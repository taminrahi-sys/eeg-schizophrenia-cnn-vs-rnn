import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis



def extract_statistical_features(signal):

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "min": np.min(signal),
        "max": np.max(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal)
    }

    return features



def hjorth_parameters(signal):

    first_derivative = np.diff(signal)

    second_derivative = np.diff(first_derivative)

    variance_signal = np.var(signal)

    variance_d1 = np.var(first_derivative)

    variance_d2 = np.var(second_derivative)

    mobility = np.sqrt(variance_d1 / variance_signal)

    complexity = np.sqrt(
        variance_d2 / variance_d1
    ) / mobility

    return {
        "mobility": mobility,
        "complexity": complexity
    }
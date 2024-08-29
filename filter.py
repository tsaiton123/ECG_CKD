import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the data.

    Args:
        data (np.array): The input signal data (e.g., ECG data).
        lowcut (float): The lower cutoff frequency of the filter (in Hz).
        highcut (float): The higher cutoff frequency of the filter (in Hz).
        fs (float): The sampling rate of the data (in Hz).
        order (int): The order of the filter.

    Returns:
        np.array: The filtered signal.
    """
    # Design the Butterworth bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the data
    y = filtfilt(b, a, data, axis=0)
    return y

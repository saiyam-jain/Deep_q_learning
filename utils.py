import numpy as np

# Normalize the data
def normalize_data(x, data_max, data_min):
    x = (x - data_min)/(data_max-data_min)
    return x

# unnormalize data
def unnormalize(x, data_max, data_min):
    return x * (data_max - data_min) + data_min

# Create sinusoidal data
def create_sinusoidal_data(length=1000, amplitude=1, frequency=0.01, noise=0):
    x = np.arange(length)
    y = amplitude * np.sin(2 * np.pi * frequency * x) + noise * np.random.randn(length)
    return y
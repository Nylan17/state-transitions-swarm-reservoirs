import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from rich.progress import track

def calculate_mc(states, signal, max_delay=50, washout=200, ridge_lambda=1e-3, show_progress=True, poly_degree: int = 1):
    """
    Calculates the memory capacity of a reservoir.

    Args:
        states (np.ndarray): The recorded states of the reservoir.
        signal (np.ndarray): The input signal used to drive the reservoir.
        max_delay (int): The maximum delay to test.
        washout (int): The number of initial timesteps to discard.
        ridge_lambda (float): The regularization strength for the Ridge regressor.

    Returns:
        tuple[float, np.ndarray]: A tuple containing the total memory capacity
                                  and an array of the memory capacity for each delay.
    """
    memories = np.zeros(max_delay)
    
    # Discard washout period
    states = states[washout:]
    signal = signal[washout:]

    iterator = track(range(1, max_delay + 1), description="Calculating MC...") if show_progress else range(1, max_delay + 1)
    for k in iterator:
        # Target is the input signal delayed by k
        y_k = signal[:-k]
        
        # Features are the reservoir states at time t, target is u(t-k)
        # So we need states from t=k to the end
        X_k = states[k:]
        
        if X_k.shape[0] != y_k.shape[0]:
            # This can happen if the signal is shorter than the states
            min_len = min(X_k.shape[0], y_k.shape[0])
            X_k = X_k[:min_len]
            y_k = y_k[:min_len]

        if X_k.shape[0] == 0:
            continue

        x_scaler = StandardScaler()
        X_k_std  = x_scaler.fit_transform(X_k)

        # Optional polynomial expansion (non-linear MC).  For canonical Jaeger
        # memory capacity keep *poly_degree = 1* so the read-out remains linear.
        if poly_degree > 1:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
            X_feats = poly.fit_transform(X_k_std)
        else:
            X_feats = X_k_std

        ridge_alphas = np.logspace(-4, 1, 6)
        model = TransformedTargetRegressor(
            regressor=RidgeCV(alphas=ridge_alphas, fit_intercept=True, scoring="r2"),
            transformer=StandardScaler(with_mean=True, with_std=True),
        )
        model.fit(X_feats, y_k)
        y_pred = model.predict(X_feats)
        
        # MC_k is the squared correlation between predicted and actual signal
        mc_k = r2_score(y_k, y_pred)
        memories[k-1] = max(0, mc_k) # Ensure non-negative

    total_mc = np.sum(memories)
    return total_mc, memories

def generate_smoothed_random_input(timesteps, seed=None, smoothing_factor=0.9, low=-1.0, high=1.0):
    """
    Generates a smoothed random signal for memory capacity tasks.
    """
    rng = np.random.default_rng(seed)
    
    # Generate white noise
    white_noise = rng.uniform(low, high, size=timesteps)
    
    # Apply a smoothing filter (exponential moving average)
    smoothed_signal = np.zeros_like(white_noise)
    smoothed_signal[0] = white_noise[0]
    for t in range(1, timesteps):
        smoothed_signal[t] = smoothing_factor * smoothed_signal[t-1] + (1 - smoothing_factor) * white_noise[t]
        
    return smoothed_signal 
#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the parent directories to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import yaml
from rich.console import Console
from rich.progress import track

from src.boids.multi_state_core import MultiStateBoidFlock
from src.boids.multi_state_features import MULTI_STATE_FEATURES, _single_state_features_list
from src.boids.input_couplers import get_coupler
from src.rc.ridge import train_readout
from src.rc.metrics import nrmse
from src.rc.memory_capacity import calculate_mc

console = Console()

def collect_features(flock, features=None):
    """Extract features from current flock state"""
    # Auto-select feature list based on num_states
    if features is None:
        features = _single_state_features_list() if getattr(flock, "num_states", 2) == 1 else MULTI_STATE_FEATURES
    all_features = []
    for feature_func in features:
        result = feature_func(flock)
        # Convert CuPy arrays/scalars to NumPy first
        if hasattr(result, 'get'):
            result = result.get()

        if hasattr(result, 'shape') and result.shape:  # Array-like
            all_features.extend(result.flatten())
        else:  # Scalar
            all_features.append(float(result))
    return np.array(all_features)

def run_multi_state_experiment(config_path, task='narma10', verbose=True):
    """Run multi-state boid reservoir experiment"""
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if verbose:
        console.print(f"[bold blue]Multi-State Boid Reservoir Experiment[/bold blue]")
        console.print(f"Task: {task}")
        console.print(f"Config: {config_path}")
        console.print(f"N={config['N']}, steps={config['steps']}")
    
    # Load dataset
    if task == 'narma10':
        # Primary dataset location (matching repo layout)
        data_path = Path(__file__).parent.parent / "data" / "narma10.txt"

        # Fallback: top-level *boid_reservoirs/data* folder used by earlier commits
        if not data_path.exists():
            alt_path = Path(__file__).parent.parent.parent / "data" / "narma10.txt"
            if alt_path.exists():
                data_path = alt_path
            else:
                console.print(f"[bold red]Error: NARMA-10 dataset not found.[/bold red]")
                console.print(f"Checked: {data_path} and {alt_path}")
                console.print("Please run `python -m boid_reservoirs.src.scripts.synth_datasets` to generate it.")
                return
            
        data = np.loadtxt(data_path)
        X = data  # only u(t)
        if 'steps' in config and int(config['steps']) < len(X):
            X = X[: int(config['steps'])]
        # The target y(t) needs to be generated based on the input u(t)
        y = np.zeros_like(X)
        for t in range(9, len(X) -1):
             y[t+1] = (0.3 * y[t] + 
                    0.05 * y[t] * np.sum(y[t-9:t+1]) +
                    1.5 * X[t-9] * X[t] + 0.1)

        #  Allow shorter simulations via config['steps'] (e.g. to save RAM)
        max_steps = int(config.get('steps', 0))
        if max_steps and max_steps < len(X):
            X = X[:max_steps]
            y = y[:max_steps]
        
        console.print(f"NARMA-10 dataset: {len(X)} samples loaded from {data_path}")

    elif task == 'memory_capacity':
        # Generate random input for memory capacity
        np.random.seed(config.get('random_seed', 42))
        X = np.random.rand(8000)
        y = None  # MC doesn't use target
        console.print(f"Memory capacity: {len(X)} samples")
    else:
        raise ValueError(f"Unknown task: {task}")
    
    #  Configuration flags for new experiments
    use_leak_features = bool(config.get('use_leak_features', False))
    leak_lambda       = float(config.get('leak_lambda', 0.8))
    whiten_features   = bool(config.get('whiten_features', False))
    input_taps        = int(config.get('input_taps', 0))  # number of recent inputs to append

    if input_taps < 0:
        raise ValueError("input_taps must be non-negative")

    # Initialize multi-state flock and input coupler
    flock = MultiStateBoidFlock(
        N=config['N'],
        box_size=config['box_size'], 
        dt=config['dt'],
        params=config,
        use_gpu=config.get('use_gpu', True),
        seed=config.get('random_seed', 42)
    )
    
    # Get the correct coupler function from the factory
    coupler_func = get_coupler(config.get('input_mode', 'temperature'))
    
    # Collect reservoir states
    n_samples = len(X)
    states = []
    
    # Ring buffer for recent centred inputs (size input_taps)
    if input_taps > 0:
        tap_buf = np.zeros(input_taps, dtype=float)
    
    # Keep a running exponential‐weighted moving average for leaky component
    leak_state = None  # initialised on first timestep if enabled
    
    if verbose:
        iterator = track(range(n_samples), description="Simulating multi-state dynamics...")
    else:
        iterator = range(n_samples)
    
    for i in iterator:
        # Centre input to range ~[-0.5, 0.5] before passing to the coupler.
        centred_val = float(X[i]) - 0.5
        force = coupler_func(flock, centred_val)
        flock.step(force)
        
        # Collect features  
        features_raw = collect_features(flock)

        if use_leak_features:
            if leak_state is None:
                leak_state = features_raw.copy()
            else:
                leak_state = leak_lambda * leak_state + (1.0 - leak_lambda) * features_raw
            features = np.concatenate([features_raw, leak_state])
        else:
            features = features_raw

        if input_taps > 0:
            # Update buffer: roll right and insert current centred_val
            tap_buf = np.roll(tap_buf, 1)
            tap_buf[0] = centred_val  # most recent at index 0
            features = np.concatenate([features, tap_buf])

        #  Safety: replace any NaN / Inf produced by extreme dynamics so that
        #  downstream sklearn regressors do not crash.
        features = np.nan_to_num(features, nan=0.0, posinf=1e9, neginf=-1e9)

        states.append(features)
    
    states = np.array(states)
    
    # Optional whitening (StandardScaler → PCA(whiten=True))
    if whiten_features:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        scaler_feat = StandardScaler()
        states_std = scaler_feat.fit_transform(states)
        pca = PCA(whiten=True)
        states_whiten = pca.fit_transform(states_std)
        states = states_whiten  # replace
    
    if verbose:
        console.print(f"Collected {states.shape[0]} states with {states.shape[1]} features")
        
        # Show some state statistics 
        console.print("\n[bold]Multi-State Feature Analysis:[/bold]")
        feature_names = ['dispersed%', 'clustered%', 'avg_energy', 'energy_var', 
                        'dispersed_speed', 'clustered_speed', 'dispersed_align', 
                        'clustered_align', 'vel_variance', 'state_separation']
        
        for i, name in enumerate(feature_names):
            if i < states.shape[1]:
                mean_val = states[:, i].mean()
                std_val = states[:, i].std()
                console.print(f"  {name}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Apply washout
    washout = config.get('washout', 50)
    states_train = states[washout:]

    #  Instantaneous R² (k = 0) – how much of u(t) is linearly
    #  decodable from the current reservoir state.
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    if task == 'memory_capacity':
        X_train_input = X[washout:]
        model_now = Ridge(alpha=float(config.get('ridge_lambda', 1e-3))).fit(states_train, X_train_input)
        r2_now = r2_score(X_train_input, model_now.predict(states_train))

        if verbose:
            console.print(f"Instantaneous R²: {r2_now:.4f}")
        
    if task == 'narma10':
        y_train = y[washout:]
        
        #  Train / test split (default 70/30) AFTER wash-out.
        train_frac = float(config.get('train_split', 0.7))
        split_idx = int(len(states_train) * train_frac)

        X_tr, X_te = states_train[:-1][:split_idx], states_train[:-1][split_idx:]
        y_tr, y_te = y_train[1:][:split_idx],      y_train[1:][split_idx:]

        lam_val = float(config.get('ridge_alpha', 1e-6))
        readout_type = config.get('readout_type', 'ridge')

        if readout_type == 'kernel_ridge':
            from src.rc.ridge import train_kernel_readout
            gamma_val = config.get('kernel_gamma', None)
            readout_model = train_kernel_readout(X_tr, y_tr, lam=lam_val, gamma=gamma_val)
        else:
            poly_deg = config.get('poly_degree', 1)
            if poly_deg > 1:
                from src.rc.ridge import train_poly_readout
                readout_model = train_poly_readout(X_tr, y_tr, lam=lam_val, degree=poly_deg)
            else:
                readout_model = train_readout(X_tr, y_tr, lam=lam_val)

        # Predict on test split
        y_pred = readout_model.predict(X_te)

        # Metrics ----------------------------------------
        error = nrmse(y_te, y_pred)
        from sklearn.metrics import r2_score
        r2_val = r2_score(y_te, y_pred)
        
        if verbose:
            console.print(f"\n[bold green]NARMA-10 Results:[/bold green]")
            console.print(f"NRMSE: {error:.4f}")
            console.print(f"Target: ≤ 0.40 (competitive)")
        
        return {
            'nrmse': error,
            'r2': r2_val,
            'n_samples_train': len(y_tr),
            'n_samples_test':  len(y_te),
            'states': states,
            'predictions': y_pred,
        }
        
    elif task == 'memory_capacity':
        X_train = X[washout:]
        
        # The states and inputs have already had the *initial* washout removed
        # above, so we pass ``washout=0`` to avoid discarding additional data
        # inside the memory-capacity routine.
        total_mc, mc_curve = calculate_mc(
            states_train,
            X_train,
            washout=0,
            show_progress=verbose,
            poly_degree=int(config.get('poly_degree', 1)),
        )
        
        peak_mc = mc_curve.max()
        peak_delay = mc_curve.argmax() + 1
        
        mc_result = {
            'total_mc': total_mc,
            'peak_mc': peak_mc,
            'peak_delay': peak_delay,
            'mc_curve': mc_curve,
            'r2_now': r2_now,
        }
        
        if verbose:
            console.print(f"\n[bold green]Memory Capacity Results:[/bold green]")
            console.print(f"Total MC: {mc_result['total_mc']:.2f}  |  R²₀: {r2_now:.3f}")
        
        return mc_result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-state boid reservoir experiment')
    parser.add_argument('--config', default='config/multi_state_defaults.yaml',
                       help='Configuration file')
    parser.add_argument('--task', choices=['narma10', 'memory_capacity'], 
                       default='narma10', help='Task to evaluate')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    
    args = parser.parse_args()
    
    result = run_multi_state_experiment(
        config_path=args.config,
        task=args.task, 
        verbose=not args.quiet
    )
    
    # Print final summary
    if args.task == 'narma10':
        print(f"NRMSE: {result['nrmse']:.4f}")
    elif args.task == 'memory_capacity':
        print(f"Total MC: {result['total_mc']:.2f}") 
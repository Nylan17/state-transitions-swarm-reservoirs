# Graceful CuPy import (falls back to NumPy)
import yaml
import numpy as np
try:
    import cupy as cp  # type: ignore
    _cp_available = True
except Exception:
    cp = np  # type: ignore
    _cp_available = False
import matplotlib.pyplot as plt
from pathlib import Path

from ..boids.core import BoidFlock
from ..boids.input_couplers import get_coupler
from ..boids.features import DEFAULT_FEATURES
from ..rc.memory_capacity import calculate_mc, generate_smoothed_random_input
from rich.progress import track

def run_mc_experiment(config_path: str, seed: int = 0, show_progress: bool = True):
    cfg = yaml.safe_load(open(config_path))
    
    # --- Simulation Phase ---
    flock_params = cfg["flock"] if "flock" in cfg else cfg
    flock = BoidFlock(
        N=flock_params["N"],
        box_size=flock_params["box_size"],
        dt=flock_params["dt"],
        params=flock_params,
        use_gpu=cfg["use_gpu"],
        seed=seed
    )
    
    simulation_steps = cfg.get("mc_sim_steps", 6000)
    dataset = generate_smoothed_random_input(simulation_steps, seed=seed)
    
    coupler = get_coupler(cfg["input_mode"])
    states, targets = [], []
    
    numpy = __import__("numpy")
    
    iterator = track(dataset, description="Simulating for MC...") if show_progress else dataset
    for t, u in enumerate(iterator):
        ext_force = coupler(flock, u)
        flock.step(ext_force=ext_force)
        
        feature_values = []
        for f in DEFAULT_FEATURES:
            val = f(flock)
            # Ensure value is an array before processing
            if not hasattr(val, 'ravel'):
                val = numpy.array([val])
            if hasattr(val, 'get'): # Check if it's a cupy array
                val = val.get()
            feature_values.append(val.ravel())
            
        feats = numpy.concatenate(feature_values)
        states.append(feats)

    states = np.vstack(states)
    
    # --- Analysis Phase ---
    # 1. One-step reconstruction (k = 0) – how well does present state encode u(t)?
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    X0 = states
    y0 = dataset.reshape(-1, 1)
    model0 = Ridge(alpha=float(cfg.get('ridge_lambda', 1e-3))).fit(X0, y0)
    r2_now = r2_score(y0, model0.predict(X0))

    # 2. Standard MC curve (k ≥ 1)
    total_mc, memories = calculate_mc(
        states, 
        dataset, 
        max_delay=cfg.get("mc_max_delay", 50),
        washout=cfg.get("mc_washout", 200),
        ridge_lambda=float(cfg["ridge_lambda"]),
        show_progress=show_progress
    )
    
    if show_progress:
        print(f"\nR² at k=0 (instantaneous): {r2_now:.4f}")
        print(f"Total Memory Capacity (k=1…{len(memories)}): {total_mc:.2f}")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(memories) + 1), memories, 'o-')
        plt.title(f'Memory Capacity (Total = {total_mc:.2f})')
        plt.xlabel('Delay (k)')
        plt.ylabel('MC_k (R^2 Score)')
        plt.grid(True)
        
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f"mc_curve_{Path(config_path).stem}_{seed}.png"
        plt.savefig(plot_path)
        print(f"Memory curve plot saved to {plot_path}")
        plt.show()

    return total_mc, memories, r2_now

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_mc <config_path>")
    else:
        config_file = sys.argv[1]
        run_mc_experiment(config_path=config_file, seed=0) 
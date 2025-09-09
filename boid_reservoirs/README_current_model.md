# Current Best Multi-State Boid Reservoir (July 2025)

This document freezes the architecture, parameters and performance of the **small-flock multi-state Boid reservoir computer** that achieves our best trade-off between memory-capacity and NARMA-10 prediction accuracy.

---
## 1  Simulation core
| Item | Value |
|------|-------|
| Number of boids | **50** |
| Arena | 2-D torus, side = 512 units |
| Integration time-step | 0.1 units |
| Steps / washout | 12 000 / 1 000 |
| Velocity noise σ | 0.05 |

### Behavioural states
| Parameter | Dispersed (0) | Clustered (1) |
|-----------|---------------|---------------|
| Perception radius | 28.0 | 35.0 |
| Alignment *k* | 0.5 | 1.0 |
| Cohesion *k*  | 0.5 | 2.0 |
| Separation *k*| 1.2 | 1.0 |

Additional state logic
* local-radius = 35.0
* energy-threshold = 3.0, hysteresis = 0.05

---
## 2  Input coupling
The external scalar signal *u(t)* (Mackey–Glass or NARMA input) is mapped to a temperature multiplier

\[  \gamma(t) = \exp\bigl(k\,\tilde u(t)\bigr) \; , \qquad k = 1.5  \]

where \( \tilde u \in [-0.5,0.5] \) is the standardised input.  The multiplier rescales **target-speed** of every boid each step.

---
## 3  Reservoir state vector (26 D)
Base statistics (13 elements)
1. State proportions [dispersed %, clustered %]
2. Mean local energy
3. Variance of local energy
4. State-specific speed [dispersed, clustered]
5. State-specific alignment [dispersed, clustered]
6. Global velocity variance
7. State separation index (centroid distance)
8. Global alignment
9. Convex-hull area
10. Centroid speed

Temporal mixing
* Exponential moving average  \(\tilde f(t) = \lambda\tilde f(t-1)+(1-\lambda)f(t)\) with **λ = 0.96**.
* Concatenate *f* and *\tilde f* → **26-dimensional** feature vector.

Pre-processing
1. StandardScaler (zero-mean, unit-var).
2. PCA whitening (decorrelated, unit eigen-variance).

---
## 4  Read-out layer
* Polynomial expansion degree = 2 (total 378 terms after bias).
* Ridge regression, α = 1 × 10⁻⁶.

---
## 5  Performance (CPU, no GPU)
| Task | Metric | Result |
|------|--------|--------|
| Memory capacity | Total-MC | **0.86** |
|  | Peak-MC (delay 1) | 0.35 |
| NARMA-10 | NRMSE | **0.617** |

(Generated with `config/best_long.yaml` + `leak_lambda = 0.96`; see figures in `figs/`.)

---
## 6  Reproducibility
1. **Config**  `boid_reservoirs/config/best_long.yaml` (set `leak_lambda` to 0.96).
2. **Training / evaluation**
```bash
python -m boid_reservoirs.src.experiments.run_multi_state \
       --config boid_reservoirs/config/best_long.yaml \
       --task memory_capacity --quiet

python -m boid_reservoirs.src.experiments.run_multi_state \
       --config boid_reservoirs/config/best_long.yaml \
       --task narma10 --quiet
```
3. **Plots** (see `src/scripts/` for helper scripts)
```bash
python -m boid_reservoirs.src.scripts.viz_memory_curve \
       --config boid_reservoirs/config/best_long.yaml \
       --leak_lambda 0.96 --out figs/memory_curve.png
```

---
## 7  Open questions / next steps
* GPU confirmation (set `use_gpu: true`) may improve numerical stability and runtime.
* Explore larger feature set (graph spectral stats, per-boid subsampling) to push MC > 1.
* Evaluate on NARMA-20 and smoothed random inputs to test generalisation. 
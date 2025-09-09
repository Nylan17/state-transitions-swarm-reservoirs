from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

def state_proportions(flock):
    """Return proportion of boids in each behavioural state (length = num_states)."""
    xp = flock.xp
    counts = xp.bincount(flock.states, minlength=getattr(flock, "num_states", 2))
    if hasattr(counts, "get"):
        counts = counts.get()
    return counts / float(flock.N)

def average_local_energy(flock):
    """Average local energy (neighbor density) across all boids"""
    return flock.local_energy.mean()

def local_energy_variance(flock):
    """Variance in local energy - indicates clustering heterogeneity"""
    return flock.local_energy.var()

def state_specific_speed(flock):
    """Average speed for each state (length = num_states)."""
    xp = flock.xp
    speeds = xp.linalg.norm(flock.vel, axis=1)
    vals = []
    for s_id in range(getattr(flock, "num_states", 2)):
        mask = flock.states == s_id
        v = speeds[mask].mean() if xp.any(mask) else 0.0
        if hasattr(v, "get"):
            v = v.get()
        vals.append(float(v))
    return np.array(vals)

def state_specific_alignment(flock):
    """Local alignment per state (length = num_states)."""
    xp = flock.xp
    aligns = []
    for s_id in range(getattr(flock, "num_states", 2)):
        mask = flock.states == s_id
        if xp.any(mask):
            vels = flock.vel[mask]
            mean_vel = vels.mean(axis=0)
            mean_speed = xp.linalg.norm(vels, axis=1).mean()
            a = xp.linalg.norm(mean_vel) / (mean_speed + 1e-9)
            if hasattr(a, "get"):
                a = a.get()
            aligns.append(float(a))
        else:
            aligns.append(0.0)
    return np.array(aligns)

def global_velocity_variance(flock):
    """Overall velocity variance (scalar)"""
    speeds = flock.xp.linalg.norm(flock.vel, axis=1)
    return speeds.var()

def state_separation_index(flock):
    """How well-separated are the two behavioral states spatially?"""
    xp = flock.xp
    positions = flock.pos
    
    dispersed_mask = flock.states == 0
    clustered_mask = flock.states == 1
    
    if not dispersed_mask.any() or not clustered_mask.any():
        return 0.0  # No separation if only one state
    
    dispersed_pos = positions[dispersed_mask]
    clustered_pos = positions[clustered_mask]
    
    # Calculate centroids
    dispersed_centroid = dispersed_pos.mean(axis=0)
    clustered_centroid = clustered_pos.mean(axis=0)
    
    # Distance between centroids
    separation = xp.linalg.norm(dispersed_centroid - clustered_centroid)
    
    if hasattr(separation, 'get'):
        separation = separation.get()
    
    return separation

# For more reservoir observability

def global_alignment(flock):
    """Alignment of the entire flock (magnitude of mean velocity ÷ mean speed)."""
    xp = flock.xp
    speeds = xp.linalg.norm(flock.vel, axis=1)
    mean_vel = flock.vel.mean(axis=0)
    mean_speed = speeds.mean() + 1e-9
    align = xp.linalg.norm(mean_vel) / mean_speed
    return float(align.get() if hasattr(align, 'get') else align)

def convex_hull_area(flock):
    """Area of the convex hull enclosing all boids (CPU calculation)."""
    pts = np.asarray(flock.pos.get() if hasattr(flock.pos, 'get') else flock.pos)
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return float(getattr(hull, 'volume', hull.area))  # 'volume' is area in 2-D
    except Exception:
        # QhullError or numerical degeneracy – return 0 so downstream code survives
        return 0.0

def centroid_speed(flock):
    """Speed of flock centroid (norm of mean velocity)."""
    xp = flock.xp
    centroid_vel = flock.vel.mean(axis=0)
    speed = xp.linalg.norm(centroid_vel)
    return float(speed.get() if hasattr(speed, 'get') else speed)

#  global mean speed (legacy feature)

def global_mean_speed(flock):
    """Mean speed of all boids (scalar)."""
    speeds = flock.xp.linalg.norm(flock.vel, axis=1)
    return float(speeds.mean().get() if hasattr(speeds, 'get') else speeds.mean())

#  density-based features inspired by Nakajima Tetrahymena work

def mean_local_density(flock):
    """Average number of neighbors within *local_radius* (scalar)."""
    return float(flock.local_energy.mean().get() if hasattr(flock.local_energy, 'get') else flock.local_energy.mean())

def local_density_variance(flock):
    """Variance of local neighbor counts (scalar)."""
    return float(flock.local_energy.var().get() if hasattr(flock.local_energy, 'get') else flock.local_energy.var())

#  Helper wrappers to isolate clustered‐state scalars

def clustered_speed(flock):
    """Average speed of boids in clustered state (state id = 1)."""
    return float(state_specific_speed(flock)[1])

def clustered_alignment(flock):
    """Alignment metric for clustered state."""
    return float(state_specific_alignment(flock)[1])

# Core feature set (matches legacy multi-state implementation more closely)
MULTI_STATE_FEATURES = [
    clustered_speed,             # 1D (requires num_states >= 2)
    global_mean_speed,           # 1D
    clustered_alignment,         # 1D (requires num_states >= 2)
    global_alignment,            # 1D
    mean_local_density,          # 1D
    local_density_variance,      # 1D
    convex_hull_area,            # 1D inverse density metric
    state_separation_index,      # 1D state centroid distance
]

# Minimal single-state feature set (no per-state index assumptions)
def _single_state_features_list():
    # Construct after function definitions to avoid forward-reference warnings
    return [
        global_mean_speed,
        global_alignment,
        convex_hull_area_legacy,
        avg_pairwise_distance,
        average_alignment_legacy,
        average_speed_legacy,
        speed_std,
        angular_momentum,
    ]

# Legacy feature parity functions (adapted from swarm_comparison)

def convex_hull_area_legacy(flock):
    """Convex hull area using CPU NumPy (match legacy)."""
    pts = np.asarray(flock.pos.get() if hasattr(flock.pos, 'get') else flock.pos)
    if pts.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # 'volume' is area in 2-D
    except Exception:
        return 0.0


def avg_pairwise_distance(flock):
    """Mean Euclidean distance between all boid pairs."""
    xp = flock.xp
    pos = flock.pos
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= xp.round(diff / flock.box) * flock.box  # periodic wrap
    dist = xp.linalg.norm(diff, axis=2)
    # exclude self-distance (0)
    mean_d = dist.sum() / (flock.N * (flock.N - 1))
    return float(mean_d.get() if hasattr(mean_d, 'get') else mean_d)


def average_alignment_legacy(flock):
    xp = flock.xp
    speeds = xp.linalg.norm(flock.vel, axis=1)
    if xp.any(speeds > 1e-5):
        unit = flock.vel / (speeds[:, None] + 1e-9)
        mean_vel = unit.mean(axis=0)
        a = xp.linalg.norm(mean_vel)
        return float(a.get() if hasattr(a, 'get') else a)
    return 0.0


def average_speed_legacy(flock):
    speeds = flock.xp.linalg.norm(flock.vel, axis=1)
    return float(speeds.mean().get() if hasattr(speeds, 'get') else speeds.mean())


def clustering_ratio(flock):
    xp = flock.xp
    return float((xp.mean(flock.states)).get() if hasattr(flock.states, 'get') else xp.mean(flock.states))


def avg_speed_state0(flock):
    xp = flock.xp
    speeds = xp.linalg.norm(flock.vel, axis=1)
    mask = flock.states == 0
    val = speeds[mask].mean() if xp.any(mask) else 0.0
    return float(val.get() if hasattr(val, 'get') else val)


def avg_speed_state1(flock):
    xp = flock.xp
    speeds = xp.linalg.norm(flock.vel, axis=1)
    mask = flock.states == 1
    val = speeds[mask].mean() if xp.any(mask) else 0.0
    return float(val.get() if hasattr(val, 'get') else val)


def angular_momentum(flock):
    xp = flock.xp
    centroid = flock.pos.mean(axis=0)
    rel_pos = flock.pos - centroid
    cross = rel_pos[:, 0] * flock.vel[:, 1] - rel_pos[:, 1] * flock.vel[:, 0]
    m = cross.mean()
    return float(m.get() if hasattr(m, 'get') else m)


def speed_std(flock):
    speeds = flock.xp.linalg.norm(flock.vel, axis=1)
    return float(speeds.std().get() if hasattr(speeds, 'get') else speeds.std())

# Extend core feature list with legacy parity features

LEGACY_FEATURES = [
    convex_hull_area_legacy,
    avg_pairwise_distance,
    average_alignment_legacy,
    average_speed_legacy,
    clustering_ratio,
    avg_speed_state0,
    avg_speed_state1,
    angular_momentum,
    speed_std,
]

# Update export for multi-state to include legacy parity features by default
MULTI_STATE_FEATURES = MULTI_STATE_FEATURES + LEGACY_FEATURES 

#  High-dimensional per-boid raw features to raise observable capacity

def first_k_boid_speeds(flock, k: int = 100):
    """Return the speeds of the first *k* boids (length = k)."""
    xp = flock.xp
    k = min(k, flock.N)
    speeds = xp.linalg.norm(flock.vel[:k], axis=1)
    return np.asarray(speeds.get() if hasattr(speeds, 'get') else speeds)


def first_k_boid_headings(flock, k: int = 100):
    """Return headings (atan2) of the first *k* boids (length = k)."""
    xp = flock.xp
    k = min(k, flock.N)
    vel = flock.vel[:k]
    headings = xp.arctan2(vel[:, 1], vel[:, 0])
    return np.asarray(headings.get() if hasattr(headings, 'get') else headings)


# Concrete wrappers so they can be placed in the feature list
def boid_speeds_100(flock):
    return first_k_boid_speeds(flock, 100)


def boid_headings_100(flock):
    return first_k_boid_headings(flock, 100)

def boid_speeds_full(flock):
    """Speeds of *all* boids (length = N)."""
    return first_k_boid_speeds(flock, flock.N)


def boid_headings_full(flock):
    """Headings of *all* boids (length = N)."""
    return first_k_boid_headings(flock, flock.N)

# Update exported feature set
MULTI_STATE_FEATURES = MULTI_STATE_FEATURES + [boid_speeds_full, boid_headings_full] 
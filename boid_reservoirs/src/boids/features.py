from scipy.spatial import ConvexHull
import numpy as np
from sklearn.cluster import KMeans

# Number of clusters to segment the swarm into for feature extraction
CLUSTER_COUNT = 4  # can override via monkey-patching before simulation

def avg_vel(flock):          return flock.vel.mean(0)
def centroid(flock):         return flock.pos.mean(0)
def local_order(flock):
    diff, dist2 = flock._pairwise()
    neigh = dist2 < flock.p["outer_r"] ** 2
    alignment = (flock.vel[None, :, :] * neigh[..., None]).sum(1)
    return flock.xp.linalg.norm(alignment, axis=1).mean() / flock.N
def velocity_variance(flock): return flock.vel.var()           # scalar

def compute_elongation(flock):
    """
    Computes the elongation of the swarm.
    """
    xp = flock.xp
    positions = flock.pos
    if positions.shape[0] < 2:
        return 1.0 

    # Handle both numpy and cupy
    if hasattr(positions, 'get'): # cupy
        positions_np = positions.get()
        covariance_matrix = np.cov(positions_np, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    else: # numpy
        covariance_matrix = xp.cov(positions, rowvar=False)
        eigenvalues = xp.linalg.eigvalsh(covariance_matrix)

    eigenvalues = xp.sort(eigenvalues)[::-1]

    if eigenvalues[1] < 1e-9:
        return 1.0 if eigenvalues[0] < 1e-9 else 100.0
    
    elongation_val = xp.sqrt(eigenvalues[0] / eigenvalues[1])
    return elongation_val if xp.isfinite(elongation_val) else 1.0

def compute_convex_hull_area(flock):
    """
    Computes the area of the convex hull of the swarm.
    """
    positions = flock.pos
    if hasattr(positions, 'get'): # cupy
        positions = positions.get()

    if positions.shape[0] > 2:
        try:
            hull = ConvexHull(positions)
            return hull.volume
        except Exception: # QhullError etc
            return 0.0
    return 0.0

def compute_positional_entropy(flock, num_bins_per_dim=10):
    """
    Computes the positional entropy of the boids.
    """
    xp = flock.xp
    positions = flock.pos
    if hasattr(positions, 'get'): # cupy
        positions = positions.get()

    if positions.shape[0] == 0:
        return 0.0

    num_boids = positions.shape[0]
    
    # Use numpy for histogram2d as it's not in cupy
    counts, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=num_bins_per_dim,
        range=[[0, flock.box], [0, flock.box]]
    )
    
    probabilities = counts[counts > 0].flatten() / num_boids
    
    if probabilities.size == 0:
        return 0.0
        
    entropy = -xp.sum(probabilities * xp.log2(probabilities))
    return entropy if xp.isfinite(entropy) else 0.0

def _get_cluster_labels(flock, n_clusters: int = CLUSTER_COUNT):
    """Return K-Means cluster labels for each boid (runs on CPU)."""
    # Always operate on CPU ndarray for sklearn
    pos = flock.pos
    if hasattr(pos, "get"):  # CuPy → NumPy
        pos = pos.get()
    # Guard against pathological tiny swarms (< n_clusters)
    n_clusters = min(n_clusters, pos.shape[0]) if pos.shape[0] > 0 else 1
    # KMeans requires at least 1 sample / cluster
    if n_clusters == 1:
        return np.zeros(pos.shape[0], dtype=int)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(pos)
    return labels


def _cluster_stats(flock, labels, cluster_id):
    """Compute per-cluster statistics mirroring the original DEFAULT_FEATURES.
    Returns a 1-D numpy array of fixed length per cluster (currently 6).
    Layout per cluster: [velocity_variance, convex_hull_area, avg_vel_x, avg_vel_y, elongation, positional_entropy]
    """
    xp = flock.xp
    idx = labels == cluster_id
    # In rare cases KMeans may return empty clusters; handle gracefully
    if not idx.any():
        return np.zeros(6, dtype=float)

    pos = flock.pos[idx]
    vel = flock.vel[idx]

    # velocity variance (scalar)
    vel_var = float(vel.var())

    # convex hull area (scalar)
    cluster_pos_cpu = pos.get() if hasattr(pos, "get") else pos
    try:
        hull_area = ConvexHull(cluster_pos_cpu).volume if cluster_pos_cpu.shape[0] > 2 else 0.0
    except Exception:
        hull_area = 0.0

    # average velocity (2-d)
    avg_v = vel.mean(0)
    avg_v_cpu = avg_v.get() if hasattr(avg_v, "get") else avg_v

    # elongation (scalar)
    positions_cpu = cluster_pos_cpu
    if positions_cpu.shape[0] < 2:
        elong = 1.0
    else:
        cov = np.cov(positions_cpu, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        elong = np.sqrt(eigvals[0] / eigvals[1]) if eigvals[1] > 1e-9 else (100.0 if eigvals[0] > 1e-9 else 1.0)
        if not np.isfinite(elong):
            elong = 1.0

    # positional entropy (scalar)
    num_bins = 10
    counts, _, _ = np.histogram2d(positions_cpu[:, 0], positions_cpu[:, 1], bins=num_bins, range=[[0, flock.box], [0, flock.box]])
    probs = counts[counts > 0].flatten() / positions_cpu.shape[0]
    pos_entropy = float(-(probs * np.log2(probs)).sum()) if probs.size > 0 else 0.0

    return np.array([
        vel_var,
        hull_area,
        avg_v_cpu[0],
        avg_v_cpu[1],
        elong,
        pos_entropy,
    ], dtype=float)


def cluster_features(flock):
    """Return flattened feature vector for the whole swarm, computed per-cluster."""
    labels = _get_cluster_labels(flock, CLUSTER_COUNT)
    feats = []
    for cid in range(CLUSTER_COUNT):
        feats.append(_cluster_stats(flock, labels, cid))
    return np.concatenate(feats)

# NEW Feature Set
DEFAULT_FEATURES = [
    cluster_features
] 
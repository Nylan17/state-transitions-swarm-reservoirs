import numpy as _np

# Attempt CuPy import but allow fallback
try:
    import cupy as _cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:
    _cp = None  # type: ignore
    _CUPY_AVAILABLE = False

class BoidFlock:
    def __init__(self, N, box_size, dt, params, use_gpu=True, seed=None):
        self.xp = _cp if (use_gpu and _CUPY_AVAILABLE) else _np
        self.rng = self.xp.random.default_rng(seed)
        self.N = N
        self.box = box_size
        self.dt = dt
        self.p = params
        self.pos = self.rng.uniform(0, box_size, size=(N, 2))
        theta = self.rng.uniform(0, 2 * self.xp.pi, N)
        self.vel = self.xp.stack([self.xp.cos(theta), self.xp.sin(theta)], axis=1) * self.p["init_speed"]

    def _pairwise(self):
        diff = self.pos[:, None, :] - self.pos[None, :, :]   # (N,N,2)
        diff -= self.xp.round(diff / self.box) * self.box         # torus wrap
        dist2 = (diff ** 2).sum(-1)
        return diff, dist2

    def step(self, ext_force=None):
        diff, dist2 = self._pairwise()
        within_outer = dist2 < self.p["outer_r"] ** 2
        within_inner = dist2 < self.p["inner_r"] ** 2
        # Separation
        sep = -(diff * within_inner[..., None]).sum(1)
        # Alignment
        align = (self.vel[None, :, :] * within_outer[..., None]).sum(1)
        # Cohesion
        cohesion = (-diff * within_outer[..., None]).sum(1)
        acc = (self.p["sep_k"] * sep +
               self.p["align_k"] * align +
               self.p["coh_k"] * cohesion)
        if ext_force is not None:
            acc += ext_force
        self.vel += acc * self.dt
        speed = self.xp.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = self.vel / speed * self.p["target_speed"]
        self.pos = (self.pos + self.vel * self.dt) % self.box 
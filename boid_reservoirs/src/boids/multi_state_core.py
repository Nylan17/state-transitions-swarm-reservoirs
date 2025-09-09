"""Multi-state boid flock dynamics supporting CPU (NumPy) and GPU (CuPy) backends.

The module attempts to import CuPy, but will gracefully fall back to NumPy if
CuPy is unavailable or if the user disables GPU usage via the ``use_gpu``
constructor argument.  All array operations throughout the class are executed
through the attribute ``self.xp`` which is set to either ``cupy`` or
``numpy`` at runtime.
"""

import numpy as _np

try:
    import cupy as _cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover – covers both ImportError and CUDA errors
    _cp = None  # Fallback placeholder so that references don't NameError
    _CUPY_AVAILABLE = False

class MultiStateBoidFlock:
    def __init__(self, N, box_size, dt, params, use_gpu=True, seed=None):
        # Gracefully degrade to CPU if CUDA runtime is unavailable *after* the
        # CuPy import succeeded (common on machines without an NVIDIA driver).
        if use_gpu and _CUPY_AVAILABLE:
            try:
                # Lightweight runtime capability check: allocate a tiny array.
                _ = _cp.zeros(1)
                self.xp = _cp
            except Exception:  # pragma: no cover – fallback path when CUDA unavailable
                self.xp = _np
        else:
            self.xp = _np

        self.rng = self.xp.random.default_rng(seed)
        self.N = N
        self.box = box_size
        self.dt = dt
        self.p = params
        # number of behavioural states (2 = original dispersed/clustered)
        self.num_states = int(params.get("num_states", 2))
        
        # Initialize positions and velocities
        self.pos = self.rng.uniform(0, box_size, size=(N, 2))
        theta = self.rng.uniform(0, 2 * self.xp.pi, N)
        self.vel = self.xp.stack([self.xp.cos(theta), self.xp.sin(theta)], axis=1) * self.p["init_speed"]
        
        # Multi-state specific: each boid has a state (0=dispersed, 1=clustered)
        self.states = self.xp.zeros(N, dtype=int)  # Start all dispersed
        self.local_energy = self.xp.zeros(N)
        
        # Target speed modulation (for temperature coupling)
        self.target_speed = self.p.get("target_speed", 1.0)

        # Per-step multiplicative speed scaling (set by input coupler such as
        # `speed_scale`).  Default to 1.0 so behaviour is unchanged unless a
        # coupler actively updates the value.
        self.speed_scale_factor = 1.0

        # Per-step multiplicative perception-radius scaling (for density input
        # coupling).  Factor applied inside `_get_state_params`.
        self.radius_scale_factor = 1.0

        # Optional velocity inertia (0 < α ≤ 1).  α = 1 → original dynamics
        self.inertia_alpha = float(self.p.get("inertia_alpha", 1.0))

    def _pairwise(self):
        diff = self.pos[:, None, :] - self.pos[None, :, :]   # (N,N,2)
        diff -= self.xp.round(diff / self.box) * self.box         # torus wrap
        dist2 = (diff ** 2).sum(-1)
        return diff, dist2

    def _compute_local_energy(self):
        """Compute local energy as *average neighbour speed* (legacy)."""
        diff, dist2 = self._pairwise()
        local_r = float(self.p.get("local_radius", 35.0))
        speeds = self.xp.linalg.norm(self.vel, axis=1)  # (N,)

        # Vectorised neighbour mask (N,N) – True where j is within radius of i (exclude self)
        nbr_mask = (dist2 < local_r ** 2) & (dist2 > 1e-12)

        # Count neighbours and sum their speeds
        counts = nbr_mask.sum(axis=1)  # (N,)
        sumspd = (nbr_mask * speeds).sum(axis=1)  # broadcasting speeds to (N,N)

        # Where a boid has no neighbours, fall back to its own speed
        le = self.xp.where(counts > 0, sumspd / counts, speeds)
        self.local_energy = le

    def _update_states(self):
        """Update boid states based on local energy and hysteresis

        For num_states == 2 behaviour unchanged.  For 3 states we use two
        threshold bands (ρ₀₁ and ρ₁₂).  Parameters expected in ``self.p``::

            rho01_up,   rho01_down,
            rho12_up,   rho12_down
        """

        if self.num_states == 2:
            energy_threshold = self.p.get("energy_threshold", 0.8)  # thresholds now in speed units
            h = self.p.get("hysteresis_factor", 0.25)
            if self.p.get("use_hysteresis", True):
                up = energy_threshold * (1 + h)
                down = energy_threshold * (1 - h)
            else:
                up = down = energy_threshold

            to1 = (self.states == 0) & (self.local_energy > up)
            to0 = (self.states == 1) & (self.local_energy < down)
            self.states[to1] = 1
            self.states[to0] = 0
            return

        # ---- 3-state logic ----
        rho01_up   = self.p.get("rho01_up", 2.0)
        rho01_down = self.p.get("rho01_down", 1.0)
        rho12_up   = self.p.get("rho12_up", 6.0)
        rho12_down = self.p.get("rho12_down", 4.0)

        st = self.states  # alias

        # 0 -> 1
        st[(st == 0) & (self.local_energy > rho01_up)] = 1
        # 1 -> 0
        st[(st == 1) & (self.local_energy < rho01_down)] = 0
        # 1 -> 2
        st[(st == 1) & (self.local_energy > rho12_up)] = 2
        # 2 -> 1
        st[(st == 2) & (self.local_energy < rho12_down)] = 1

    def _get_state_params(self, state):
        """Get parameters for a specific behavioral state"""
        if state == 0:  # Dispersed state
            return {
                'perception_radius': self.p.get("dispersed_outer_r", 28.0) * getattr(self, 'radius_scale_factor', 1.0),
                'align_k': self.p.get("dispersed_align_k", 0.0),
                'coh_k': self.p.get("dispersed_coh_k", 0.0), 
                'sep_k': self.p.get("dispersed_sep_k", 1.8)
            }
        elif state == 1 and self.num_states == 3:  # Semi-clustered (new)
            return {
                'perception_radius': self.p.get("semi_outer_r", 32.0) * getattr(self, 'radius_scale_factor', 1.0),
                'align_k': self.p.get("semi_align_k", 0.75),
                'coh_k':   self.p.get("semi_coh_k",   1.2),
                'sep_k':  self.p.get("semi_sep_k",  1.1),
            }
        else:  # Clustered state (id 1 in 2-state, id 2 in 3-state)
            return {
                'perception_radius': self.p.get("clustered_outer_r", 35.0) * getattr(self, 'radius_scale_factor', 1.0),
                'align_k': self.p.get("clustered_align_k", 1.5),
                'coh_k': self.p.get("clustered_coh_k", 2.0),
                'sep_k': self.p.get("clustered_sep_k", 1.0)
            }

    def step(self, ext_force=None):
        # Update local energy and states first
        self._compute_local_energy()
        self._update_states()
        
        # Apply forces based on current state
        diff, dist2 = self._pairwise()
        
        acc = self.xp.zeros_like(self.vel)
        
        #  Legacy‐style force calculation (per boid)
        inner_r = float(self.p.get("inner_r", 3.0))
        for i in range(self.N):
            s_id = int(self.states[i])
            params_i = self._get_state_params(s_id)

            # Compute neighbour masks for this focal boid
            dist_i = self.xp.sqrt(dist2[i])
            mask_outer = (dist_i > 1e-9) & (dist_i < params_i['perception_radius'])
            nbr_cnt = int(mask_outer.sum())

            if nbr_cnt == 0:
                continue  # no neighbours → no social forces

            # Alignment
            align_vec = self.vel[mask_outer].mean(axis=0) - self.vel[i]

            # Cohesion
            cohesion_vec = self.pos[mask_outer].mean(axis=0) - self.pos[i]

            # Separation
            mask_inner = dist_i < inner_r
            sep_vec = self.xp.zeros(2)
            if mask_inner.any():
                sep_vec = (self.pos[i] - self.pos[mask_inner]).sum(axis=0)

            acc[i] = (params_i['sep_k'] * sep_vec +
                       params_i['align_k'] * align_vec +
                       params_i['coh_k'] * cohesion_vec)

        #  Anchoring term towards centre of arena (matches legacy behaviour)
        anchor_k = float(self.p.get("anchoring_weight", 0.01))
        if anchor_k != 0.0:
            centre_vec = self.xp.array([self.box / 2.0, self.box / 2.0])
            acc += (centre_vec - self.pos) * anchor_k

        # Add external forces
        if ext_force is not None:
            acc += ext_force
            

        alpha = getattr(self, "inertia_alpha", 1.0)
        if alpha >= 0.999:
            self.vel += acc * self.dt
        else:
            self.vel = (1.0 - alpha) * self.vel + alpha * (self.vel + acc * self.dt)

        # Apply multiplicative scaling *before* any speed normalisation so the
        # injected signal is preserved, matching the legacy implementation.
        if getattr(self, "speed_scale_factor", 1.0) != 1.0:
            self.vel *= float(self.speed_scale_factor)
        # Reset for next step so stale value is not reused accidentally.
        self.speed_scale_factor = 1.0

        # Optional Gaussian velocity noise to enhance dynamical richness
        noise_std = self.p.get("noise_std", 0.0)
        if noise_std > 0.0:
            noise = self.rng.standard_normal(self.vel.shape) * noise_std
            self.vel += noise

        #  Soft velocity normalisation
        speed = self.xp.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-9  # avoid div/0
        relax = float(self.p.get("speed_relax", 1.0))  # 1.0 = hard reset (legacy)

        if relax >= 0.999:  # keep original behaviour for speed
            self.vel = self.vel / speed * self.target_speed
        else:
            desired = self.vel / speed * self.target_speed  # direction preserved
            self.vel = self.vel * (1.0 - relax) + desired * relax

        # Clip speed to avoid numeric blow-up when relaxation is low
        clip = float(self.p.get("speed_clip", 3.0 * self.target_speed))
        if clip > 0:
            sp = self.xp.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-9
            scale = self.xp.minimum(1.0, clip / sp)
            self.vel *= scale

        # Reset radius scaling factor likewise
        self.radius_scale_factor = 1.0

        self.pos = (self.pos + self.vel * self.dt) % self.box 
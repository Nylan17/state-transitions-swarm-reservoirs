def temperature_modulation(flock, value):
    """Scale the flock's target speed as a function of the scalar *value*.

    The original single–state implementation expected the configuration to
    contain the keys ``base_speed`` and ``temp_gain``.  The newer multi–state
    configuration, however, uses ``target_speed`` (or ``init_speed``) and
    ``temp_k`` instead.  To ease transition we support **both** naming schemes
    and fall back to sensible defaults if none are provided.
    """

    base_speed = flock.p.get("base_speed", flock.p.get("target_speed", 1.0))
    # Mapping mode: 'linear' (legacy) or 'exp'.
    mapping_mode = flock.p.get("temp_mapping", "linear").lower()
    gain        = flock.p.get("temp_gain", flock.p.get("temp_k", 0.1))

    if mapping_mode == "exp":
        # Map standardised input value *value*∈[−0.5,0.5] to γ via an exponential.
        # The config provides *gain* as k in γ = base_speed·exp(k·value).
        gamma = base_speed * float(_safe_exp(gain * value))
    else:  # linear (default)
        gamma = base_speed * (1.0 + gain * value)

    # Clip γ to reasonable bounds to avoid numerical blow-ups.
    gamma = max(0.1 * base_speed, min(5.0 * base_speed, gamma))

    flock.target_speed = gamma
    return None

def predator_agent(flock, value):
    # map 1-D input to predator's x-velocity; y is 0
    xp = flock.xp
    pred_force = xp.zeros_like(flock.pos)
    pred_force[0] = xp.array([value, 0.0])    # boid 0 is predator
    return pred_force

def acceleration_injection(flock, value):
    """Inject scalar *value* as x-axis acceleration for all boids.

    This mode is intended for memory-capacity benchmarks: the input becomes a
    direct additive term in the equations of motion rather than a speed
    rescaling, therefore preserving linear information about past inputs.
    """
    xp = flock.xp
    acc = xp.zeros_like(flock.vel)
    acc[:, 0] = float(value)  # add to x-direction
    return acc

def speed_scale(flock, value):
    """Scale every boid's velocity magnitude by a factor (1 + gain·value).

    This mode replicates the simple multiplicative temperature coupling used
    in the original *swarm_comparison* implementation: the centred input
    drives *γ(t) = 1 + k·u~(t)* which directly scales the velocity vector.
    The mapping is linear to maintain a faithful linear echo of past inputs.
    """
    # Centre input value ∈[−0.5, 0.5] → multiplier γ ∈[0.5, 1.5]
    scale = 1.0 + float(value)

    # Clip to avoid runaway blow-ups if inputs go out of range.
    scale = max(0.1, min(5.0, scale))

    flock.speed_scale_factor = scale
    return None

def radius_scale(flock, value):
    """Scale perception radii by (1 + value).

    With centred *value* ∈ [−0.5,0.5] this yields a 0.5× to 1.5× variation,
    directly affecting neighbour density.
    """
    scale = 1.0 + float(value)
    scale = max(0.5, min(1.5, scale))
    flock.radius_scale_factor = scale
    return None

def mixed_scale(flock, value):
    """Combine speed_scale and radius_scale in one call."""
    speed_scale(flock, value)
    radius_scale(flock, value)
    return None

def get_coupler(name: str):
    if name == "temperature":
        return temperature_modulation
    elif name == "predator":
        return predator_agent
    elif name in ("accel", "acceleration"):
        return acceleration_injection
    elif name in ("scale", "speed_scale", "mult"):
        return speed_scale
    elif name in ("radius", "radius_scale", "density"):
        return radius_scale
    elif name in ("mixed", "combo"):
        return mixed_scale
    else:
        raise ValueError(f"Unknown input_mode: {name}")

#  safe exponential with overflow protection

def _safe_exp(x: float, limit: float = 60.0):
    """exp(x) but truncate extreme arguments to avoid over/under-flow."""
    import math
    if x > limit:
        return math.exp(limit)
    if x < -limit:
        return math.exp(-limit)
    return math.exp(x) 
import torch
from fmai.robustness import Pred, Defined, And, Or, Not, WeightedRobustness

TAU1, TAU2 = 2, 2
TAU_FLOCK = 4  # Recovery horizon tau for flocking (matches W = TAU1 + TAU2 = 4)

def dgoal(obs):                      
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1)

def dlidar(obs):
    return obs[..., 6:18].amax(dim=-1)

def lag(x, s, seg):
    """x[t - s] for x: [E,T,A]; NaN if t < s or t - s lies in an earlier episode. seg: [E,T]."""
    if s == 0:
        return x
    nan = float("nan")
    out = torch.full_like(x, nan)
    out[:, s:] = x[:, :-s]
    same = torch.zeros_like(seg, dtype=torch.bool)
    same[:, s:] = seg[:, s:] == seg[:, :-s]
    return torch.where(same[..., None], out, torch.full_like(out, nan))

# --- Navigation signals ---
def vgoal(s):
    """Goal progress: dgoal(t-s) - dgoal(t) > 0 (distance to goal is decreasing)"""
    return lambda o: lag(dgoal(o), s, o[..., 0, -1]) - dgoal(o)

def shift_dlidar(s):
    """dlidar(t-s): LiDAR reading shifted back by s steps"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1])

def vlidar(s):
    """Separation progress: dlidar(t-s) - dlidar(t) > 0 (LiDAR proximity is decreasing)"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1]) - dlidar(o)

# --- Flocking signals (requires obs layout: [pos(0:2), vel(2:4), rel_pos(4:6), lidar(6:18), rel_vel(18:20), ep_id(-1)]) ---
def vabs(obs):
    """Absolute agent speed: ||v_a(t)||"""
    return torch.linalg.vector_norm(obs[..., 2:4], dim=-1)

def dband_err(obs, r_t=0.30):
    """Orbital radius error: | ||p_a(t) - p_t(t)|| - r_t |"""
    return (dgoal(obs) - r_t).abs()

def shift_dband_err(s, r_t=0.30):
    """dband_err(t - s) shifted back by s steps"""
    return lambda o: lag(dband_err(o, r_t), s, o[..., 0, -1])

def vtan(obs):
    """Signed counter-clockwise tangential orbital speed around target: (r_x * v_rel_y - r_y * v_rel_x) / ||r(t)||"""
    rel_pos = obs[..., 4:6]
    rel_vel = obs[..., 18:20]
    dist = torch.linalg.vector_norm(rel_pos, dim=-1).clamp(min=1e-6)
    return (rel_pos[..., 0] * rel_vel[..., 1] - rel_pos[..., 1] * rel_vel[..., 0]) / dist

def vrel_err(obs, v_r=0.20):
    """Tangential speed error: | v_tan(t) - v_r |"""
    return (vtan(obs) - v_r).abs()

def shift_vrel_err(s, v_r=0.20):
    """vrel_err(t - s) shifted back by s steps"""
    return lambda o: lag(vrel_err(o, v_r), s, o[..., 0, -1])


def build_nav_spec():
    """
    CNF Specification for navigation environment:
    1. Goal progress when safely spaced: ~da(t-TAU1) | ~da(t) | dg(t) | vg(1..TAU1)
    2. Spacing recovery or active avoidance: da(t) | da(t-1..t-TAU2) | va(1..TAU2)
    3. Hard safety (never critical distance): dap(t)
    """
    dg   = Pred(dgoal, "<=", 0.1)                        # At goal
    da   = Pred(dlidar, "<", 0.2)                        # Well-spaced now
    dap  = Pred(dlidar, "<", 0.8)                        # Never in critical distance
    sda1 = Pred(shift_dlidar(TAU1), "<", 0.2)            # Well-spaced TAU1 steps ago

    # Progress toward goal over [1, TAU1]
    vg = [Defined(Pred(vgoal(s), ">", 0.0)) for s in range(1, 1 + TAU1)]

    # Well-spaced in the past [1, TAU2] OR actively increasing separation over [1, TAU2]
    past_da = [Defined(Pred(shift_dlidar(s), "<", 0.2)) for s in range(1, 1 + TAU2)]
    va      = [Defined(Pred(vlidar(s), ">", 0.0)) for s in range(1, 1 + TAU2)]

    c_goal_prog = Or(Defined(Not(sda1)), Not(da), dg, *vg)
    c_space_rec = Or(da, *past_da, *va)
    phi = And(c_goal_prog, c_space_rec, dap)

    desc = [
        r"$\forall t: (\neg da(t-\tau_1) \vee \neg da(t) \vee d_g(t) \vee v_g(t)) "
        r"\wedge (\bigvee_{s=0}^{\tau_2} da(t-s) \vee \bigvee_{s=1}^{\tau_2} v_a(t,s)) "
        r"\wedge d_a^{+}(t)$"
    ]
    spec = WeightedRobustness([phi], weights=None, clip=1.0)
    spec.sub_formulas = {
        "goal_prog": c_goal_prog,
        "space_rec": c_space_rec,
        "hard_safe": dap,
    }
    return spec, desc


def build_floc_spec(cfg=None, tau=TAU_FLOCK):
    """
    Normalized SFO Specification for multi-agent flocking (all predicates scaled so ideal = +1.0, threshold = 0.0):
        phi_flock(t) = phi_c(t) & phi_v^m(t) & phi_s^d(t) & phi_s^v(t)
    """
    tau   = cfg.TAU_FLOCK if cfg is not None else tau
    r_t   = cfg.TARGET_RADIUS if cfg is not None else 0.30
    eps_t = cfg.EPSILON_T if cfg is not None else 0.055
    p_t   = cfg.MIN_PROXIMITY if cfg is not None else 0.50
    v_r   = cfg.TARGET_REL_SPEED if cfg is not None else 0.20
    eps_v = cfg.EPSILON_V if cfg is not None else 0.05
    v_max = cfg.V_MAX if cfg is not None else 0.55

    # 1. Normalized strict invariants: ideal = +1.0, boundary = 0.0
    phi_c   = Pred(lambda o: dlidar(o) / p_t, "<", 1.0)
    phi_v_m = Pred(lambda o: vabs(o) / v_max, "<", 1.0)

    # 2. Normalized bounded-time recovery for orbital distance band: ideal = +1.0, edge = 0.0
    phi_d_lags = [
        Pred(lambda o: dband_err(o, r_t) / eps_t, "<=", 1.0) if i == 0
        else Defined(Pred(lambda o, s=i: shift_dband_err(s, r_t)(o) / eps_t, "<=", 1.0))
        for i in range(tau + 1)
    ]
    phi_s_d = Or(*phi_d_lags)

    # 3. Normalized bounded-time recovery for tangential velocity band: ideal = +1.0, edge = 0.0
    phi_v_lags = [
        Pred(lambda o: vrel_err(o, v_r) / eps_v, "<=", 1.0) if i == 0
        else Defined(Pred(lambda o, s=i: shift_vrel_err(s, v_r)(o) / eps_v, "<=", 1.0))
        for i in range(tau + 1)
    ]
    phi_s_v = Or(*phi_v_lags)

    # Complete normalized formula: phi_flock(t) in (-inf, 1.0], satisfied iff >= 0.0
    phi_flock = And(phi_c, phi_v_m, phi_s_d, phi_s_v)

    desc = [
        r"$\tilde{\varphi}_{\text{flock}}(t) = \tilde{\varphi}_c(t) \wedge \tilde{\varphi}_v^m(t) "
        r"\wedge \left(\bigvee_{i=0}^{\tau} \tilde{\varphi}_d(t-i)\right) "
        r"\wedge \left(\bigvee_{i=0}^{\tau} \tilde{\varphi}_v^a(t-i)\right)$"
    ]
    spec = WeightedRobustness([phi_flock], weights=None, clip=2.0)
    spec.sub_formulas = {
        "phi_c (collision)": phi_c,
        "phi_vm (v_max)": phi_v_m,
        "phi_sd (dist_band)": phi_s_d,
        "phi_sv (tan_speed)": phi_s_v,
    }
    return spec, desc


def build_spec(cfg=None):
    scenario = cfg.SCENARIO if cfg is not None else "navigation"
    if scenario == "navigation":
        return build_nav_spec()
    elif scenario == "flocking":
        return build_floc_spec(cfg)
    raise ValueError(f"Unknown scenario: {scenario}")
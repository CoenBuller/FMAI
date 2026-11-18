import torch
from fmai.robustness import Pred, Defined, And, Or, Not, WeightedRobustness

TAU1, TAU2 = 1, 2

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

def vgoal(s):
    """Goal progress: dgoal(t-s) - dgoal(t) > 0 (distance to goal is decreasing)"""
    return lambda o: lag(dgoal(o), s, o[..., 0, -1]) - dgoal(o)

def shift_dlidar(s):
    """dlidar(t-s): LiDAR reading shifted back by s steps"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1])

def vlidar(s):
    """Separation progress: dlidar(t-s) - dlidar(t) > 0 (LiDAR proximity is decreasing)"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1]) - dlidar(o)


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
    sda1 = Pred(shift_dlidar(TAU1), "<", 0.2)                # Well-spaced TAU1 steps ago

    # Progress toward goal over [1, TAU1]
    vg = [Defined(Pred(vgoal(s), ">", 0.0)) for s in range(1, 1 + TAU1)]

    # Well-spaced in the past [1, TAU2] OR actively increasing separation over [1, TAU2]
    past_da = [Defined(Pred(shift_dlidar(s), "<", 0.2)) for s in range(1, 1 + TAU2)]
    va      = [Defined(Pred(vlidar(s), ">", 0.0)) for s in range(1, 1 + TAU2)]

    phi = And(
        Or(Defined(Not(sda1)), Not(da), dg, *vg),
        Or(da, *past_da, *va),
        dap,
    )

    desc = [
        r"$\forall t: (\neg da(t-\tau_1) \vee \neg da(t) \vee d_g(t) \vee v_g(t)) "
        r"\wedge (\bigvee_{s=0}^{\tau_2} da(t-s) \vee \bigvee_{s=1}^{\tau_2} v_a(t,s)) "
        r"\wedge d_a^{+}(t)$"
    ]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc

def dband_err(obs, r_t=0.3):
    """Distance deviation from the target orbit radius r_t: | ||pos_a - pos_t|| - r_t |"""
    return (dgoal(obs) - r_t).abs()

def vband(s, r_t=0.3):
    """Band progress: dband_err(t-s) - dband_err(t) > 0 (moving closer to target radius r_t)"""
    return lambda o: lag(dband_err(o, r_t), s, o[..., 0, -1]) - dband_err(o, r_t)

def vrel_err(obs, target_rel_speed=0.25):
    """Deviation from steady relative speed: | ||v_a - v_t|| - target_rel_speed |"""
    rel_speed = torch.linalg.vector_norm(obs[..., 18:20], dim=-1)
    return (rel_speed - target_rel_speed).abs()

def vabs(obs):
    """Absolute agent speed: ||v_a||"""
    return torch.linalg.vector_norm(obs[..., 2:4], dim=-1)


def build_floc_spec(r_t=0.3, eps_t=0.05, target_rel_speed=0.25, eps_v=0.05, v_max=0.5):
    """
    CNF Specification for flocking environment:
    1. Band & steady relative speed maintenance/progress when safely spaced
    2. Spacing recovery or active avoidance: da(t) | da(t-1..t-TAU2) | va(1..TAU2)
    3. Hard safety (never critical proximity < 0.8 and never overspeeding <= v_max)
    """
    in_band  = Pred(lambda o: dband_err(o, r_t), "<=", eps_t)                  # Inside target distance band
    in_vrel  = Pred(lambda o: vrel_err(o, target_rel_speed), "<=", eps_v)      # Maintaining steady relative velocity
    dg       = And(in_band, in_vrel)                                           # Goal state: in band AND steady rel speed

    da       = Pred(dlidar, "<", 0.2)                                          # Well-spaced now (proximity < 0.2)
    dap      = Pred(dlidar, "<", 0.8)                                          # Never in critical proximity (< 0.8)
    vmax_ok  = Pred(vabs, "<=", v_max)                                         # Staying under absolute max velocity
    sda1     = Pred(shift_dlidar(TAU1), "<", 0.2)                              # Well-spaced TAU1 steps ago

    # Progress toward the target band over [1, TAU1]
    vg = [Defined(Pred(vband(s, r_t), ">", 0.0)) for s in range(1, 1 + TAU1)]

    # Well-spaced in the past [1, TAU2] OR actively reducing proximity over [1, TAU2]
    past_da = [Defined(Pred(shift_dlidar(s), "<", 0.2)) for s in range(1, 1 + TAU2)]
    va      = [Defined(Pred(vlidar(s), ">", 0.0)) for s in range(1, 1 + TAU2)]

    phi = And(
        Or(Defined(Not(sda1)), Not(da), dg, *vg),
        Or(da, *past_da, *va),
        dap,
        vmax_ok,
    )

    desc = [
        r"$\forall t: (\neg da(t-\tau_1) \vee \neg da(t) \vee (d_{\text{band}}(t) \wedge v_{\text{rel}}(t)) \vee v_{\text{band}}(t)) "
        r"\wedge (\bigvee_{s=0}^{\tau_2} da(t-s) \vee \bigvee_{s=1}^{\tau_2} v_a(t,s)) "
        r"\wedge d_a^{+}(t) \wedge v_{\max}(t)$"
    ]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc


def build_spec(scenario: str = "flocking"):
    if scenario == "navigation":
        return build_nav_spec()
    elif scenario == "flocking":
        return build_floc_spec()
    raise ValueError(f"Unknown scenario: {scenario}")

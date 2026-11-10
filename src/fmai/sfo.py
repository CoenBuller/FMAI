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

def shift_da(s):
    """dlidar(t-s): LiDAR reading shifted back by s steps"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1])

def vagent(s):
    """Separation progress: dlidar(t-s) - dlidar(t) > 0 (LiDAR proximity is decreasing)"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1]) - dlidar(o)


def build_spec():
    """
    CNF Specification:
    1. Goal progress when safely spaced: ~da(t-TAU1) | ~da(t) | dg(t) | vg(1..TAU1)
    2. Spacing recovery or active avoidance: da(t) | da(t-1..t-TAU2) | va(1..TAU2)
    3. Hard safety (never critical distance): dap(t)
    """
    dg   = Pred(dgoal, "<=", 0.1)                        # At goal
    da   = Pred(dlidar, "<", 0.2)                        # Well-spaced now
    dap  = Pred(dlidar, "<", 0.8)                        # Never in critical distance
    sda1 = Pred(shift_da(TAU1), "<", 0.2)                # Well-spaced TAU1 steps ago

    # Progress toward goal over [1, TAU1]
    vg = [Defined(Pred(vgoal(s), ">", 0.0)) for s in range(1, 1 + TAU1)]

    # Well-spaced in the past [1, TAU2] OR actively increasing separation over [1, TAU2]
    past_da = [Defined(Pred(shift_da(s), "<", 0.2)) for s in range(1, 1 + TAU2)]
    va      = [Defined(Pred(vagent(s), ">", 0.0)) for s in range(1, 1 + TAU2)]

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
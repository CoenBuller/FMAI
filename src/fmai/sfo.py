import torch
from fmai.robustness import Pred, Defined, And, Or, Not, WeightedRobustness

TAU1, TAU2 = 1, 2

# Helper function to get the maximum euclidean distance to the goal between agents
def dgoal(obs):                      
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1)

def dlidar(obs):
    return obs[..., 6:18].amax(dim=-1)

def lag(x, s, seg):
    """x[t - s] for x: [E,T,A]; NaN if t < s or t - s lies in an earlier episode. seg: [E,T]."""
    nan = float("nan")
    out = torch.full_like(x, nan)
    out[:, s:] = x[:, :-s]
    same = torch.zeros_like(seg, dtype=torch.bool)
    same[:, s:] = seg[:, s:] == seg[:, :-s]
    return torch.where(same[..., None], out, torch.full_like(out, nan))

def vgoal(s):
    """dg(t-s) - dg(t) >= V*s   (margin: gain - V*s)"""
    gain = lambda o: lag(dgoal(o), s, o[..., 0, -1]) - dgoal(o)
    return gain

def shift_da(s):
    """da(t-s): returns a shifted callable of da(t), which shifts it back by s """
    return lambda o: lag(dlidar(o), s, o[..., 0, -1])

def vagent(s):
    gain = lambda o: lag(dlidar(o), s, o[..., 0, -1]) - dlidar(o)
    return gain



def build_spec():
    """At every step: (at goal  OR  progressed over some window s<=W)  AND  separated from all other agents."""
    
    # Predicates 
    dg = Pred(dgoal, "<=", 0.1)
    vg = [Defined(Pred(vgoal(s), ">", 0.0)) for s in range(1, 1+TAU1)]
    da = Pred(dlidar, "<", 0.2)
    sda2 = Pred(shift_da(TAU2), "<", 0.2)
    sda1 = Pred(shift_da(TAU1+TAU2), "<", 0.2)
    dap = Pred(dlidar, "<", 0.8)
    # SFO formula in CNF form
    phi = And(
        Or(Not(sda1), dg, *vg),
        Or(sda2, da),
        dap,
    )

    desc = [r"$\forall t: (\neg da(t-\tau_1) \vee v_g(t) \vee d_g(t))\wedge(da(t-\tau_1 -\tau_2) \vee  da(t)) \wedge d_a^{+}(t)$"]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc

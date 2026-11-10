import torch
from fmai.robustness import Pred, Defined, And, Or, Not, WeightedRobustness

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

def vagent(s):
    gain = lambda o: lag(dlidar(o), s, o[..., 0, -1]) - dlidar(o)
    return gain


def build_spec():
    """At every step: (at goal  OR  progressed over some window s<=W)  AND  separated from all other agents."""
    
    # Predicates 
    dg = Pred(dgoal, "<=", 0.1)
    vg = [Defined(Pred(vgoal(s), ">", 0.0)) for s in range(1, 1+W)]
    da = Pred(dlidar, "<", 0.2)
    va = [Defined(Pred(vagent(s), ">", 0.0)) for s in range(1, 1+W)]

    # SFO formula in CNF form
    phi = And(
        Or(Not(da), dg, *vg),
        da,
    )

    desc = [f"dg<={0.1} || dg(t-1)-dg(t) >= {0.05} ; and dsep>={0.2}"]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc

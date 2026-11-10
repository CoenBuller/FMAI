import torch
from fmai.robustness import Pred, Defined, And, Or, Not, WeightedRobustness

TAU1 = 1  # Window (steps) to measure progress toward the goal
TAU2 = 2  # Max consecutive steps allowed to not be well-spaced (delta time)
W = max(TAU1, TAU2)  # Total history window needed 

# Euclidean distance to the goal
def dgoal(obs):                      
    return torch.linalg.vector_norm(obs[..., 4:6], dim=-1)

# Max LiDAR reading (higher = closer to another agent)
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

def vgoal(s, min_vel=0.0):
    """Progress toward goal over s steps: dg(t-s) - dg(t) - min_vel * s"""
    return lambda o: lag(dgoal(o), s, o[..., 0, -1]) - dgoal(o) - (min_vel * s)

def shift_da(s):
    """dlidar(t-s): LiDAR reading shifted back by s steps"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1])


def build_spec():
    """
    1. Well-spaced -> moving to target or at target:  ~da(t) | dg(t) | vg(t, 1..TAU1)
    2. Not well-spaced -> well-spaced within TAU2:    da(t) | da(t-1) | ... | da(t-TAU2)
    3. Never within critical distance:                dap(t)
    """
    # --- Predicates ---
    dg  = Pred(dgoal, "<=", 0.1)     # At the target
    da  = Pred(dlidar, "<", 0.2)     # Well-spaced now
    dap = Pred(dlidar, "<", 0.3)     # Not in critical distance now

    # Progress toward target over s in [1, TAU1]
    vg = [Defined(Pred(vgoal(s, min_vel=0.005), ">", 0.0)) for s in range(1, 1 + TAU1)]

    # Well-spaced at some point in the last TAU2 steps: da(t-s) for s in [1, TAU2]
    # Note: Using Not(Defined(Not(...))) makes undefined warmup steps +inf (trivially satisfied)
    # or use Defined(...) if you want the agent to rely on current da(t) at step 0.
    past_da = [Defined(Pred(shift_da(s), "<", 0.2)) for s in range(1, 1 + TAU2)]

    # --- Clauses ---
    # Clause 1: If well-spaced now, must be at goal or making progress toward goal
    clause_goal = Or(Not(da), dg, *vg)

    # Clause 2: If not well-spaced TAU2 steps ago, must become well-spaced within TAU2 steps
    clause_spacing = Or(da, *past_da)

    # Clause 3: Never enter critical distance
    clause_critical = dap

    phi = And(
        clause_goal,
        clause_spacing,
        clause_critical,
    )

    desc = [
        r"$\forall t: (\neg da(t) \vee d_g(t) \vee \bigvee_{s=1}^{\tau_1} v_g(t, s)) "
        r"\wedge (\bigvee_{s=0}^{\tau_2} da(t-s)) "
        r"\wedge d_a^{+}(t)$"
    ]
    return WeightedRobustness([phi], weights=None, clip=1.0), desc
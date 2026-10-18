import numpy as np

from dataclasses import dataclass

from qsfo.monitoring.quantitative import OnlineMonitor
from qsfo.monitoring.trace import SignalsTrace
from qsfo.polyhedron import INFTY, NEG_INFTY, frac
from qsfo.parser import Parser



@dataclass(frozen=True)
class Sig:
    agent: int | slice = 0          # 0, slice(None) for all agents, ...
    obs: int | slice = 0            # 4, slice(6, 18), ...
    agg: str = "min"                # only used if the selection has more than one element

def extract(obs: np.ndarray, spec: Sig) -> np.ndarray:
    """obs: [E, T, A, D] -> [E, T]. Aggregates over all selected agents/entries."""
    agg = {
        "min": np.min, 
        "max": np.max, 
        "mean": np.mean
        }
    sel = obs[:, :, spec.agent, spec.obs]
    sel = sel.reshape(sel.shape[0], sel.shape[1], -1)     
    return agg[spec.agg](sel, axis=-1)


class WeighedRobustnessTracker():
    def __init__(self, formulas: dict[str, dict[str, Sig]], formula_weights: list[float], horizon: int, clip: float = 10.0):
        self.trackers = [RobustnessTracker(f, names, horizon) for f, names in formulas.items()]
        self.weights = formula_weights
        self.clip = clip

    def reset(self):
        for tracker in self.trackers:
            tracker.reset()

    def step(self, t, values_list):
        rs = [tr.step(t, vals) for tr, vals in zip(self.trackers, values_list)]  # always updates prev
        if any(r is None for r in rs):
            return None
        return sum(w * max(-self.clip, min(self.clip, r)) for w, r in zip(self.weights, rs)) # type: ignore


class RobustnessTracker:
    def __init__(self, formula: str, signal_names: dict[str, Sig], horizon: int):
        self.formula = self.parse_formula(formula)
        self.names = signal_names 
        self.H = horizon
        self.reset()

    def parse_formula(self, formula: str):
        parser = Parser()
        return parser.parse(formula)

    def reset(self):
        self.mon = OnlineMonitor(self.formula, self.H)   # fresh history per episode
        self.prev = None

    def step(self, t: float | int, values: dict):
        """Call once per env step. Returns robustness at time t (None for the first sample).
        
        values is a dict containing one measurement of each signal per variable. For variables 'u' and 'x' 
        with values 0.42 and 1 it becomes {"u": 0.42, "x": 1}""" 
        t_frac = frac(t)
        rho = None
        if self.prev is not None:
            t0, v0 = self.prev
            segs = {n: self.make_segment(n, t0, v0[n], t_frac, values[n]) for n in self.names}
            ref = segs[list(self.names.keys())[0]]
            time_interval = ref.time_bounds_as_ph().substitute(
                {ref.timevar(): self.mon.timevar}
            )
            M = self.mon.update(segs, time_interval)
            rho = self.value_at(M, t0, self.mon.timevar)
        self.prev = (t_frac, values)
        return rho


    def make_segment(self, name, t0, v0, t1, v1):
        tr = SignalsTrace(["t", name], [{"t": t0, name: v0}, {"t": t1, name: v1}])
        return tr.piecewise_linear_signal(name)[0]


    def value_at(self, M, t0, tvar):
        for r, S in M:
            if t0 in S:
                if r == INFTY:     return float("inf")
                if r == NEG_INFTY: return float("-inf")
                return float(r.subs(tvar, t0)) if hasattr(r, "subs") else float(r)
        return None   # undefined at t0

    
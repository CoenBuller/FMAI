from qsfo.monitoring.quantitative import OnlineMonitor
from qsfo.monitoring.trace import SignalsTrace
from qsfo.polyhedron import INFTY, NEG_INFTY, frac, Var

from qsfo.parser import Parser

class WeighedRobustnessTracker():
    def __init__(self, formulas: dict[str, list[str]], formula_weights: list[float], horizon: int, clip: float = 10.0):
        self.trackers = [RobustnessTracker(f, names, horizon) for f, names in formulas.items()]
        self.weights = formula_weights
        self.clip = clip

    def reset(self):
        for tracker in self.trackers:
            tracker.reset()

    def step(self, t: float | int, values_list: list[dict]):
        total, any_defined = 0.0, False
        for tr, w, vals in zip(self.trackers, self.weights, values_list):
            r = tr.step(t, vals)              # always updates tr.prev
            if r is None:
                return None                   # undefined robustness for a formula. Return None.
            r = max(-self.clip, min(self.clip, r))
            total += w * r
            any_defined = True
        return total if any_defined else None



class RobustnessTracker:
    def __init__(self, formula: str, signal_names: list[str], horizon: int):
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
            ref = segs[self.names[0]]
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

    
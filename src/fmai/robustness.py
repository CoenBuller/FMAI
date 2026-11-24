import torch
import torch.nn.functional as F

def _window(x, a, b, op):
    """Past-time window over [t-b, t-a] for x: [E, T]. Steps before t=0 are skipped (neutral padding)."""
    neutral = float("inf") if op == "min" else float("-inf")
    xp = F.pad(x, (b, 0), value=neutral)                 # [E, T+b]
    w = xp.unfold(1, b + 1, 1)[..., : b - a + 1]         # [E, T, b-a+1], covers times t-b .. t-a
    return w.amin(-1) if op == "min" else w.amax(-1)

class Pred:                      # fn(obs) <= c  /  fn(obs) >= c
    def __init__(self, fn, op, c): self.fn, self.op, self.c = fn, op, c
    def __call__(self, obs):
        x = self.fn(obs)
        return self.c - x if self.op in ("<=", "<") else x - self.c

class Not:
    def __init__(self, f): self.f = f
    def __call__(self, obs): return -self.f(obs)

class And:
    def __init__(self, *fs): self.fs = fs
    def __call__(self, obs): return torch.stack([f(obs) for f in self.fs]).amin(0)

class Or:
    def __init__(self, *fs): self.fs = fs
    def __call__(self, obs): return torch.stack([f(obs) for f in self.fs]).amax(0)

class Always:                    # forall s in [a, b]: phi(t - s)
    def __init__(self, f, a, b): self.f, self.a, self.b = f, a, b
    def __call__(self, obs): return _window(self.f(obs), self.a, self.b, "min")

class Eventually:                # exists s in [a, b]: phi(t - s)
    def __init__(self, f, a, b): self.f, self.a, self.b = f, a, b
    def __call__(self, obs): return _window(self.f(obs), self.a, self.b, "max")

class Defined:                                               # undefined (NaN) -> -inf, so it never wins an Or
    def __init__(self, f): self.f = f
    def __call__(self, obs): return torch.nan_to_num(self.f(obs), nan=float("-inf"))

class WeightedRobustness:
    def __init__(self, formulas, weights=None, clip=10.0):
        self.formulas = formulas
        self.weights = weights or [1 / len(formulas)] * len(formulas)
        self.clip = clip
    @torch.no_grad()
    def __call__(self, obs):                             # obs: [E, T, A, D] -> rho: [E, T]
        rho = 0.0
        for w, f in zip(self.weights, self.formulas):
            rho = rho + w * f(obs).clamp(-self.clip, self.clip)
        return rho
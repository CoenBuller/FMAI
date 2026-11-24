import torch
from fmai.robustness import Pred, Defined, And, Or, Not, WeightedRobustness

TAU1, TAU2 = 0, 1
TAU_FLOCK = 1  # Recovery horizon tau for flocking (matches W = TAU1 + TAU2 = 4)

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
    if scenario == "navigation":                 # scored after the rollout on the true positions (see GatedSpec)
        spec = GatedSpec.from_config(cfg)
        return spec, spec.desc()
    elif scenario == "flocking":
        return build_floc_spec(cfg)
    raise ValueError(f"Unknown scenario: {scenario}")


class GatedSpec:
    """Normalized to range [-1, 1] to make spec fair for different terms
    Specifically made for navigation environment.
    (1 = best achievable, 0 = about to violate, -1 = worst violation). Every term scores the state the action leads to:
         Prog_i  = h(d_i(before) - d_i(after)),  h: -1 at -v_max, 0 at v_min, 1 at v_max;  1 if within eps_g of the goal
         Agent_i = min_j [h_ij(t+1) - (1-gamma) h_ij(t)] / (gamma (d_far - d_safe)),  h_ij = d_ij - d_safe
                   (barrier: >= 0 when every gap shrinks by at most a fraction gamma per step, so d_ij never drops
                   below d_safe; gamma = drag / (1 - drag): the closing speed from which an agent can still coast to a stop)
         Zone_i  = g(min_k |p_i - z_k| - r),      g: -1 at zone_depth inside the edge, 0 at m_z outside it, 1 at >= zone_far
         rho_i   = Prog_i                         if Prog_i, Agent_i, Zone_i >= 0   (safe: go fast)
                   min(Prog_i, Agent_i, Zone_i)   otherwise                         (worst violation is the signal)
    Anticipation is left to the critic (GAE)."""

    def __init__(self, v_max=0.039, v_min=0.01, eps_g=0.1, d_safe=0.30, d_far=0.6, gamma=0.11,
                 zone_radius=0.25, m_z=0.0, zone_depth=0.1, zone_far=0.6):
        # v_min > 0 makes standing still negative; d_far, m_z, zone_depth: design choices
        self.v_max, self.v_min, self.eps_g = v_max, v_min, eps_g
        self.d_safe, self.d_far, self.gamma = d_safe, d_far, gamma
        self.zone_r, self.m_z, self.zone_depth, self.zone_far = zone_radius, m_z, zone_depth, zone_far

    @classmethod
    def from_config(cls, cfg, dt=0.1, substeps=2, u_range=1.0):   # VMAS Navigation defaults
        """Take every parameter the env defines from cfg, so the spec follows the yaml."""
        if cfg is None:
            return cls()
        # Top speed per step (full throttle along one axis). Each step: vel *= 1 - drag, then per substep
        # vel += u/mass * dt/substeps and pos += vel * dt/substeps. In steady state this moves:
        n, drag = substeps, cfg.DRAG
        v_max = u_range * dt ** 2 / (cfg.AGENT_MASS * drag) * (1 - drag * (n - 1) / (2 * n))
        return cls(v_max=v_max, eps_g=cfg.GOAL_EPS, d_safe=cfg.SAFETY_DIST, gamma=drag / (1 - drag),
                   zone_radius=cfg.ZONE_RADIUS, zone_far=cfg.ZONE_LIDAR_RANGE)

    @staticmethod
    def _pw(x, neg, zero, pos):
        """Piecewise linear: -1 at x = neg, 0 at x = zero, 1 at x = pos (clipped outside).
        Makes sure all parts are within range [-1,1] without cutting information"""
        return torch.where(x >= zero, (x - zero) / (pos - zero), (x - zero) / (zero - neg)).clamp(-1.0, 1.0)

    @torch.no_grad()
    def terms(self, pos, goal, pos_prev, zones=None):
        """pos, pos_prev: positions after / before each step's action, goal: [E, T, A, 2]; zones: [E, T, K, 2]
        -> dict of [E, T, A]."""
        dg = torch.linalg.vector_norm(pos - goal, dim=-1)
        step = torch.linalg.vector_norm(pos_prev - goal, dim=-1) - dg
        prog = torch.where(dg <= self.eps_g, torch.ones_like(dg), self._pw(step, -self.v_max, self.v_min, self.v_max))

        A = pos.shape[2]
        h, h_prev = torch.cdist(pos, pos) - self.d_safe, torch.cdist(pos_prev, pos_prev) - self.d_safe   # gap per pair
        cbf = (h - (1 - self.gamma) * h_prev) / (self.gamma * (self.d_far - self.d_safe))   # [E, T, A, A]
        agent = (cbf + torch.diag(torch.full((A,), float("inf"), device=pos.device))).amin(dim=-1).clamp(-1.0, 1.0)

        if zones is None:
            zone = torch.ones_like(dg)
        else:
            edge = (torch.linalg.vector_norm(pos[..., :, None, :] - zones[..., None, :, :], dim=-1)
                    - self.zone_r).amin(dim=-1)                                            # signed distance to nearest zone edge
            zone = self._pw(edge, -self.zone_depth, self.m_z, self.zone_far)

        worst = torch.minimum(prog, torch.minimum(agent, zone))
        return {"prog": prog, "agent": agent, "zone": zone, "total": torch.where(worst < 0, worst, prog)}

    def reward(self, td):
        """Per-agent robustness of a collected batch td [E, T] -> [E, T, A]."""
        inf = td["next", "agents", "info"]
        zones = inf["zones"][..., 0, :, :] if "zones" in inf.keys() else None             # same for every agent
        return self.terms(inf["pos"], inf["goal"], td["agents", "info", "pos"], zones)["total"]   # root info = before

    def desc(self):
        return [
            r"$\rho_{Prog_i}(t) = h(d_i(t) - d_i(t+1))$, $h$: $-1@-v_{max},\ 0@v_{min},\ 1@v_{max}$ (1 at goal)",
            r"$\rho_{Agent_i}(t) = \min_{j \neq i} \frac{h_{ij}(t+1) - (1-\gamma)\,h_{ij}(t)}{\gamma\,(d_{far} - d_{safe})}$, "
            r"$h_{ij} = d_{ij} - d_{safe}$",
            r"$\rho_{Zone_i}(t) = g(\min_k \lVert p_i(t+1) - z_k \rVert - r)$, $g$: $-1@-\delta_z,\ 0@m_z,\ 1@z_{far}$",
            r"$\rho_i = \rho_{Prog_i}$ if $\min(\cdot) \geq 0$, else $\min(\rho_{Prog_i}, \rho_{Agent_i}, \rho_{Zone_i})$",
            rf"$v_{{max}} = {self.v_max:.4f},\ v_{{min}} = {self.v_min},\ \gamma = {self.gamma:.3f},\ "
            rf"d_{{safe}} = {self.d_safe},\ d_{{far}} = {self.d_far},\ \delta_z = {self.zone_depth},\ m_z = {self.m_z},\ "
            rf"z_{{far}} = {self.zone_far}$",
        ]

import torch
from fmai.robustness import Pred, Defined, And, Or, Not, WeightedRobustness

TAU1, TAU2 = 1, 2
TAU_K, KAPPA = 20, 0.1   # luc: distance to goal must shrink by >= KAPPA over every TAU_K-step segment
TAU_R = 20               # luc: an agent closer than DA must be back at >= DA within TAU_R steps
WINDOW = max(TAU1 + TAU2, TAU_K, TAU_R)   # history length the online reward needs (RobustnessReward)

# Lidar obs = 0.35 - distance from the agent's centre to the other agent's surface (agent radius 0.1), so
# centre-to-centre distance = 0.45 - obs.
DA = 0.15       # "well-spaced": centres > 0.30 apart (= Config.SAFETY_DIST)
DAP = 0.25      # hard safety: centres > 0.20 apart, i.e. agents never touch (was 0.8, which can never be violated)

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

def vgoal_kappa(tau, kappa):
    """Segment progress: dgoal(t-k) - dgoal(t) - kappa*k/tau, with k the longest available lag <= tau
    (k = tau once the episode is tau steps old; shorter right after a reset). NaN at the first step."""
    def f(o):
        d, seg = dgoal(o), o[..., 0, -1]
        out = torch.full_like(d, float("nan"))
        for s in range(1, tau + 1):
            past = lag(d, s, seg)
            out = torch.where(torch.isnan(past), out, past - d - kappa * s / tau)
        return out
    return f

def shift_da(s):
    """dlidar(t-s): LiDAR reading shifted back by s steps"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1])

def vagent(s):
    """Separation progress: dlidar(t-s) - dlidar(t) > 0 (LiDAR proximity is decreasing)"""
    return lambda o: lag(dlidar(o), s, o[..., 0, -1]) - dlidar(o)


def build_spec(spec_type):
    if spec_type in ("future", "zone", "gated", "gated2", "gated3"):   # scored after the rollout (FutureSpec / ZoneSpec / GatedSpec)
        spec = {"future": FutureSpec, "zone": ZoneSpec, "gated": GatedSpec,
                # gated2: zone robustness 0 exactly at the zone edge, no look-ahead (the critic learns to anticipate),
                # -1 already 0.1 inside the edge; standing still is negative (Prog 0 at v_min = 0.01/step = kappa/tau_prog)
                "gated2": lambda: GatedSpec(v_min=0.01, m_z=0.0, tau_zone=0, zone_depth=0.1),
                # gated3: gated2 with horizon 1 everywhere: every term scores the state the action leads to (t+1),
                # Prog = the progress the action made; anticipation is left to the critic (GAE)
                "gated3": lambda: GatedSpec(v_min=0.01, m_z=0.0, tau_zone=0, zone_depth=0.1, tau_safe=0,
                                            one_step=True)}[spec_type]()
        return spec, spec.desc()
    if spec_type == "coen":
        """
        CNF Specification:
        1. Goal progress when safely spaced: ~da(t-TAU1) | ~da(t) | dg(t) | vg(1..TAU1)
        2. Spacing recovery or active avoidance: da(t) | da(t-1..t-TAU2) | va(1..TAU2)
        3. Hard safety (never critical distance): dap(t)
        """
        dg   = Pred(dgoal, "<=", 0.1)                        # At goal
        da   = Pred(dlidar, "<", DA)                         # Well-spaced now
        dap  = Pred(dlidar, "<", DAP)                        # Never in critical distance
        sda1 = Pred(shift_da(TAU1), "<", DA)                 # Well-spaced TAU1 steps ago

        # Progress toward goal over [1, TAU1]
        vg = [Defined(Pred(vgoal(s), ">", 0.0)) for s in range(1, 1 + TAU1)]

        # Well-spaced in the past [1, TAU2] OR actively increasing separation over [1, TAU2]
        past_da = [Defined(Pred(shift_da(s), "<", DA)) for s in range(1, 1 + TAU2)]
        va      = [Defined(Pred(vagent(s), ">", 0.0)) for s in range(1, 1 + TAU2)]

        phi = And(
            Or(Defined(Not(sda1)), Not(da), dg, *vg),
            Or(da, *past_da, *va),
            dap,
        )

        desc = [
            r"$\rho_{Prog_i}(t) = \max(d_{safe} - \bar d_i(t-\tau_1),\ d_{safe} - \bar d_i(t),\ "
            r"\epsilon_g - d_i(t),\ \max_{s \in [1,\tau_1]} d_i(t-s) - d_i(t))$",
            r"$\rho_{Rec_i}(t) = \max(\max_{s \in [0,\tau_2]} \bar d_i(t-s) - d_{safe},\ "
            r"\max_{s \in [1,\tau_2]} \bar d_i(t) - \bar d_i(t-s))$",
            r"$\rho_{Safe_i}(t) = \bar d_i(t) - d_{crit}$",
            r"$\rho_i(t) = \mathrm{clip}(\min(\rho_{Prog_i}, \rho_{Rec_i}, \rho_{Safe_i}), -c, c)$",
            rf"$\tau_1 = {TAU1},\ \tau_2 = {TAU2},\ \epsilon_g = 0.1,\ "
            rf"d_{{safe}} = {0.45 - DA:.2f},\ d_{{crit}} = {0.45 - DAP:.2f},\ c = 0.2$",
        ]
        return WeightedRobustness([phi], weights=None, clip=0.2), desc

    elif spec_type == "luc":
        """
        Coen's spec with two changes (past-time, so it can be a per-step reward):
        1. Goal progress over a segment: ~da(t-TAU1) | ~da(t) | dg(t) | vk(t), where vk(t) requires that the
           distance to the goal shrank by at least KAPPA over the last TAU_K steps (pro rata near episode start).
        2. Recovery within TAU_R steps: exists s in [0, TAU_R]: da(t-s). Getting closer than DA is not a
           violation by itself, only staying closer for more than TAU_R consecutive steps.
        3. Hard safety, unchanged: dap(t)
        """
        dg   = Pred(dgoal, "<=", 0.1)
        da   = Pred(dlidar, "<", DA)
        dap  = Pred(dlidar, "<", DAP)
        sda1 = Pred(shift_da(TAU1), "<", DA)
        vk   = Defined(vgoal_kappa(TAU_K, KAPPA))

        recent_da = [Defined(Pred(shift_da(s), "<", DA)) for s in range(0, 1 + TAU_R)]

        phi = And(
            Or(Defined(Not(sda1)), Not(da), dg, vk),
            Or(*recent_da),
            dap,
        )

        desc = [
            r"$\rho_{Prog_i}(t) = \max(d_{safe} - \bar d_i(t-\tau_1),\ d_{safe} - \bar d_i(t),\ "
            r"\epsilon_g - d_i(t),\ d_i(t-k) - d_i(t) - \kappa k/\tau_k)$",
            r"$\rho_{Rec_i}(t) = \max_{s \in [0,\tau_r]} \bar d_i(t-s) - d_{safe}$",
            r"$\rho_{Safe_i}(t) = \bar d_i(t) - d_{crit}$",
            r"$\rho_i(t) = \mathrm{clip}(\min(\rho_{Prog_i}, \rho_{Rec_i}, \rho_{Safe_i}), -c, c)$",
            rf"$\tau_1 = {TAU1},\ \tau_k = {TAU_K},\ \kappa = {KAPPA},\ \tau_r = {TAU_R},\ \epsilon_g = 0.1,\ "
            rf"d_{{safe}} = {0.45 - DA:.2f},\ d_{{crit}} = {0.45 - DAP:.2f},\ k = \min(\tau_k, t - t_0),\ c = 0.2$",
        ]

        return WeightedRobustness([phi], weights=None, clip=0.2), desc

# ------------------------------ future-time spec ("future") ------------------------------
# Scored after the rollout on the true positions (info["pos"], info["goal"]), so a step's robustness can use the
# steps that follow it. Windows stop at the end of the episode / batch (tilde-tau = min(tau, t_end - t)).

def lead(x, u, seg):
    """x[:, t + u] for x: [E, T, ...]; NaN if t + u is past the batch or in a later episode. seg: [E, T]."""
    out = torch.full_like(x, float("nan"))
    out[:, :-u] = x[:, u:]
    same = torch.zeros_like(seg, dtype=torch.bool)
    same[:, :-u] = seg[:, :-u] == seg[:, u:]
    return torch.where(same.reshape(*same.shape, *[1] * (x.dim() - 2)), out, torch.full_like(out, float("nan")))


class FutureSpec:
    """rho_i(t) = min(Prog_i, min_j Rec_ij, min_j Safe_ij), per agent, from the future window of step t:
         Prog_i(t)  = max(d_i(t) - d_i(t+T) - kappa*T/tau_prog,  eps_g - d_i(t+T)),   T = min(tau_prog, t_end - t)
         Rec_ij(t)  = max_{u in [0, min(tau_rec, t_end - t)]} d_ij(t+u) - d_safe
         Safe_ij(t) = d_ij(t) - d_crit"""

    def __init__(self, tau_prog=20, tau_rec=5, kappa=0.2, eps_g=0.1, d_safe=0.30, d_crit=0.20):
        # kappa: a third of the progress of a trained policy (0.032/step -> 0.64 per 20 steps, measured with analyze.py);
        # feasible with an avoidance manoeuvre as long as kappa <= v * (tau_prog - 2 * tau_rec) = 0.32
        self.tau_prog, self.tau_rec, self.kappa = tau_prog, tau_rec, kappa
        self.eps_g, self.d_safe, self.d_crit = eps_g, d_safe, d_crit

    @torch.no_grad()
    def terms(self, pos, goal, done):
        """pos, goal: [E, T, A, 2]; done: [E, T] or [E, T, 1] -> dict of [E, T, A] (prog, rec, safe, total)."""
        d = done.reshape(done.shape[0], done.shape[1]).long()
        seg = d.cumsum(1) - d                                            # episode index of each step
        dg = torch.linalg.vector_norm(pos - goal, dim=-1)                # [E, T, A]
        A = pos.shape[2]
        dij = torch.cdist(pos, pos) + torch.diag(torch.full((A,), float("inf"), device=pos.device))  # [E, T, A, A]

        prog = (self.eps_g - dg).clamp(min=0.0)                          # T = 0 (last step): max(0, eps_g - d_i)
        for u in range(1, self.tau_prog + 1):                            # keep the longest available window
            fut = lead(dg, u, seg)
            val = torch.maximum(dg - fut - self.kappa * u / self.tau_prog, self.eps_g - fut)
            prog = torch.where(torch.isnan(fut), prog, val)

        rec = dij - self.d_safe
        for u in range(1, self.tau_rec + 1):
            fut = lead(dij, u, seg)
            rec = torch.where(torch.isnan(fut), rec, torch.maximum(rec, fut - self.d_safe))
        rec = rec.amin(dim=-1)
        safe = (dij - self.d_crit).amin(dim=-1)
        return {"prog": prog, "rec": rec, "safe": safe, "total": torch.minimum(prog, torch.minimum(rec, safe))}

    def reward(self, td):
        """Per-agent robustness of a collected batch td [E, T] -> [E, T, A]."""
        inf = td["next", "agents", "info"]
        return self.terms(inf["pos"], inf["goal"], td["next", "done"])["total"]

    def desc(self):
        return [
            r"$\rho_{Prog_i}(t) = \max(d_i(t) - d_i(t+\tilde\tau_{prog}) - \kappa \tilde\tau_{prog}/\tau_{prog},\ "
            r"\epsilon_g - d_i(t+\tilde\tau_{prog}))$",
            r"$\rho_{Rec_{ij}}(t) = \max_{u \in [0,\tilde\tau_{rec}]} d_{ij}(t+u) - d_{safe}$",
            r"$\rho_{Safe_{ij}}(t) = d_{ij}(t) - d_{crit}$",
            r"$\rho_i(t) = \min(\rho_{Prog_i}, \min_{j \neq i} \rho_{Rec_{ij}}, \min_{j \neq i} \rho_{Safe_{ij}})$",
            rf"$\tau_{{prog}} = {self.tau_prog},\ \tau_{{rec}} = {self.tau_rec},\ \kappa = {self.kappa},\ "
            rf"\epsilon_g = {self.eps_g},\ d_{{safe}} = {self.d_safe},\ d_{{crit}} = {self.d_crit},\ "
            r"\tilde\tau = \min(\tau, t_{end} - t)$",
        ]


class ZoneSpec(FutureSpec):
    """Normalised future-time spec with danger zones (SFO temporal formula, free time variable t, per agent i):
         phi_i(t) = (0 <= g_prog_i(t)  or  0 <= g_goal_i(t + tau_prog))
                    and AND_{j != i} exists c in [0, tau_rec].  0 <= g_safe_ij(t + c)
                    and AND_{j != i} 0 <= g_crit_ij(t)
                    and AND_k forall c in [0, tau_zone].  0 <= g_zone_ik(t + c)
       Every derived signal g is clip(margin / scale, -1, 1): 0 = just satisfied, 1 = comfortably satisfied.
       Quantifiers range over the steps that exist in the episode / batch (SFO: undefined accesses are skipped);
       g_prog over a shorter window is pro rata (kappa * k / tau_prog), as in FutureSpec."""

    def __init__(self, zone_radius=0.25, tau_zone=5, m_z=0.1, prog_scale=0.4, band=0.2, **kw):
        # prog_scale: Prog = clip((gain - kappa) / prog_scale): standing still -0.5, saturates at gain 0.6 per 20 steps
        # (0.03/step, about native speed), so faster is better until then. band: Rec/Safe reach -1 only band below
        # their threshold (was d_safe - d_crit = 0.1), so agents that are too close still get a gradient apart.
        self.prog_scale, self.band = prog_scale, band
        super().__init__(**kw)
        self.zone_r, self.tau_zone, self.m_z = zone_radius, tau_zone, m_z

    @torch.no_grad()
    def terms(self, pos, goal, done, zones=None):
        """pos, goal: [E, T, A, 2]; done: [E, T(, 1)]; zones: [E, T, K, 2] -> dict of [E, T, A] in [-1, 1]."""
        clip = lambda x: x.clamp(-1.0, 1.0)
        d = done.reshape(done.shape[0], done.shape[1]).long()
        seg = d.cumsum(1) - d
        dg = torch.linalg.vector_norm(pos - goal, dim=-1)
        A = pos.shape[2]
        dij = torch.cdist(pos, pos) + torch.diag(torch.full((A,), float("inf"), device=pos.device))
        band = self.band

        prog = clip((self.eps_g - dg) / self.eps_g).clamp(min=0.0)       # no window left (last step): neutral or at goal
        for u in range(1, self.tau_prog + 1):
            fut = lead(dg, u, seg)
            val = torch.maximum((dg - fut - self.kappa * u / self.tau_prog) / (self.prog_scale * u / self.tau_prog),
                                (self.eps_g - fut) / self.eps_g)
            prog = torch.where(torch.isnan(fut), prog, clip(val))

        rec = clip((dij - self.d_safe) / band)
        for u in range(1, self.tau_rec + 1):
            fut = lead(dij, u, seg)
            rec = torch.where(torch.isnan(fut), rec, torch.maximum(rec, clip((fut - self.d_safe) / band)))
        rec = rec.amin(dim=-1)
        safe = clip((dij - self.d_crit) / band).amin(dim=-1)

        if zones is None:
            zone = torch.ones_like(dg)
        else:
            gz = clip((torch.linalg.vector_norm(pos[..., :, None, :] - zones[..., None, :, :], dim=-1) - self.zone_r)
                      / self.m_z).amin(dim=-1)                          # nearest zone, [E, T, A]
            zone = gz
            for u in range(1, self.tau_zone + 1):
                fut = lead(gz, u, seg)
                zone = torch.where(torch.isnan(fut), zone, torch.minimum(zone, fut))
        total = torch.minimum(torch.minimum(prog, rec), torch.minimum(safe, zone))
        return {"prog": prog, "rec": rec, "safe": safe, "zone": zone, "total": total}

    def reward(self, td):
        inf = td["next", "agents", "info"]
        zones = inf["zones"][..., 0, :, :] if "zones" in inf.keys() else None   # same for every agent
        return self.terms(inf["pos"], inf["goal"], td["next", "done"], zones)["total"]

    def desc(self):
        return [
            r"$\varphi_i(t) \equiv (0 \leq g^{prog}_i(t) \lor 0 \leq g^{goal}_i(t+\tau_{prog})) "
            r"\land \bigwedge_{j \neq i} \exists c \in [0,\tau_{rec}].\ 0 \leq g^{safe}_{ij}(t+c) "
            r"\land \bigwedge_{j \neq i} 0 \leq g^{crit}_{ij}(t) "
            r"\land \bigwedge_k \forall c \in [0,\tau_{zone}].\ 0 \leq g^{zone}_{ik}(t+c)$",
            r"$g^{prog}_i = \mathrm{clip}((d_i(t) - d_i(t+\tau_{prog}) - \kappa)/s),\ "
            r"g^{goal}_i = \mathrm{clip}((\epsilon_g - d_i)/\epsilon_g),\ "
            r"g^{safe}_{ij} = \mathrm{clip}((d_{ij} - d_{safe})/b),\ "
            r"g^{crit}_{ij} = \mathrm{clip}((d_{ij} - d_{crit})/b),\ "
            r"g^{zone}_{ik} = \mathrm{clip}((\lVert p_i - z_k \rVert - r_k)/m_z)$",
            rf"$\tau_{{prog}} = {self.tau_prog},\ \kappa = {self.kappa},\ \epsilon_g = {self.eps_g},\ "
            rf"\tau_{{rec}} = {self.tau_rec},\ d_{{safe}} = {self.d_safe},\ d_{{crit}} = {self.d_crit},\ "
            rf"\tau_{{zone}} = {self.tau_zone},\ r_k = {self.zone_r},\ m_z = {self.m_z},\ s = {self.prog_scale},\ "
            rf"b = {self.band}$",
        ]


class GatedSpec(ZoneSpec):
    """Two-part ("gated") robustness reward, every term piecewise linear in [-1, 1]
    (1 = best achievable, 0 = about to violate, -1 = worst violation):
         Prog_i(t)  = h((d_i(t) - d_i(t+T)) / T),  T = min(tau_prog, t_end - t);  1 if within eps_g of the goal at t+T
                      h: -1 at -v_max (full speed away), 0 at v_min, 1 at v_max (full speed towards the goal);
                      v_min > 0 makes standing still negative
         Agent_i(t) = min_{u in [0, tau_safe]} min_{j != i} f(d_ij(t+u)),  f: 1 at >= d_far, 0 at d_safe, -1 at d_crit
         Zone_i(t)  = min_{u in [0, tau_zone]} min_k  g(|p_i - z_k| - r)(t+u),  g: 1 at >= zone_far from the edge,
                      0 at m_z outside the edge, -1 at zone_depth inside the edge (default: the zone centre)
         rho_i(t)   = Prog_i(t)                        if Agent_i, Zone_i, Prog_i >= 0   (safe: go fast)
                      min(Prog_i, Agent_i, Zone_i)     otherwise                         (worst violation is the signal)"""

    def __init__(self, v_max=0.039, v_min=0.0, tau_safe=5, d_far=0.6, zone_far=0.6, zone_depth=None, one_step=False,
                 **kw):
        # v_max: measured top speed per step with AGENT_MASS 2.5, DRAG 0.1 (full thrust, ~0.039 after ~50 steps)
        # one_step: Prog = progress made by the action itself, d_i(before) - d_i(after) (horizon 1, no future window);
        # combine with tau_safe = tau_zone = 0 so every term only scores the state the action leads to
        super().__init__(**kw)
        self.v_max, self.v_min, self.tau_safe, self.d_far, self.zone_far = v_max, v_min, tau_safe, d_far, zone_far
        self.zone_depth = self.zone_r if zone_depth is None else zone_depth
        self.one_step = one_step

    @staticmethod
    def _pw(x, neg, zero, pos):
        """Piecewise linear: -1 at x = neg, 0 at x = zero, 1 at x = pos (clipped outside)."""
        return torch.where(x >= zero, (x - zero) / (pos - zero), (x - zero) / (zero - neg)).clamp(-1.0, 1.0)

    @staticmethod
    def _future_min(x, tau, seg):
        out = x
        for u in range(1, tau + 1):
            fut = lead(x, u, seg)
            out = torch.where(torch.isnan(fut), out, torch.minimum(out, fut))
        return out

    @torch.no_grad()
    def terms(self, pos, goal, done, zones=None, pos_prev=None):
        """pos: positions after each step's action; pos_prev: before it (only needed with one_step)."""
        d = done.reshape(done.shape[0], done.shape[1]).long()
        seg = d.cumsum(1) - d
        dg = torch.linalg.vector_norm(pos - goal, dim=-1)                                 # [E, T, A]
        A = pos.shape[2]
        dij = torch.cdist(pos, pos) + torch.diag(torch.full((A,), float("inf"), device=pos.device))

        at_goal = lambda x: x <= self.eps_g
        if self.one_step:                                                                  # progress of this action
            step = torch.linalg.vector_norm(pos_prev - goal, dim=-1) - dg
            prog = torch.where(at_goal(dg), torch.ones_like(dg), self._pw(step, -self.v_max, self.v_min, self.v_max))
        else:
            prog = at_goal(dg).float()                                                     # no window left: 1 at goal, else 0
            for u in range(1, self.tau_prog + 1):                                          # keep the longest available window
                fut = lead(dg, u, seg)
                val = self._pw((dg - fut) / u, -self.v_max, self.v_min, self.v_max)
                val = torch.where(at_goal(fut), torch.ones_like(val), val)
                prog = torch.where(torch.isnan(fut), prog, val)

        agent = self._future_min(self._pw(dij.amin(dim=-1), self.d_crit, self.d_safe, self.d_far), self.tau_safe, seg)

        if zones is None:
            zone = torch.ones_like(dg)
        else:
            edge = (torch.linalg.vector_norm(pos[..., :, None, :] - zones[..., None, :, :], dim=-1)
                    - self.zone_r).amin(dim=-1)                                            # signed distance to nearest zone edge
            zone = self._future_min(self._pw(edge, -self.zone_depth, self.m_z, self.zone_far), self.tau_zone, seg)

        worst = torch.minimum(prog, torch.minimum(agent, zone))
        total = torch.where(worst < 0, worst, prog)
        return {"prog": prog, "agent": agent, "zone": zone, "worst": worst, "total": total}

    def reward(self, td):
        inf = td["next", "agents", "info"]
        zones = inf["zones"][..., 0, :, :] if "zones" in inf.keys() else None
        pos_prev = td["agents", "info", "pos"] if self.one_step else None                 # root info = state before the action
        return self.terms(inf["pos"], inf["goal"], td["next", "done"], zones, pos_prev)["total"]

    def desc(self):
        if self.one_step:
            return [
                r"$\rho_{Prog_i}(t) = h(d_i(t) - d_i(t+1))$, $h$: $-1@-v_{max},\ 0@v_{min},\ 1@v_{max}$ (1 at goal)",
                r"$\rho_{Agent_i}(t) = f(\min_{j \neq i} d_{ij}(t+1))$, $f$: $-1@d_{crit},\ 0@d_{safe},\ 1@d_{far}$",
                r"$\rho_{Zone_i}(t) = g(\min_k \lVert p_i(t+1) - z_k \rVert - r)$, $g$: $-1@-\delta_z,\ 0@m_z,\ 1@z_{far}$",
                r"$\rho_i = \rho_{Prog_i}$ if $\min(\cdot) \geq 0$, else $\min(\rho_{Prog_i}, \rho_{Agent_i}, \rho_{Zone_i})$",
                rf"$v_{{max}} = {self.v_max},\ v_{{min}} = {self.v_min},\ d_{{crit}} = {self.d_crit},\ "
                rf"d_{{safe}} = {self.d_safe},\ d_{{far}} = {self.d_far},\ \delta_z = {self.zone_depth},\ m_z = {self.m_z},\ "
                rf"z_{{far}} = {self.zone_far}$",
            ]
        return [
            r"$\rho_{Prog_i}(t) = h((d_i(t) - d_i(t+\tilde\tau_{prog})) / \tilde\tau_{prog})$, "
            r"$h$: $-1@-v_{max},\ 0@v_{min},\ 1@v_{max}$ (1 at goal)",
            r"$\rho_{Agent_i}(t) = \min_{u \in [0,\tau_{safe}]} f(\min_{j \neq i} d_{ij}(t+u))$, "
            r"$f$: $-1@d_{crit},\ 0@d_{safe},\ 1@d_{far}$",
            r"$\rho_{Zone_i}(t) = \min_{u \in [0,\tau_{zone}]} g(\min_k \lVert p_i - z_k \rVert - r)$, "
            r"$g$: $-1@-\delta_z,\ 0@m_z,\ 1@z_{far}$",
            r"$\rho_i = \rho_{Prog_i}$ if $\min(\cdot) \geq 0$, else $\min(\rho_{Prog_i}, \rho_{Agent_i}, \rho_{Zone_i})$",
            rf"$v_{{max}} = {self.v_max},\ v_{{min}} = {self.v_min},\ \delta_z = {self.zone_depth},\ \tau_{{prog}} = {self.tau_prog},\ \tau_{{safe}} = {self.tau_safe},\ "
            rf"\tau_{{zone}} = {self.tau_zone},\ d_{{crit}} = {self.d_crit},\ d_{{safe}} = {self.d_safe},\ "
            rf"d_{{far}} = {self.d_far},\ r = {self.zone_r},\ m_z = {self.m_z},\ z_{{far}} = {self.zone_far}$",
        ]

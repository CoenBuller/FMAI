#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Callable, Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.15)

        # Lidar configuration (make sure lidar_max_range >= min_proximity)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)
        self.lidar_max_range = kwargs.pop("lidar_max_range", 0.35)

        # --- Reward Parameters ---
        # 1. Distance band around target: r_d(t) = epsilon_t - |r_i(t) - r_t|
        self.target_radius = kwargs.pop("target_radius", 0.3)          # r_t
        self.epsilon_t = kwargs.pop("epsilon_t", 0.05)                 # epsilon_t (band half-width & positive bonus)
        self.flat_band_bonus = kwargs.pop("flat_band_bonus", False)    # if True: +epsilon_t anywhere inside band

        # 2. Lidar proximity threshold p_t
        self.min_proximity = kwargs.pop("min_proximity", 0.15)         # p_t

        # 3. Velocity limit v_max: r_v^max(t) = v_max - ||v_a(t)||
        self.v_max = kwargs.pop("v_max", 0.5)                          # v_max

        # Reward weights (all 1.0 by default: R(t) = r_v_rel + r_v_max + r_p + r_d)
        self.w_dist = kwargs.pop("w_dist", 1.0)
        self.w_prox = kwargs.pop("w_prox", 1.0)
        self.w_v_rel = kwargs.pop("w_v_rel", 1.0)
        self.w_v_max = kwargs.pop("w_v_max", 1.0)

        # Wind parameters
        self.wind_strength = kwargs.pop("wind_strength", 0.08)                  # Set to 0 if you want no wind
        self.wind_gust_strength = kwargs.pop("wind_gust_strength", 0.04)        # Set to 0 if you want no wind gusts
        self.random_wind_direction = kwargs.pop("random_wind_direction", True)

        # Target movement parameters
        self.target_speed_action = kwargs.pop("target_speed_action", 0.5)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert (
            self.lidar_max_range >= self.min_proximity
        ), "lidar_max_range must be >= min_proximity so Lidar can detect violations of p_t."

        self.plot_grid = True
        self.x_dim = 1
        self.y_dim = 1

        # Base wind angle per environment in the batch
        self.wind_angle = torch.zeros((batch_dim, 1), device=device, dtype=torch.float32)

        # Make world (no obstacles)
        world = World(batch_dim, device, collision_force=400, substeps=5)

        # Add moving target agent
        self._target = Agent(
            name="target",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            action_script=self.action_script_creator(),
        )
        world.add_agent(self._target)

        # Lidar detects other Agents (VMAS Lidar automatically ignores self)
        agent_entity_filter: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.lidar_max_range,
                        entity_filter=agent_entity_filter,
                    )
                ],
                render_action=True,
            )
            agent.r_d = torch.zeros(batch_dim, device=device)
            agent.r_p = torch.zeros(batch_dim, device=device)
            agent.r_v_rel = torch.zeros(batch_dim, device=device)
            agent.r_v_max = torch.zeros(batch_dim, device=device)

            world.add_agent(agent)

        return world

    def action_script_creator(self):
        def action_script(agent, world):
            t = self.t / 30
            agent.action.u = (
                torch.stack([torch.cos(t), torch.sin(t)], dim=1)
                * self.target_speed_action
            )

        return action_script

    def get_wind_at(self, agent: Agent) -> Tensor:
        """Computes steady base wind + time/space-varying gusts for an agent."""

        if self.wind_strength > 0:
            base_wind = self.wind_strength * torch.cat(
                [torch.cos(self.wind_angle), torch.sin(self.wind_angle)], dim=-1
            )
        if self.wind_gust_strength > 0:
            t = self.t.unsqueeze(-1) / 20.0
            gust_x = torch.sin(t + agent.state.pos[:, X : X + 1] * 3.0)
            gust_y = torch.cos(t + agent.state.pos[:, Y : Y + 1] * 3.0)
            gust = self.wind_gust_strength * torch.cat([gust_x, gust_y], dim=-1)
            return base_wind + gust
        return base_wind

    def process_action(self, agent: Agent):
        """Adds wind force to policy agents before physics stepping."""
        if agent is self._target:
            return
        agent.action.u = agent.action.u + self.get_wind_at(agent)

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
            if self.random_wind_direction:
                self.wind_angle = (
                    torch.rand(
                        (self.world.batch_dim, 1),
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    * 2
                    * torch.pi
                )
        else:
            self.t[env_index] = 0
            if self.random_wind_direction:
                self.wind_angle[env_index] = (
                    torch.rand((1,), device=self.world.device, dtype=torch.float32)
                    * 2
                    * torch.pi
                )

        target_pos = torch.zeros(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
            dtype=torch.float32,
        )
        target_pos[:, Y] = -self.y_dim * 0.5
        self._target.set_pos(target_pos, batch_index=env_index)

        ScenarioUtils.spawn_entities_randomly(
            self.world.policy_agents,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
            occupied_positions=target_pos.unsqueeze(1),
        )

    def reward(self, agent: Agent):
        is_first = self.world.policy_agents.index(agent) == 0
        if is_first:
            self.t += 1

        # 1. Distance reward r_d(t) with band margin epsilon_t
        #    Positive inside [r_t - epsilon_t, r_t + epsilon_t], negative outside
        r_i = torch.linalg.vector_norm(
            agent.state.pos - self._target.state.pos, dim=-1
        )
        radial_error = (r_i - self.target_radius).abs()
        if self.flat_band_bonus:
            in_band = radial_error <= self.epsilon_t
            agent.r_d = torch.where(
                in_band,
                torch.full_like(r_i, self.epsilon_t),
                self.epsilon_t - radial_error,
            ) * self.w_dist
        else:
            agent.r_d = (self.epsilon_t - radial_error) * self.w_dist

        # 2. Lidar-based proximity penalty r_p(t)
        #    Lidar rays return distance in [0, lidar_max_range] (lidar_max_range when nothing hit)
        lidar_readings = agent.sensors[0].measure()
        min_lidar_dist, _ = torch.min(lidar_readings, dim=-1)  # p_a(t)
        # Negative penalty when closer than min_proximity (p_t), 0 otherwise
        agent.r_p = torch.clamp(min_lidar_dist - self.min_proximity, max=0.0) * self.w_prox

        # 3. Relative Velocity Reward r_v^rel(t) = ||v_a(t) - v_t(t)||
        rel_vel_norm = torch.linalg.vector_norm(
            agent.state.vel - self._target.state.vel, dim=-1
        )
        agent.r_v_rel = rel_vel_norm * self.w_v_rel

        # 4. Max Velocity Reward r_v^max(t) = v_max - ||v_a(t)||
        abs_vel_norm = torch.linalg.vector_norm(agent.state.vel, dim=-1)
        agent.r_v_max = (self.v_max - abs_vel_norm) * self.w_v_max

        # Total Reward: R(t) = r_v^rel(t) + r_v^max(t) + r_p(t) + r_d(t)
        return agent.r_v_rel + agent.r_v_max + agent.r_p + agent.r_d

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,                          # [0:2]
                agent.state.vel,                          # [2:4]
                agent.state.pos - self._target.state.pos, # [4:6]
                agent.state.vel - self._target.state.vel, # [6:8]
                self.get_wind_at(agent),                  # [8:10]
                agent.sensors[0].measure(),               # [10:10+n_lidar_rays]
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "r_d": agent.r_d,
            "r_p": agent.r_p,
            "r_v_rel": agent.r_v_rel,
            "r_v_max": agent.r_v_max,
        }


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        current_pos = observation[:, 0:2]
        rel_pos_to_target = observation[:, 4:6]
        wind = observation[:, 8:10]
        lidar = observation[:, 10:22]

        circle_origin = current_pos - rel_pos_to_target
        circle_radius = 0.3
        v = rel_pos_to_target
        v_norm = torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-6)
        closest_point_on_circ = circle_origin + (v / v_norm) * circle_radius

        radial_from_origin = closest_point_on_circ - circle_origin
        closest_point_on_circ_normal = torch.stack(
            [radial_from_origin[:, Y], -radial_from_origin[:, X]], dim=1
        )
        normal_norm = torch.linalg.norm(
            closest_point_on_circ_normal, dim=1, keepdim=True
        ).clamp(min=1e-6)
        closest_point_on_circ_normal = (closest_point_on_circ_normal / normal_norm) * 0.1
        des_pos = closest_point_on_circ + closest_point_on_circ_normal

        object_visible = torch.any(lidar < 0.15, dim=1)
        _, object_dir_index = torch.min(lidar, dim=1)
        object_dir = object_dir_index / lidar.shape[1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=1)
        des_pos_object = current_pos - object_vec * 0.15
        des_pos[object_visible] = des_pos_object[object_visible]

        action = torch.clamp(
            (des_pos - current_pos) * 10 - wind,
            min=-u_range,
            max=u_range,
        )

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
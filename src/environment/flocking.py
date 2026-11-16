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


class Flocking(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.15)

        # Lidar configuration
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)
        self.lidar_max_range = kwargs.pop("lidar_max_range", 0.35)

        # Distance & Speed targets
        self.desired_target_distance = kwargs.pop("desired_target_distance", 0.3)
        self.taget_distance_band_width = kwargs.pop("target_distance_band_width", 0.05)
        self.desired_speed = kwargs.pop("desired_speed", 0.25)
        self.relative_speed_target = kwargs.pop("relative_speed_target", True)

        # Reward weights
        self.collision_reward = kwargs.pop("collision_reward", -0.1)
        self.dist_shaping_factor = kwargs.pop("dist_shaping_factor", 1.0)
        self.speed_shaping_factor = kwargs.pop("speed_shaping_factor", 0.5)

        # Wind parameters
        self.wind_strength = kwargs.pop("wind_strength", 0.08)                          # Set to 0 for no wind
        self.wind_gust_strength = kwargs.pop("wind_gust_strength", 0.04)                # Set to 0 for no wind gusts       
        self.random_wind_direction = kwargs.pop("random_wind_direction", True)    

        # Target movement parameters
        self.target_speed_action = kwargs.pop("target_speed_action", 0.5)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.plot_grid = True
        self.min_collision_distance = 0.005
        self.x_dim = 1
        self.y_dim = 1

        # Wind state: base wind angle per environment in the batch
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

        # IMPORTANT FIX: Detect other Agents with Lidar (original had `not isinstance(e, Agent)`)
        # VMAS Lidar automatically ignores the sensing agent itself.
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
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.dist_rew = torch.zeros(batch_dim, device=device)
            agent.speed_rew = torch.zeros(batch_dim, device=device)
            agent.distance_shaping = torch.zeros(batch_dim, device=device)

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
            # Spatially and temporally varying sinusoidal gusts
            gust_x = torch.sin(t + agent.state.pos[:, X : X + 1] * 3.0)
            gust_y = torch.cos(t + agent.state.pos[:, Y : Y + 1] * 3.0)
            gust = self.wind_gust_strength * torch.cat([gust_x, gust_y], dim=-1)
            return base_wind + gust
        return base_wind

    def process_action(self, agent: Agent):
        """Hook called by VMAS before stepping physics. Adds wind force to policy agents."""
        if agent is self._target:
            return
        wind_force = self.get_wind_at(agent)
        agent.action.u = agent.action.u + wind_force

    def _compute_distance_shaping(self, agent: Agent, env_index: int = None) -> Tensor:
        """Computes L1 norm between desired distance and agents position."""
        if env_index is None:
            # 1. Distance to the moving target vs. desired_target_distance
            dist_to_target = torch.linalg.vector_norm(
                agent.state.pos - self._target.state.pos, dim=-1
            )
            target_shaping = self.target_distance_band_width - torch.abs((dist_to_target - self.desired_target_distance))
            
            return target_shaping * self.dist_shaping_factor
        else:
            dist_to_target = torch.linalg.vector_norm(
                agent.state.pos[env_index] - self._target.state.pos[env_index]
            )
            target_shaping = self.target_distance_band_width - torch.abs((dist_to_target - self.desired_target_distance))

            return target_shaping * self.dist_shaping_factor

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

        # Spawn only policy agents (obstacles removed)
        ScenarioUtils.spawn_entities_randomly(
            self.world.policy_agents,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
            occupied_positions=target_pos.unsqueeze(1),
        )

        for agent in self.world.policy_agents:
            if env_index is None:
                agent.distance_shaping = self._compute_distance_shaping(
                    agent, env_index=None
                )
            else:
                agent.distance_shaping[env_index] = self._compute_distance_shaping(
                    agent, env_index=env_index
                )

    def reward(self, agent: Agent):
        is_first = self.world.policy_agents.index(agent) == 0

        if is_first:
            self.t += 1

            # Avoid collisions with each other and the target
            if self.collision_reward != 0:
                for a in self.world.policy_agents:
                    a.collision_rew[:] = 0

                for i, a in enumerate(self.world.agents):
                    for j, b in enumerate(self.world.agents):
                        if j <= i:
                            continue
                        collision = (
                            self.world.get_distance(a, b) <= self.min_collision_distance
                        )
                        if a.action_script is None:
                            a.collision_rew[collision] += self.collision_reward
                        if b.action_script is None:
                            b.collision_rew[collision] += self.collision_reward

        # 1. Distance shaping reward (orbit radius around target + inter-agent spacing)
        current_dist_shaping = self._compute_distance_shaping(agent, env_index=None)
        agent.dist_rew = agent.distance_shaping - current_dist_shaping
        agent.distance_shaping = current_dist_shaping

        # 2. Specific speed reward (penalize deviation from desired_speed)
        vel = (
            agent.state.vel - self._target.state.vel
            if self.relative_speed_target
            else agent.state.vel
        )
        current_speed = torch.linalg.vector_norm(vel, dim=-1)
        agent.speed_rew = (
            -(current_speed - self.desired_speed).abs() * self.speed_shaping_factor
        )

        return agent.collision_rew + agent.dist_rew + agent.speed_rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,                               # [0:2]: global position of agent
                agent.state.vel,                               # [2:4]: global velocity of agent
                agent.state.pos - self._target.state.pos,      # [4:6]: agents relative position to target
                agent.state.vel - self._target.state.vel,      # [6:8]: agents relative velocity to target
                self.get_wind_at(agent),                       # [8:10]: the wind strength in x,y direction for agent 
                agent.sensors[0].measure(),                    # [10:10+n_lidar_rays]: agents lidar readings (proximity)
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "agent_collision_rew": agent.collision_rew,
            "agent_distance_rew": agent.dist_rew,
            "agent_speed_rew": agent.speed_rew,
        }


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        current_pos = observation[:, 0:2]
        rel_pos_to_target = observation[:, 4:6]
        wind = observation[:, 8:10]
        lidar = observation[:, 10:22]

        # Orbit around the moving target's actual position at radius 0.3
        circle_origin = current_pos - rel_pos_to_target
        circle_radius = 0.3
        v = rel_pos_to_target
        v_norm = torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-6)
        closest_point_on_circ = circle_origin + (v / v_norm) * circle_radius

        # Tangent vector for circular motion around the target
        radial_from_origin = closest_point_on_circ - circle_origin
        closest_point_on_circ_normal = torch.stack(
            [radial_from_origin[:, Y], -radial_from_origin[:, X]], dim=1
        )
        normal_norm = torch.linalg.norm(
            closest_point_on_circ_normal, dim=1, keepdim=True
        ).clamp(min=1e-6)
        closest_point_on_circ_normal = (closest_point_on_circ_normal / normal_norm) * 0.1
        des_pos = closest_point_on_circ + closest_point_on_circ_normal

        # Move away from other agents within Lidar visibility range
        object_visible = torch.any(lidar < 0.15, dim=1)
        _, object_dir_index = torch.min(lidar, dim=1)
        object_dir = object_dir_index / lidar.shape[1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=1)
        des_pos_object = current_pos - object_vec * 0.15
        des_pos[object_visible] = des_pos_object[object_visible]

        # Counteract local wind while tracking desired position
        action = torch.clamp(
            (des_pos - current_pos) * 10 - wind,
            min=-u_range,
            max=u_range,
        )

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP pieces for the "stay stable while the arm operates" task.

Two single-responsibility pieces, kept independent of each other (neither reads the other's
state) so each can be reasoned about/tuned on its own:
  - randomize_arm_pose: an event term that gives the arm something to do (a disturbance source
    for the legs), with no neural net and no observation plumbing - deliberately NOT a trained
    policy, since arm task quality is out of scope for this task.
  - arm_leg_contact_penalty: a reward term that penalizes the arm/gripper physically touching
    the robot's own legs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_arm_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    range_scale: float = 0.2,
) -> None:
    """Sample a new random joint-position target within each joint's own soft limits (scaled
    toward its center by `range_scale`) and hold it there via the joints' existing PD actuators.

    Stateless and network-free on purpose (unlike CUBOS_JUANPA's PatasCongeladas, which runs a
    full trained policy every step): the arm here only needs to be a disturbance source for the
    legs to learn to handle, not something being evaluated for its own quality, so it shouldn't
    need its own tuning loop. Works under both `mode="reset"` and `mode="interval"` - the
    EventManager itself decides *when* to call this (same mechanism already driving the
    inherited `events.push_robot`), this function only decides what happens when it's called.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids, :]
    center = (limits[..., 0] + limits[..., 1]) / 2.0
    half_range = (limits[..., 1] - limits[..., 0]) / 2.0 * range_scale
    target = center + torch.empty_like(center).uniform_(-1.0, 1.0) * half_range
    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)


def arm_leg_contact_penalty(
    env: ManagerBasedRLEnv,
    arm_sensor_names: list[str],
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contact force between any dedicated arm/gripper ContactSensor (each filtered
    against every leg body prim - see stability_task.py) and the legs.

    Same per-body-sensor-with-filter_prim_paths_expr pattern platform_climb_mdp's
    foot_on_platform_bonus already uses, for the same underlying reason: filtered
    `force_matrix_w` reporting only works when the sensor's own `prim_path` matches a single
    primitive per environment, so this needs one sensor per arm body (each filtered against
    every leg body) rather than one shared multi-body sensor.
    """
    total = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in arm_sensor_names:
        sensor = env.scene[sensor_name]
        # force_matrix_w: (num_envs, B=1, M=num leg bodies in the filter list, 3)
        force = sensor.data.force_matrix_w[:, 0, :, :]
        in_contact_with_any_leg = (torch.norm(force, dim=-1) > force_threshold).any(dim=-1)
        total += in_contact_with_any_leg.float()
    return total

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registers the "stay stable while the arm operates" task.

Separate file from anymal_d_custom_task.py on purpose (left untouched) - this one ADDS two new
gym ids by building on top of the existing AnymalDFlatCustomNoPayloadEnvCfg, reusing it as-is
via inheritance:
  - Isaac-Stability-Anymal-D-Custom-v0
  - Isaac-Stability-Anymal-D-Custom-Play-v0

Goal: train the LEGS to keep tracking a velocity command and stay upright while the DynaArm
moves through random poses (a disturbance source, not a trained behavior - see
stability_mdp.randomize_arm_pose) and to avoid the arm physically hitting the legs. Explicitly
NOT about arm task quality - no reaching/lifting/gripper reward anywhere here.

What changes vs. the Walk task (anymal_d_custom_task.py):
  - scene.robot -> swapped for ANYMAL_D_DYNAARM_CFG (legs + DynaArm + Robotiq 2F-85; see
    anymal_d_dynaarm.py). enabled_self_collisions=True is baked into that cfg (opposite of
    CUBOS_JUANPA's own combined-body cfg) so an arm-leg collision can even register a contact
    force - see that file's docstring for the accepted spawn-stability risk.
  - events.arm_pose_reset / events.arm_pose_interval -> added (stability_mdp.randomize_arm_pose,
    see that function's docstring for why this is stateless/network-free on purpose).
  - actions.joint_pos.joint_names -> inherited UNCHANGED ([".*HAA", ".*HFE", ".*KFE"]): none of
    the arm/gripper joint names match that pattern, so the arm is already outside the trained
    action space without needing a new restriction.
  - Everything else (velocity-command tracking, torque/accel/action-rate penalties,
    feet_air_time, dof_pos_limits, undesired_contacts, base_contact termination, push/mass/
    friction randomization, PPO runner config) is inherited unchanged.

STAGED IMPLEMENTATION - the arm-vs-leg contact sensors/reward (stability_mdp.
arm_leg_contact_penalty) are NOT wired in yet. They need the DynaArm's real body names, which
`strings`-scanning the USD only gave mangled partial matches for (see anymal_d_dynaarm.py and
scripts/Stability/daily_log.md) - not trustworthy to hardcode. Run
scripts/Stability/print_body_names.py on the VM first, then add the sensors/reward here (one
dedicated ContactSensorCfg per confirmed arm body, filtered against the leg body prims, same
pattern as platform_climb_task.py's shank/box sensors) and log that as its own dated entry.

Usage: same as the other tasks - import this module once before --task resolves. Add, right
after the existing anymal_d_custom_task import:

    import stability_task  # noqa: F401
"""

import gymnasium as gym

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import stability_mdp
from anymal_d_custom_task import AGENT_CFG_ENTRY_POINT, AnymalDFlatCustomNoPayloadEnvCfg  # noqa: E402
from anymal_d_dynaarm import ANYMAL_D_DYNAARM_CFG, ARM_JOINT_NAMES  # noqa: E402


@configclass
class AnymalDStabilityEnvCfg(AnymalDFlatCustomNoPayloadEnvCfg):
    """Walk + stay upright while the DynaArm moves through random disturbance poses."""

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------
        # Robot: legs + DynaArm + Robotiq 2F-85 instead of legs-only.
        # ---------------------------------------------------------------
        self.scene.robot = ANYMAL_D_DYNAARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---------------------------------------------------------------
        # Events: the arm's "operation" - random joint-position targets, held by its own PD
        # actuators. One EventTerm per mode (EventTermCfg.mode is singular) sharing the same
        # function: a fresh pose at episode start, then reshuffled periodically during the
        # episode so the arm keeps moving - interval_range_s matches the inherited
        # events.push_robot's own cadence style (random interval, not fixed).
        # ---------------------------------------------------------------
        arm_asset_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
        self.events.arm_pose_reset = EventTerm(
            func=stability_mdp.randomize_arm_pose,
            mode="reset",
            params={"asset_cfg": arm_asset_cfg},
        )
        self.events.arm_pose_interval = EventTerm(
            func=stability_mdp.randomize_arm_pose,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"asset_cfg": arm_asset_cfg},
        )

        # TODO(stage 2, blocked on confirmed body names - see module docstring): add one
        # ContactSensorCfg per arm body (filtered against every leg body prim) and wire
        # self.rewards.arm_leg_contact_penalty = RewTerm(func=stability_mdp.arm_leg_contact_penalty, ...)


class AnymalDStabilityEnvCfg_PLAY(AnymalDStabilityEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


gym.register(
    id="Isaac-Stability-Anymal-D-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDStabilityEnvCfg",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

gym.register(
    id="Isaac-Stability-Anymal-D-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDStabilityEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

# -----------------------------------------------------------------------------
# To make `--task Isaac-Stability-Anymal-D-Custom-v0` resolvable, add this line to
# scripts/reinforcement_learning/rsl_rl/train.py (and play.py), right after the existing
# `import anymal_d_custom_task` line:
#
#     import stability_task  # noqa: F401
# -----------------------------------------------------------------------------

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

Real body names confirmed via print_body_names.py (2026-10-19): the DynaArm/gripper bodies
are prefixed "dynaarm_" (SHOULDER, UPPERARM, ELBOW, FOREARM, WRIST_1, WRIST_2_base) plus the
gripper's own "base_link" - see ARM_CONTACT_BODY_NAMES below for which ones get a dedicated
collision sensor.

Usage: same as the other tasks - import this module once before --task resolves. Add, right
after the existing anymal_d_custom_task import:

    import stability_task  # noqa: F401
"""

import gymnasium as gym

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import stability_mdp
from anymal_d_custom_task import AGENT_CFG_ENTRY_POINT, AnymalDFlatCustomNoPayloadEnvCfg  # noqa: E402
from anymal_d_dynaarm import ANYMAL_D_DYNAARM_CFG, ARM_JOINT_NAMES  # noqa: E402

# Arm/gripper bodies most likely to swing into leg space - the end-effector chain, not the
# shoulder/upper-arm segments closer to the body. One dedicated ContactSensorCfg per body
# (see the loop below) - same filter_prim_paths_expr constraint as platform_climb_task.py's
# shank/box sensors: filtered reporting only works when the sensor's own prim_path matches a
# single primitive per environment.
#
# Maps body name -> prim path RELATIVE TO "Robot/", not just the body name: the Robotiq 2F-85
# gripper is a separate sub-assembly nested under "robotiq_2f85/" (confirmed by CUBOS_JUANPA's
# own env_brazo.py FrameTransformerCfg target, "Robot/robotiq_2f85/base_link"), unlike the
# dynaarm_* bodies, which sit flat under "Robot/" same as the legs. Learned this the hard way:
# the first print_body_names.py run errored on exactly this ("base_link" with no
# "robotiq_2f85/" prefix resolved to nothing with a contact reporter).
ARM_CONTACT_BODY_SUBPATHS = {
    "dynaarm_FOREARM": "dynaarm_FOREARM",
    "dynaarm_WRIST_1": "dynaarm_WRIST_1",
    "dynaarm_WRIST_2_base": "dynaarm_WRIST_2_base",
    "base_link": "robotiq_2f85/base_link",
}
ARM_CONTACT_BODY_NAMES = list(ARM_CONTACT_BODY_SUBPATHS.keys())
LEG_CONTACT_BODY_NAMES = [
    "LF_THIGH", "LH_THIGH", "RF_THIGH", "RH_THIGH",
    "LF_SHANK", "LH_SHANK", "RF_SHANK", "RH_SHANK",
]


@configclass
class AnymalDStabilityEnvCfg(AnymalDFlatCustomNoPayloadEnvCfg):
    """Walk + stay upright while the DynaArm moves through random disturbance poses."""

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------
        # Robot: legs + DynaArm + Robotiq 2F-85 instead of legs-only.
        # ---------------------------------------------------------------
        self.scene.robot = ANYMAL_D_DYNAARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # ---------------------------------------------------------------
        # Events: the arm's "operation" - random joint-position targets, held by its own PD
        # actuators. One EventTerm per mode (EventTermCfg.mode is singular) sharing the same
        # function: a fresh pose at episode start, then reshuffled periodically during the
        # episode so the arm keeps moving - interval_range_s matches the inherited
        # events.push_robot's own cadence style (random interval, not fixed).
        # Additional diminishing of original volatility in possible velocity, reducing the
        # stress of the task to the robot.
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

        # ---------------------------------------------------------------
        # Arm-vs-leg collision: one dedicated contact sensor per arm/gripper body in
        # ARM_CONTACT_BODY_NAMES, each filtered against every leg body prim - same
        # single-primitive-per-sensor pattern platform_climb_task.py already uses for its
        # shank/box sensors.
        # ---------------------------------------------------------------
        leg_prim_paths = [f"{{ENV_REGEX_NS}}/Robot/{name}" for name in LEG_CONTACT_BODY_NAMES]
        for arm_body, subpath in ARM_CONTACT_BODY_SUBPATHS.items():
            setattr(
                self.scene,
                f"{arm_body.lower()}_leg_contact",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{subpath}",
                    filter_prim_paths_expr=leg_prim_paths,
                ),
            )
        self.rewards.arm_leg_contact_penalty = RewTerm(
            func=stability_mdp.arm_leg_contact_penalty,
            # -1.0 -> -2.0 (2026-10-20): the first run with the softer range_scale=0.2
            # disturbance never registered a single arm-leg contact in 4000 iterations
            # (stayed at flat 0.0 the whole run) - gentler arm motion plus a stable gait now
            # just doesn't reach the legs often, if at all. Doubling the weight now, before
            # the arm's disturbance range gets opened back up later, so the penalty is
            # already meaningfully in place if/when contact starts happening again.
            weight=-2.0,
            params={
                "arm_sensor_names": [f"{name.lower()}_leg_contact" for name in ARM_CONTACT_BODY_NAMES],
            },
        )
        self.rewards.dof_pos_limits.weight = -2.0


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

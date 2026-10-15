"""Registers custom ANYmal-D flat-terrain velocity tasks using:
  - our own USD (converted from anymal_d_simple_description/urdf/anymal.urdf, WITH
    --merge-joints so fixed-joint links collapse into their parent rigid body - this
    avoids massive self-collision/interpenetration explosions at spawn)
  - the real ANYdrive 3.6 actuator net (converted from anydrive_3_6_20210303.txt)

Reuses Isaac Lab's official AnymalDFlatEnvCfg (rewards, observations, terrain, curriculum, etc.)
and only swaps out the robot articulation. The PPO runner config is reused as-is since it
doesn't depend on which physical robot asset is spawned.

Registers TWO tasks so both variants stay available to compare:
  - Isaac-Velocity-Flat-Anymal-D-Custom-v0            -> WITH inspection payload
  - Isaac-Velocity-Flat-Anymal-D-Custom-NoPayload-v0   -> WITHOUT inspection payload
(plus their *-Play-v0 counterparts for play.py)

NOTE: because --merge-joints collapses the LF_FOOT/RF_FOOT/... links (connected to their
shanks via a fixed joint) into the LF_SHANK/RF_SHANK/... bodies, there is no separate
".*FOOT" body anymore - the foot is now physically part of the shank's rigid body. The
official feet_air_time reward term looks for ".*FOOT" and would fail to resolve, so we
override its sensor_cfg to use ".*SHANK" instead. undesired_contacts (".*THIGH") and
base_contact ("base") are unaffected.

The URDF also carries 2 unactuated revolute joints in the WITH-payload variant
(inspection_payload_mount_to_pan, inspection_payload_pan_to_tilt) with no actuator net
behind them, so we restrict the JointPositionActionCfg to only the 12 real leg joints
(the no-payload variant doesn't have these joints at all, so this is a no-op there).

Usage: import this module once before calling gym.make(...) / before train.py resolves
--task, so the gym.register() calls below actually run. See the bottom of this file for the
one-line change needed in scripts/reinforcement_learning/rsl_rl/train.py and play.py.
"""

import gymnasium as gym

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg,
)

from anymal_d_custom import ANYMAL_D_CUSTOM_CFG  # noqa: E402
from anymal_d_custom_no_payload import ANYMAL_D_CUSTOM_NO_PAYLOAD_CFG  # noqa: E402

AGENT_CFG_ENTRY_POINT = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg:"
    "AnymalDFlatPPORunnerCfg"
)


@configclass
class AnymalDFlatCustomEnvCfg(AnymalDFlatEnvCfg):
    """WITH the inspection payload."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_D_CUSTOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*SHANK"
        )
        self.actions.joint_pos.joint_names = [".*HAA", ".*HFE", ".*KFE"]


class AnymalDFlatCustomEnvCfg_PLAY(AnymalDFlatCustomEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AnymalDFlatCustomNoPayloadEnvCfg(AnymalDFlatEnvCfg):
    """WITHOUT the inspection payload."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_D_CUSTOM_NO_PAYLOAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*SHANK"
        )
        self.actions.joint_pos.joint_names = [".*HAA", ".*HFE", ".*KFE"]


class AnymalDFlatCustomNoPayloadEnvCfg_PLAY(AnymalDFlatCustomNoPayloadEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDFlatCustomEnvCfg",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDFlatCustomEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-Custom-NoPayload-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDFlatCustomNoPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-Custom-NoPayload-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDFlatCustomNoPayloadEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

# -----------------------------------------------------------------------------
# To make `--task Isaac-Velocity-Flat-Anymal-D-Custom-v0` (or the -NoPayload- variant)
# resolvable by train.py / play.py, this module needs to be imported before gym.make()
# runs. Add these two lines to scripts/reinforcement_learning/rsl_rl/train.py (and
# play.py), right after the existing `import isaaclab_tasks  # noqa: F401` line:
#
#     import sys
#     sys.path.insert(0, "/home/nexus/assets")
#     import anymal_d_custom_task  # noqa: F401
# -----------------------------------------------------------------------------

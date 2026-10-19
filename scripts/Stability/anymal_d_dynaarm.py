# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-D + DynaArm + Robotiq 2F-85 configuration (legs + arm, same real hardware CUBOS_JUANPA
uses) - for the "stay stable while the arm operates" task only.

USD and actuator net are the same real assets CUBOS_JUANPA already converted (no in-house URDF
exists for this combined body, unlike the legs-only ANYmal-D in anymal_d_custom.py /
anymal_d_custom_no_payload.py) - copied here rather than reconverted, see
assets/anymal_dynaarm_2f85/ and this repo's root README for provenance.

Place anymal_dynaarm_2f85/ and actuator_net/anydrive_3_6_mlp_jit.pt in ~/assets/ on the training
VM (same convention as anymal_d_custom.py - actuator_net/anydrive_3_6_mlp_jit.pt is literally the
same file the Walk/Climb tasks already expect there, nothing new to place twice).

Single responsibility: this module only defines the robot's ArticulationCfg. Task/reward/event
logic lives in stability_task.py / stability_mdp.py.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# -----------------------------------------------------------------------------
# Paths - adjust if you moved things around.
# -----------------------------------------------------------------------------
ASSETS_DIR = os.path.expanduser("~/assets")
DYNAARM_USD_PATH = f"{ASSETS_DIR}/anymal_dynaarm_2f85/anymal_dynaarm.usd"
ACTUATOR_NET_PATH = f"{ASSETS_DIR}/actuator_net/anydrive_3_6_mlp_jit.pt"

# -----------------------------------------------------------------------------
# Default joint pose - legs: same as ANYMAL_D_CUSTOM_NO_PAYLOAD_CFG. Arm/gripper: same
# resting pose CUBOS_JUANPA's _comun.py uses for the real DynaArm + Robotiq 2F-85.
# -----------------------------------------------------------------------------
LEG_JOINT_NAMES = [".*HAA", ".*HFE", ".*KFE"]
ARM_JOINT_NAMES = ["SH_ROT", "SH_FLE", "EL_FLE", "FA_ROT", "WRIST_1", "WRIST_2"]
GRIPPER_PASSIVE_JOINT_NAMES = [
    "right_outer_knuckle_joint", "left_outer_finger_joint", "right_outer_finger_joint",
    "left_inner_finger_joint", "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint",
]

# -----------------------------------------------------------------------------
# Actuator - real ANYdrive 3.6 MLP actuator net, same file/scales as anymal_d_custom.py.
# input_idx kept at [0, 1, 2] to match what Walk/Climb already use - CUBOS uses [0, 2, 4]
# with its own sim.dt=0.005. OPEN ITEM (see scripts/Stability/daily_log.md): confirm our
# actual inherited sim.dt before trusting either choice - affects Walk/Climb too, not just
# this file. Not changed here without that verification.
# -----------------------------------------------------------------------------
ANYDRIVE_3_6_MLP_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=LEG_JOINT_NAMES,
    network_file=ACTUATOR_NET_PATH,
    pos_scale=5.0,
    vel_scale=0.2,
    torque_scale=60.0,
    input_order="vel_pos",
    input_idx=[0, 1, 2],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)
"""Same ANYdrive 3.6 actuator net as anymal_d_custom.py - same real hardware, same file."""

# -----------------------------------------------------------------------------
# Articulation - combined body. enabled_self_collisions=True on purpose (opposite of
# CUBOS_JUANPA's anymal_completo_cfg, which uses False): this task's whole point is
# penalizing arm-vs-leg self contact, which needs self-collision ON to even register a
# contact force to penalize. OPEN ITEM: verify this doesn't explode at spawn from
# convex-hull overlap at the neutral pose (CUBOS turned it off, possibly because of
# exactly this) - see scripts/Stability/print_body_names.py and daily_log.md.
# -----------------------------------------------------------------------------
ANYMAL_D_DYNAARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DYNAARM_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,
            ".*F_HFE": 0.4, ".*H_HFE": -0.4,
            ".*F_KFE": -0.8, ".*H_KFE": 0.8,
            "SH_ROT": 0.0, "SH_FLE": 0.5, "EL_FLE": 1.5,
            "FA_ROT": 0.0, "WRIST_1": 0.5, "WRIST_2": 0.0,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "legs": ANYDRIVE_3_6_MLP_ACTUATOR_CFG,
        "arm_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["SH_ROT", "SH_FLE", "EL_FLE"],
            effort_limit_sim=80.0, velocity_limit_sim=8.0, stiffness=600.0, damping=20.0,
        ),
        "arm_forearm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["FA_ROT", "WRIST_1"],
            effort_limit_sim=60.0, velocity_limit_sim=8.0, stiffness=400.0, damping=15.0,
        ),
        "arm_wrist_rot": ImplicitActuatorCfg(
            joint_names_expr=["WRIST_2"],
            effort_limit_sim=60.0, velocity_limit_sim=8.0, stiffness=200.0, damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=15.0, velocity_limit_sim=4.0, stiffness=3.0, damping=3.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=GRIPPER_PASSIVE_JOINT_NAMES,
            effort_limit_sim=4.0, velocity_limit_sim=4.0, stiffness=0.0, damping=0.5,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""ANYmal-D + DynaArm + Robotiq 2F-85: same real USD/actuator net as CUBOS_JUANPA, same arm/gripper
PD gains (same real hardware) - only enabled_self_collisions differs (True here, on purpose)."""

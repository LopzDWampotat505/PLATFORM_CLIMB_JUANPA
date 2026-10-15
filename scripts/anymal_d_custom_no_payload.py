"""Custom ANYmal-D configuration (WITHOUT the inspection payload) using:
  - our own USD converted from anymal_d_simple_description/urdf/anymal_no_payload.urdf
    (inspection payload removed - ~5.5kg of unactuated, asymmetric mass stripped out)
  - the real ANYdrive 3.6 actuator net (anydrive_3_6_20210303.txt -> anydrive_3_6_mlp_jit.pt)

Same as anymal_d_custom.py, except it points at the no-payload USD. Kept as a separate
file (rather than editing anymal_d_custom.py in place) so both variants stay available
to compare side by side.

Place this file in ~/assets/ on the training VM (importable from your task module).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg
from isaaclab.assets.articulation import ArticulationCfg

# -----------------------------------------------------------------------------
# Paths - adjust if you moved things around.
# -----------------------------------------------------------------------------
ASSETS_DIR = os.path.expanduser("~/assets")
CUSTOM_USD_PATH = f"{ASSETS_DIR}/anymal_d_simple_description/usd/anymal_d_no_payload.usd"
ACTUATOR_NET_PATH = f"{ASSETS_DIR}/actuator_net/anydrive_3_6_mlp_jit.pt"

# -----------------------------------------------------------------------------
# Actuator - real ANYdrive 3.6 MLP actuator net (converted from anydrive_3_6_20210303.txt)
# -----------------------------------------------------------------------------
ANYDRIVE_3_6_MLP_ACTUATOR_CFG_NO_PAYLOAD = ActuatorNetMLPCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
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
"""Configuration for the real ANYdrive 3.6 actuator net (converted from ANYbotics' deployed model)."""

# -----------------------------------------------------------------------------
# Articulation - custom USD (no payload) + custom actuator net
# -----------------------------------------------------------------------------
ANYMAL_D_CUSTOM_NO_PAYLOAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CUSTOM_USD_PATH,
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,
            ".*F_HFE": 0.4,
            ".*H_HFE": -0.4,
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
    actuators={"legs": ANYDRIVE_3_6_MLP_ACTUATOR_CFG_NO_PAYLOAD},
    soft_joint_pos_limit_factor=0.95,
)
"""Custom ANYmal-D (no payload): own URDF-converted USD + real ANYdrive 3.6 actuator-net (MLP)."""

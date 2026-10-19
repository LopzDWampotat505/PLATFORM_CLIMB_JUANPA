# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot verification tool for the two open items blocking stability_task.py's stage 2
(see that file's module docstring and scripts/Stability/daily_log.md):

  1. The DynaArm/gripper's real body (link) names - needed for the arm-vs-leg contact
     sensors' prim_path/filter_prim_paths_expr. A `strings` scan of the USD crate file only
     gave mangled partial matches (SHOULDER, something like UPPERARM/ELBOW, WRIST_1) - not
     trustworthy to hardcode.
  2. Whether enabled_self_collisions=True (set in anymal_d_dynaarm.py, opposite of CUBOS_
     JUANPA's own combined-body cfg) causes a spawn explosion from convex-hull overlap at the
     neutral pose - same failure mode diagnose_spawn.py was originally built to catch for the
     legs-only robot.

Usage:
    ./isaaclab.sh -p scripts/Stability/print_body_names.py --num_envs 1 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, "/home/nexus/assets")
import anymal_d_custom_task  # noqa: F401, E402
import stability_task  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Stability-Anymal-D-Custom-v0"
env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
env = gym.make(TASK, cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]

print("=" * 70)
print("BODY NAMES (copy the arm/gripper ones into stability_task.py's stage 2):")
for i, name in enumerate(robot.data.body_names):
    print(f"  [{i:2d}] {name}")
print("=" * 70)
print("RIGHT AFTER RESET (0 physics steps taken) - checking for spawn explosion:")
print(f"root_pos_w (xyz):      {robot.data.root_pos_w[0].tolist()}")
print(f"root_lin_vel_w:        {robot.data.root_lin_vel_w[0].tolist()}")
print(f"root_ang_vel_w:        {robot.data.root_ang_vel_w[0].tolist()}")
print("=" * 70)

num_joints = env.action_space.shape[1]
zero_action = torch.zeros((args.num_envs, num_joints), device=env.unwrapped.device)

for i in range(5):
    env.step(zero_action)
    print(f"--- after step {i+1} ---")
    print(f"root_pos_w:     {robot.data.root_pos_w[0].tolist()}")
    print(f"root_lin_vel_w: {robot.data.root_lin_vel_w[0].tolist()}")

env.close()
simulation_app.close()

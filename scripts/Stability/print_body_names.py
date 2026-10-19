# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification tool for stability_task.py. Body names are already confirmed (2026-10-19 run -
dynaarm_SHOULDER/UPPERARM/ELBOW/FOREARM/WRIST_1/WRIST_2_base + gripper's base_link), now wired
into stage 2's contact sensors. What's still open:

  Whether enabled_self_collisions=True (set in anymal_d_dynaarm.py, opposite of CUBOS_JUANPA's
  own combined-body cfg) causes real instability - the first 5-step check showed root
  linear velocity growing monotonically (0.08 -> 0.67 m/s over 100ms), which is ambiguous
  between two different causes:
    (a) genuine self-collision overlap pressure (convex-hull meshes overlapping at the
        neutral pose), vs.
    (b) the arm's own disturbance event swinging hard toward a random full-range target
        right at reset, reacting through the free-floating base (Newton's third law) -
        already a known, accepted-for-now effect (see scripts/Stability/daily_log.md).

This run goes longer (60 steps = 1.2s) and prints, every 5 steps: root height, root linear
velocity, projected_gravity_b (near (0,0,-1) = upright; drifting away = tipping), and the max
per-body contact force from each arm-vs-leg sensor (nonzero from step 0 = (a); nonzero only
sometimes, correlated with the arm swinging = (b)).

Usage (after copying anymal_d_dynaarm.py, stability_task.py, stability_mdp.py, and this file
flat into ~/assets/ on the VM, same as every other script in this repo - see root README):
    ./isaaclab.sh -p ~/assets/print_body_names.py --num_envs 1 --headless
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
from stability_task import ARM_CONTACT_BODY_NAMES  # noqa: E402

arm_sensor_names = [f"{name.lower()}_leg_contact" for name in ARM_CONTACT_BODY_NAMES]

print("=" * 70)
print("BODY NAMES:")
for i, name in enumerate(robot.data.body_names):
    print(f"  [{i:2d}] {name}")
print("=" * 70)


def print_state(label: str):
    print(f"--- {label} ---")
    print(f"root height (z):    {robot.data.root_pos_w[0, 2].item():.4f}")
    print(f"root_lin_vel_w:     {robot.data.root_lin_vel_w[0].tolist()}")
    print(f"projected_gravity_b: {robot.data.projected_gravity_b[0].tolist()}  (upright ~= (0,0,-1))")
    for sensor_name in arm_sensor_names:
        sensor = env.unwrapped.scene[sensor_name]
        max_force = sensor.data.force_matrix_w[0].norm(dim=-1).max().item()
        print(f"  max contact force, {sensor_name}: {max_force:.3f} N")


print_state("right after reset (0 physics steps taken)")

num_joints = env.action_space.shape[1]
zero_action = torch.zeros((args.num_envs, num_joints), device=env.unwrapped.device)

for i in range(60):
    env.step(zero_action)
    if (i + 1) % 5 == 0:
        print_state(f"after step {i + 1} ({(i + 1) * 0.02:.2f}s)")

env.close()
simulation_app.close()

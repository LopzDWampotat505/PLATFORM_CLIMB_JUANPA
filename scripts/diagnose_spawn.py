"""Print precise numeric diagnostics right after reset: root height/velocity, joint
positions/velocities, and body positions, BEFORE taking any simulation step, then for
a few steps afterward. Useful for catching spawn issues (explosions, sinking into the
ground, constant resets) with hard numbers instead of guessing from screenshots.

Usage:
    ./isaaclab.sh -p diagnose_spawn.py --num_envs 1 --headless
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

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Velocity-Flat-Anymal-D-Custom-v0"
env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
env = gym.make(TASK, cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]

print("=" * 70)
print("RIGHT AFTER RESET (0 physics steps taken):")
print(f"root_pos_w (xyz):      {robot.data.root_pos_w[0].tolist()}")
print(f"root_lin_vel_w:        {robot.data.root_lin_vel_w[0].tolist()}")
print(f"root_ang_vel_w:        {robot.data.root_ang_vel_w[0].tolist()}")
print(f"joint_names:           {robot.data.joint_names}")
print(f"joint_pos:             {robot.data.joint_pos[0].tolist()}")
print(f"joint_vel:             {robot.data.joint_vel[0].tolist()}")
print(f"body_names:            {robot.data.body_names}")
print(f"body_pos_w (per link): {robot.data.body_pos_w[0].tolist()}")
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

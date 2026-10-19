# Daily log — Stability (walk + stay upright while the DynaArm operates)

Every change to this folder's scripts gets an entry here, with a timestamp, in the same commit
as the change. Most recent first.

## 2026-10-19 16:06 -03 — fixed print_body_names.py's usage example
Docstring said `./isaaclab.sh -p scripts/Stability/print_body_names.py ...` - wrong, that path
doesn't exist inside `~/IsaacLab` (the user hit exactly this running it). Fixed to
`~/assets/print_body_names.py`, matching the flat-deployment convention every other script in
this repo already follows.

## 2026-10-19 15:21 -03 — task created (stage 1, no contact sensors yet)
Third policy: train the LEGS to track a velocity command and stay upright while the DynaArm
moves through random disturbance poses, and (once stage 2 lands) avoid the arm hitting the
legs. Explicitly not about arm task quality - no reaching/lifting/gripper reward.

Added:
- `anymal_d_dynaarm.py` — `ANYMAL_D_DYNAARM_CFG`: ANYmal-D + DynaArm + Robotiq 2F-85, reusing
  CUBOS_JUANPA's already-converted USD + actuator net (copied into `assets/anymal_dynaarm_2f85/`
  and `actuator_net/anydrive_3_6_mlp_jit.pt` at the repo root) rather than reconverting - no
  in-house URDF exists for this combined body. `enabled_self_collisions=True`, the OPPOSITE of
  CUBOS's own combined-body cfg (`False`) - needed so an arm-leg collision can register a
  contact force at all; accepted risk of a spawn explosion from convex-hull overlap at the
  neutral pose, to be checked with `print_body_names.py` below.
- `stability_mdp.py` — `randomize_arm_pose` (stateless event: samples a random joint target
  within each arm joint's own soft limits, held by the existing PD actuators; no neural net,
  deliberately decoupled from arm-task quality) and `arm_leg_contact_penalty` (reward function,
  not yet wired into the task - needs real body names, see below).
- `stability_task.py` — registers `Isaac-Stability-Anymal-D-Custom[-Play]-v0`, inherits
  `AnymalDFlatCustomNoPayloadEnvCfg` unchanged except: robot swap, and two new events
  (`arm_pose_reset` mode="reset", `arm_pose_interval` mode="interval", both calling
  `randomize_arm_pose`). `actions.joint_pos.joint_names` inherited as-is - no arm/gripper joint
  name matches that pattern, so the arm is already outside the trained action space.
- `print_body_names.py` — one-shot verification script (mirrors `diagnose_spawn.py`'s pattern):
  prints `robot.data.body_names` and 5 post-reset steps of root pose/velocity, to resolve both
  open items below in one VM run.

**Open items, not yet resolved (do not trust a real training run until these are closed):**
1. Real arm/gripper body names — a `strings` scan of the USD crate file only gave mangled
   partial matches (`SHOULDER`, something like `UPPERARM`/`ELBOW`, `WRIST_1`) - not reliable
   to hardcode. Run `print_body_names.py`, then add the arm-vs-leg `ContactSensorCfg`s +
   `self.rewards.arm_leg_contact_penalty` to `stability_task.py` (stage 2) with the confirmed
   names, logged as its own dated entry here.
2. Whether `enabled_self_collisions=True` actually explodes at spawn — same
   `print_body_names.py` run covers this (watch the post-reset root_pos_w/root_lin_vel_w for a
   teleport/explosion, same technique `diagnose_spawn.py` already uses).
3. `ActuatorNetMLPCfg.input_idx` — kept at `[0, 1, 2]` here to match Walk/Climb, but CUBOS_
   JUANPA uses `[0, 2, 4]` with its own `sim.dt=0.005`. Need to confirm our tasks' actual
   inherited `sim.dt` (from `AnymalDFlatEnvCfg`) before trusting either value - affects
   Walk/Climb too, not just this task.
4. Whether the inherited `ObservationsCfg.PolicyCfg.joint_pos`/`joint_vel` terms are unfiltered
   (all of the robot's joints) or leg-specific. If unfiltered, the policy already observes the
   arm's current joint state for free now that the robot has more joints - worth confirming
   rather than assuming, since it changes whether the legs are reacting blindly to disturbance
   or with proprioceptive knowledge of the arm's pose.

Not implemented yet, out of scope for now: the later obstacle-avoidance curriculum stage (see
`stability_task.py`'s module docstring - reuse `PlatformHeightCurriculum`'s pattern when that
phase starts).

# Daily log — Stability (walk + stay upright while the DynaArm operates)

Every change to this folder's scripts gets an entry here, with a timestamp, in the same commit
as the change. Most recent first.

## 2026-10-20 11:48 -03 — enabled dof_pos_limits (weight 0.0 -> -2.0)
User added `self.rewards.dof_pos_limits.weight = -2.0` in `stability_task.py`, recycling
`platform_climb_task.py`'s exact fix for the same underlying problem (joints riding their
mechanical limits as a crutch instead of balancing properly) - no new func/params needed,
`dof_pos_limits` was already inherited from `AnymalDFlatCustomNoPayloadEnvCfg`, just disabled
at weight 0.0. Unlike Climb, this now watches ALL actuated joints (legs + arm + gripper), not
just the 12 leg joints, since no `asset_cfg`/joint_names filter was added - worth watching
whether the arm's own joints ever approach their limits too, not just the legs'.

## 2026-10-20 11:44 -03 — arm_leg_contact_penalty weight: -1.0 -> -2.0
First 4000-iteration run with `range_scale=0.2` was a real breakthrough: `base_contact`
(falling) stayed at ~100% through iteration ~2200 (same stuck pattern as the failed first
run), then broke free - by iteration 3800, `base_contact` down to 81.7%, `time_out` up to
18.3%, mean episode length up from a flat ~6s to 369.5 (max 503), `Train_mean_reward` crossed
into positive territory (+3.37) and still climbing with no sign of plateauing. Confirmed
visually in the recorded video - robot stayed upright, arm raised, for 20+ second stretches.

One side effect: `arm_leg_contact_penalty` stayed at flat 0.0 the ENTIRE 4000-iteration run -
not a single registered contact (unlike the previous run's occasional tiny blips). With the
gentler arm motion and a more stable gait, contact is apparently rare-to-nonexistent now.
Doubled the weight to -2.0 anyway, before the arm's range gets opened back up later, so the
penalty already has real teeth whenever contact does start happening again.

## 2026-10-19 18:05 -03 — randomize_arm_pose's range_scale default: 1.0 -> 0.2
First full 2000-iteration training run came back with `Episode_Termination/base_contact`
pinned at ~100% (and `time_out` at 0%) for essentially the entire run, `Train_mean_episode_length`
flat at ~6.1-6.4s from iteration ~100 onward with no improvement - the robot falls almost every
episode, the whole run, never learning to survive a disturbance. Matches what the user saw in
the recorded video (episode ending right as the arm reaches its random target). Root cause:
`range_scale=1.0` samples the arm's target across its FULL joint range every reset/interval -
too violent for an untrained policy to ever experience surviving one, so there was no gradient
signal toward that skill. User lowered the default to 0.2 directly in
`randomize_arm_pose`'s signature (affects both `arm_pose_reset` and `arm_pose_interval` without
touching `stability_task.py`, since neither EventTerm passes `range_scale` explicitly) - more
conservative than the 0.3 first discussed. Need a fresh training run to see whether this is
enough for episodes to start surviving past a few seconds.

Bucketed CSV data that triggered this (100-iteration buckets, full 2000-iteration run):

```
step   base_contact   time_out   mean_episode_length
   0        0.9814      0.0007                 12.52
 100        1.0000      0.0000                  6.40
 500        1.0000      0.0000                  6.20
1000        1.0000      0.0000                  6.13
1500        1.0000      0.0000                  6.11
1900        1.0000      0.0000                  6.09
```

## 2026-10-19 16:17 -03 — fixed gripper sensor prim path (second run crashed)
Second `print_body_names.py` run crashed at env creation: `ContactSensor` at
`.../Robot/base_link` found no body with the contact reporter API. Wrong assumption in stage
2's sensor loop - the Robotiq 2F-85 gripper is a separate sub-assembly nested under
`robotiq_2f85/`, not flat under `Robot/` like the `dynaarm_*` bodies (confirmed by
CUBOS_JUANPA's own `env_brazo.py`, whose `FrameTransformerCfg` target is
`"Robot/robotiq_2f85/base_link"` - should have checked that the first time instead of
assuming a flat hierarchy for every body). The 3 `dynaarm_*` sensors did NOT error, so those
really are flat. Replaced `ARM_CONTACT_BODY_NAMES` (a flat list) with
`ARM_CONTACT_BODY_SUBPATHS` (name -> path relative to `Robot/`) so each sensor gets its own
correct path; `base_link` now resolves to `Robot/robotiq_2f85/base_link`, the rest unchanged.

## 2026-10-19 16:12 -03 — first real run: body names confirmed, stage 2 wired, input_idx fixed
First `print_body_names.py` run on the VM. Resolved:
- **Body names confirmed**: `dynaarm_SHOULDER`, `dynaarm_UPPERARM`, `dynaarm_ELBOW`,
  `dynaarm_FOREARM`, `dynaarm_WRIST_1`, `dynaarm_WRIST_2_base` (arm) + `base_link` and 7
  knuckle/finger bodies (gripper) - clean names, no guessing needed after all.
- **Stage 2 wired** in `stability_task.py`: one `ContactSensorCfg` per body in
  `ARM_CONTACT_BODY_NAMES = [dynaarm_FOREARM, dynaarm_WRIST_1, dynaarm_WRIST_2_base,
  base_link]` (the end-effector chain, not the shoulder/upper-arm segments closer to the
  body), each filtered against all 8 THIGH/SHANK leg bodies, feeding
  `self.rewards.arm_leg_contact_penalty` (weight -1.0, a starting guess - same order of
  magnitude as the inherited `undesired_contacts`, not yet tuned against real data).
- **`sim.dt` confirmed as 0.005s** (from the run's own startup log) - fixed
  `ActuatorNetMLPCfg.input_idx` here to `[0, 2, 4]` (was `[0, 1, 2]`), same fix applied to
  Walk/Climb's actuator cfg - see `scripts/Walk/daily_log.md` for the full explanation.
- **`joint_pos`/`joint_vel` observations confirmed unfiltered** (76-dim obs group, `joint_pos`/
  `joint_vel` each shape (26,) = all 26 actuated joints, legs+arm+gripper) - the legs policy
  already observes the arm's current state proprioceptively, for free, no extra wiring needed.
  Closes open item 4 from the previous entry.

**Still open** - the self-collision spawn-stability question isn't resolved yet: the first
5-step check showed root linear velocity growing monotonically (0.08 -> 0.67 m/s over 100ms),
ambiguous between real self-collision overlap pressure vs. the arm's own disturbance event
swinging hard toward a random target right at reset and reacting through the free-floating
base. Extended `print_body_names.py` to run 60 steps and print root height/velocity/
projected_gravity_b/max arm-vs-leg contact force every 5 steps, to tell the two apart - rerun
needed.

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

# PLATFORM_CLIMB_JUANPA — ANYmal-D learns to climb onto a platform and reach a goal point

Isaac Lab training pipeline for ANYmal-D, built on our own USD (converted from
`anymal_d_simple_description`'s URDF) and the real ANYdrive 3.6 actuator net (not Isaac Lab's
generic one). Three training rutines:

1. **Flat-terrain velocity tracking** (`anymal_d_custom_task.py`) — the base layer: follow a
   random velocity command on flat ground. Everything else is built on top of this by
   inheritance, never by editing it.
2. **Platform climb** (`platform_climb_task.py` + `platform_climb_mdp.py`) — reach a goal
   point on top of a box of randomized position/height, as a stepping stone toward testing a
   perceptive/rough-terrain policy against the real robot's controller later on.
3. **Stability while the arm operates** (`stability_task.py` + `stability_mdp.py`) — same
   legs, now carrying a DynaArm + Robotiq 2F-85 (body reused from CUBOS_JUANPA), training the
   legs to keep walking and avoid the arm hitting them while the arm moves through random
   disturbance poses. Not about arm task quality - no reaching/lifting/gripper reward.

```
scripts/Walk/anymal_d_custom.py              ArticulationCfg: custom USD + ANYdrive 3.6 actuator net (WITH inspection payload)
scripts/Walk/anymal_d_custom_no_payload.py   same, WITHOUT the payload (~5.5 kg of asymmetric unactuated mass removed)
scripts/Walk/anymal_d_custom_task.py         registers the 4 flat-terrain velocity tasks (base layer — never edited)
scripts/Walk/daily_log.md                    timestamped change log for this folder (every change gets an entry)
scripts/Climb/platform_climb_task.py         registers the 2 platform-climb tasks, built on the base layer by inheritance only
scripts/Climb/platform_climb_mdp.py          the command/reward/termination/curriculum code platform_climb_task.py wires in
scripts/Climb/daily_log.md                   timestamped change log for this folder (every change gets an entry)
scripts/Stability/anymal_d_dynaarm.py        ArticulationCfg: legs + DynaArm + Robotiq 2F-85 (USD/net reused from CUBOS_JUANPA)
scripts/Stability/stability_task.py          registers the 2 stability tasks, built on the base layer by inheritance only
scripts/Stability/stability_mdp.py           the arm-disturbance event + arm-leg collision reward stability_task.py wires in
scripts/Stability/print_body_names.py        one-shot verification script - see Known gotchas / this folder's daily_log.md
scripts/Stability/daily_log.md                timestamped change log for this folder (every change gets an entry)
scripts/convert_actuator_net.py              one-off: ANYbotics' flat-text ANYdrive 3.6 net -> TorchScript .pt
scripts/diagnose_spawn.py                    manual debug script: exact post-reset numbers (height, joints, bodies)
assets/anymal_dynaarm_2f85/                  combined-body USD, copied from CUBOS_JUANPA (same real hardware)
actuator_net/anydrive_3_6_mlp_jit.pt         real ANYdrive 3.6 net, copied from CUBOS_JUANPA (same file Walk/Climb expect on the VM)
docs/decisions/0001-...md                    ADR: why the goal success region must stay inside the platform footprint
```

```bash
conda activate env_isaaclab
cd ~/IsaacLab

# flat-terrain walking (base policy)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-Custom-NoPayload-v0 --headless --num_envs 2048

# platform climb (built on top of the walking policy's task config)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-PlatformClimb-Anymal-D-Custom-v0 --headless --num_envs 4096

# stability while the arm operates (stage 1 - run print_body_names.py first, see Known gotchas)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Stability-Anymal-D-Custom-v0 --headless --num_envs 2048

# watch any of them (swap the task id; *-Play-v0 loads the latest checkpoint automatically)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-PlatformClimb-Anymal-D-Custom-Play-v0 --num_envs 50
```

`train.py` and `play.py` need these lines added once, in this order, before `--task` gets
resolved (only add the tasks you're actually using):

```python
sys.path.insert(0, "/home/nexus/assets")
import anymal_d_custom_task   # noqa: F401  (base layer)
import platform_climb_task    # noqa: F401  (platform-climb layer, imports anymal_d_custom_task itself)
import stability_task         # noqa: F401  (stability layer, imports anymal_d_custom_task itself)
```

On the VM, every `.py` under `scripts/` (and its `Walk/`/`Climb`/`Stability/` subfolders) gets
copied **flat** into `~/assets/` — those subfolders are just this repo's own organization,
`sys.path.insert` above only adds `~/assets/` itself, and every import above is a bare module
name with no package prefix.

## Requirements
- Isaac Sim 4.5.0 + Isaac Lab 2.3.0, conda env `env_isaaclab`, Python 3.10 exact.
- NVIDIA GPU (training runs thousands of environments in parallel).
- Assets this repo ships (`assets/anymal_dynaarm_2f85/`, `actuator_net/anydrive_3_6_mlp_jit.pt`
  — copied from CUBOS_JUANPA, same real hardware) still need to be placed in `~/assets/` on
  the training VM, matching the paths `anymal_d_custom*.py`/`anymal_d_dynaarm.py` expect —
  committed here for version control, not auto-deployed.
- Assets this repo does **not** ship (regenerate them with `convert_actuator_net.py` / Isaac
  Lab's `convert_urdf.py` — see Known gotchas — and drop them in `~/assets/` likewise):
  - `anymal_d_simple_description/usd/anymal_d.usd` and `anymal_d_no_payload.usd` (legs-only
    body, Walk/Climb)
  - Trained checkpoints land under `~/IsaacLab/logs/rsl_rl/<experiment>/` on the VM; none are
    committed here yet.

## Architecture

Isaac Lab uses a "manager-based" pattern: a task is a `ManagerBasedRLEnvCfg` — a plain data
object — that declares, per manager (Scene, Observations, Actions, Rewards, Terminations,
Curriculum, Commands, Events), which small Python function to call and with what parameters.
`ManagerBasedRLEnv` reads that config and does the actual calling; adding or removing
behavior almost never means touching simulation code, just wiring a new function into the
right manager slot inside `__post_init__` (or setting an existing one to `None`).

### Inheritance chain

```
AnymalDFlatEnvCfg                     (official isaaclab_tasks: flat terrain, generic rewards/obs)
        │
        │  anymal_d_custom_task.py — swaps in our robot (ArticulationCfg) and redirects the
        │  feet_air_time sensor to .*SHANK (our USD merges FOOT into SHANK). Never edited.
        ▼
AnymalDFlatCustomNoPayloadEnvCfg
        │
        ├──────────────────────────────┬──────────────────────────────┐
        │                               │                               │
        │ platform_climb_task.py:       │ stability_task.py:            │
        │ box, goal command, climbing   │ swaps scene.robot for the     │
        │ rewards, success termination, │ DynaArm-carrying body, adds   │
        │ height curriculum             │ the arm-disturbance events    │
        ▼                               ▼
AnymalDPlatformClimbEnvCfg       AnymalDStabilityEnvCfg
        │                               │
        │ play.py-only overrides        │ play.py-only overrides
        ▼                               ▼
AnymalDPlatformClimbEnvCfg_PLAY   AnymalDStabilityEnvCfg_PLAY
```

Both branches inherit everything else from `AnymalDFlatCustomNoPayloadEnvCfg` unchanged
(velocity tracking, torque/accel/action-rate penalties, fall termination, push/friction
randomization) — they are siblings, neither inherits from the other.

Each level calls `super().__post_init__()` first and only adds/replaces specific manager
entries (e.g. `self.rewards.track_lin_vel_xy_exp = None`). This is what makes "never edit
`anymal_d_custom_task.py`" practical: every new file inherits 100% of the parent's behavior
and only touches what genuinely changes.

### Config vs. runtime: `PlatformGoalCommand`

`PlatformGoalCommandCfg` is declarative (which class to instantiate, which parameters). At
runtime the `CommandManager` builds one `PlatformGoalCommand` instance (a `CommandTerm`
subclass) that actually carries state between steps — sampled goal position, distance to
goal, whether it's been reached. Four independent pieces read that shared state through
`env.command_manager.get_term("goal_position")` rather than talking to each other directly:
`progress_to_goal`, `foot_on_platform_bonus`, the `goal_reached` termination, and
`PlatformHeightCurriculum`. The same mechanism drives the red/green debug-vis marker.

That decoupling is also why it was easy to silently lose behavior during the migration from
the flat-walking task: dropping a reward term (`feet_air_time = None`) doesn't break
anything structurally — the `RewardManager` just stops calling it — but nothing in the
architecture flags that the removed term was covering a real need (teaching proper foot
lift-off) that now has no replacement. See Known gotchas.

## Fundamental Bases

**Environment / conversion pipeline:**
- `--merge-joints` is mandatory in `convert_urdf.py`. Without it, every fixed joint in the
  URDF becomes a separate, spatially-coincident rigid body — with self-collisions enabled,
  PhysX sees dozens of overlapping geometries at spawn and depenetrates explosively.
- That same `--merge-joints` fuses `*_FOOT` into `*_SHANK` — the official `feet_air_time`
  reward looks for `.*FOOT` and silently fails to resolve; redirect its `sensor_cfg` to
  `.*SHANK`.
- The payload URDF has 14 joints but only 12 actuated (the payload's pan/tilt have no
  actuator) — restrict `actions.joint_pos.joint_names` to `.*HAA|.*HFE|.*KFE`.
- `ActuatorNetMLPCfg.input_order` must be `"vel_pos"`, not Isaac Lab's default `"pos_vel"` —
  the original C++ actuation model concatenates velocity before position.
- `setuptools>=81` breaks the Isaac Lab install (drops `pkg_resources`, which `flatdict`
  needs to build) — pin `setuptools<81` before `./isaaclab.sh --install`.
- Isaac Sim 4.5 requires exactly Python 3.10 in the conda env.
- The `anymal_d_custom_task` (and, for platform-climb, `platform_climb_task`) import must be
  added to **both** `train.py` and `play.py`, not just one.

**Platform-climb specific:**
- `ContactSensorCfg.filter_prim_paths_expr` only works when the sensor's own `prim_path`
  matches a single primitive per environment — hence one dedicated `*_box_contact` sensor per
  shank, instead of filtering the shared multi-body `contact_forces` sensor.
- `VisualizationMarkers` draws nothing just because `debug_vis=True` is set in the cfg — a
  `CommandTerm` subclass needs its own `_set_debug_vis_impl`/`_debug_vis_callback` hooks
  (mirrored from `ObjectUniformPoseCommand`).

**Stability specific (unresolved — see [scripts/Stability/daily_log.md](scripts/Stability/daily_log.md)):**
- No in-house URDF exists for ANYmal-D + DynaArm + Robotiq 2F-85 — the combined USD/actuator
  net are reused as-is from CUBOS_JUANPA rather than reconverted.
- `enabled_self_collisions=True` on this robot (opposite of CUBOS_JUANPA's own combined-body
  cfg) so an arm-leg collision can register a contact force at all — unverified whether this
  causes a spawn explosion from convex-hull overlap at the neutral pose. Run
  `print_body_names.py` before trusting any training run.
- The arm/gripper's real body names aren't confirmed yet (a `strings` scan of the USD crate
  file only gave mangled partial matches) — the arm-vs-leg contact sensors/reward aren't wired
  in until `print_body_names.py` confirms them.

## Status

`feet_air_time` is reimplemented (no longer dropped) and `success_height_tol` was tightened to
0.05 to make "standing on the platform" harder to fake by dragging. A short post-fix run (2000
iterations) shows `base_contact` (falling) dropping and `foot_on_platform_bonus` rising faster
than in any prior run — but `dof_pos_limits`/`undesired_contacts` start worsening in lockstep
right as climbing attempts ramp up (~iteration 900), suggesting the emerging climb strategy
leans on riding the knee joint limit as part of the maneuver itself, not just a lazy standing
posture. Not yet resolved. See [scripts/Climb/daily_log.md](scripts/Climb/daily_log.md) for the
detailed, timestamped history behind every change above.

**Stability** is brand new (stage 1 only): the robot swap, velocity tracking, and the
arm-disturbance events are wired, but the arm-vs-leg collision sensors/reward are not yet —
blocked on confirming real body names and spawn stability with self-collision on. Nothing has
been run on the VM yet. See [scripts/Stability/daily_log.md](scripts/Stability/daily_log.md).

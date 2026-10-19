# Daily log — Climb (platform-climb task)

Every change to this folder's scripts gets an entry here, with a timestamp, in the same commit
as the change. Most recent first.

## 2026-10-19 13:13 -03 — reorganized into scripts/Climb/
`platform_climb_mdp.py`, `platform_climb_task.py` moved from `scripts/` into `scripts/Climb/`
(no content changes). Added this daily log per new project convention: every change from now
on gets logged here with a timestamp.

## 2026-10-19 — CSV review after the two fixes below (2000-iteration run)
`base_contact` (falling) drops from ~100% to ~72% and `foot_on_platform_bonus` climbs to ~0.25
much faster than any prior run. But `dof_pos_limits` worsens from ~0 to -0.0265 and
`undesired_contacts` worsens in parallel, both starting almost exactly where
`foot_on_platform_bonus` starts climbing in earnest (~iteration 900) — timing suggests the
emerging climb strategy leans on riding the knee joint limit as part of the climbing maneuver
itself, not just a lazy standing posture (matches what was seen visually: robot stays up
better, but shows signs of putting the knees down again). `Train_mean_reward` keeps climbing
overall despite this. Not yet resolved — options on the table: let it train longer and see if
`dof_pos_limits` (weight -2.0) self-corrects, or raise that weight now given the much clearer
timing correlation than in previous runs.

## 2026-10-19 — success_height_tol tightened, success_radius kept loose
- `success_height_tol`: 0.15 → 0.05 in `platform_climb_task.py`'s `PlatformGoalCommandCfg(...)`
  call — targets the drag/knees regression directly: a dragging/crawling gait keeps the base
  below nominal standing height, so a tighter height check discriminates "standing on the
  platform" from "sprawled across it".
- `success_radius`: kept at 0.5 (already loose enough — the ADR-0001 margin invariant,
  `box_half_footprint (1.3) - goal_xy_jitter (0.7) = 0.6 >= success_radius`, still holds with
  room to spare).
- Synced `PlatformGoalCommandCfg`'s class defaults in `platform_climb_mdp.py` (had drifted to a
  stale 0.2/0.05 from a direct edit that never touched the task's explicit override, so it had
  zero effect) to match what's actually active, with a docstring note on why.

## 2026-10-19 — feet_air_time reimplemented
`platform_climb_task.py` previously set `rewards.feet_air_time = None` because its
`command_name="base_velocity"` param stopped resolving once the task switched to a goal
command — but that dropped the whole term, not just the now-irrelevant zero-command gate,
removing the only reward that penalizes dragging a foot instead of lifting it clear before the
next step (leading hypothesis for the drag-instead-of-step gait seen in `play.py` videos).
Added `platform_climb_mdp.feet_air_time`, reusing the official mechanism verified against
Isaac Lab's real source (`locomotion/velocity/mdp/rewards.py`), same weight/threshold
(0.125 / 0.5, from `velocity_env_cfg.py`), minus the gate.

## 2026-10-15 — repo created
Repo created (`PLATFORM_CLIMB_JUANPA`), moved out of the `anymal-26.06` catkin_ws lab
notebook. Carries forward two fixes already made in the prior session, before this repo
existed:
- `foot_on_platform_bonus`: added an XY-footprint check (was paying out for a shank pressed
  against the box's side face, not actually on top — `body_pos_w` for a shank is the link's
  knee-end origin, not the foot tip).
- `BOX_FOOTPRINT`: 2.0 → 2.6, with `goal_xy_jitter`'s formula corrected to keep it at 0.7 —
  closes the "lean from outside the platform" exploit where `success_radius` exceeded the
  goal-to-edge margin.

See [ADR-0001](../../docs/decisions/0001-goal-success-region-must-not-exceed-platform-footprint.md)
for the full writeup of both fixes above.

# Daily log — Walk (flat-terrain velocity tracking)

Every change to this folder's scripts gets an entry here, with a timestamp, in the same commit
as the change. Most recent first.

## 2026-10-19 16:12 -03 — fixed ActuatorNetMLPCfg.input_idx in both robot variants
`input_idx` was `[0, 1, 2]` in both `anymal_d_custom.py` and `anymal_d_custom_no_payload.py`,
assuming `sim.dt=0.01s` (so 3 consecutive samples = lags 0/0.01/0.02s, matching the real
ANYdrive's training history interval). `scripts/Stability/print_body_names.py`'s `env.reset()`
log confirmed the real inherited `sim.dt` is **0.005s**, not 0.01s - so `[0,1,2]` was actually
sampling lags 0/0.005/0.01s, half of what the real actuator expects. Changed to `[0, 2, 4]`
(every other slot = 0/0.01/0.02s), matching CUBOS_JUANPA's own actuator cfg, which already
uses `[0,2,4]` with this same `sim.dt`. This silently miscalibrated the leg actuator net in
every Walk/Climb run so far (`anymal_d_custom_no_payload.py` is the robot Climb inherits
unchanged) - not catastrophic (the net still produces *a* torque), but not matching the real
hardware's response as closely as it should have.

## 2026-10-19 13:13 -03
Reorganized: `anymal_d_custom.py`, `anymal_d_custom_no_payload.py`, `anymal_d_custom_task.py`
moved from `scripts/` into `scripts/Walk/` (no content changes). Added this daily log per new
project convention: every change from now on gets logged here with a timestamp.

## 2026-10-15
Repo created (`PLATFORM_CLIMB_JUANPA`), moved out of the `anymal-26.06` catkin_ws lab
notebook. These three files came in unchanged from that prior location — no behavior changes,
just a change of home. See the repo's root README for architecture/usage.

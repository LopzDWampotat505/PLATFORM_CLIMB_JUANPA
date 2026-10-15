# ADR-0001: The goal success region must never extend past the platform's physical footprint

## Status
Accepted

## Date
2026-10-14

## Context
The platform-climb task rewards/terminates on `reached_goal`: the robot's base is within
`success_radius` (XY) and `success_height_tol` (Z) of a goal point sampled on top of a box.
Two separate from-scratch training runs showed an inflated success rate (reward curves and
`foot_on_platform_bonus`/`base_contact` looked healthy) that did not match what the recorded
`play.py` videos actually showed: robots registering success without ever placing a foot on
top of the platform.

Zoomed-in video frames (ground-level, not the usual oblique camera angle) revealed two
independent geometric loopholes:

1. **`foot_on_platform_bonus` side-push**: the reward checked a shank's contact + height
   against the box's top surface, but `body_pos_w` for a shank link is the link's own origin
   (near the knee), not the foot tip. A robot standing on the ground with a knee braced
   against the box's vertical side face could satisfy "in contact" + "at/above top height"
   without ever climbing.
2. **Success radius exceeding the goal-to-edge margin**: `goal_xy_jitter` only guarantees the
   sampled goal is `box_half_footprint - goal_xy_jitter` away from the nearest edge. Once
   `success_radius` grew past that margin (0.5 m radius vs. a 0.3 m margin), the success
   circle itself poked past the platform's physical edge for goals sampled near the jitter
   extreme — letting the robot "succeed" by leaning/reaching in from outside instead of
   actually climbing and walking in.

Both loopholes let the policy collect reward and register success through contact geometry
alone, without the base actually resting on top of the platform — defeating the task's whole
point (testing genuine climbing, not reaching/leaning).

## Decision
Treat "the success region must be fully contained within the platform's physical top
surface" as an invariant the task config must maintain, not just an incidental property of
whatever numbers happen to be in place:

1. `foot_on_platform_bonus` (`platform_climb_mdp.py`) now also requires the contact point to
   be horizontally within the box's own XY footprint (`box_half_footprint` param), not just
   above the top-surface height. A side-face push can't satisfy this, since the box's edge is
   exactly the footprint boundary.
2. `BOX_FOOTPRINT` (`platform_climb_task.py`) was increased from 2.0 to 2.6 m specifically so
   that `box_half_footprint - goal_xy_jitter >= success_radius` holds with a safety buffer
   (0.6 m margin vs. 0.5 m success_radius), instead of shrinking `success_radius` back down.
   `goal_xy_jitter`'s formula was corrected in the same change to keep it at 0.7 m (unchanged
   walk-in distance) rather than silently growing with the bigger footprint, which would have
   reopened the same gap at a different box size.

## Alternatives Considered

### Shrink `success_radius` back down instead of growing the platform
- Pros: no change to the box/robot spawn geometry; smaller diff.
- Cons: doesn't fix the root cause (the invariant can be violated again by any future change
  to `goal_xy_jitter` or `success_radius` individually) and makes the success condition
  stricter in a way not motivated by the actual skill being trained.
- Rejected: treats the symptom, not the geometric relationship that has to hold.

### Keep a single combined "contact" reward instead of adding a footprint check
- Pros: simpler reward function.
- Cons: this is exactly the mechanism that created loophole 1 — contact + height alone cannot
  distinguish a foot on top from a knee against the side wall.
- Rejected: the footprint check is the minimal addition that actually closes the gap.

## Consequences
- Any future change to `BOX_FOOTPRINT`, `goal_xy_jitter`, or `success_radius` must re-check
  the margin invariant (`box_half_footprint - goal_xy_jitter >= success_radius`) — both
  constants carry inline comments in `platform_climb_task.py` stating this explicitly.
- The "honest" success rate measured after both fixes is substantially lower than the
  pre-fix numbers (mostly 0-1%, vs. an earlier apparent peak of ~14%), which is expected and
  correct: most of that earlier number was the two loopholes above, not real climbing skill.
  `success_rate_threshold` in `PlatformHeightCurriculum` needs to be recalibrated against
  this lower, honest baseline rather than the old inflated one.
- `foot_on_platform_bonus` now needs one dedicated single-body `ContactSensorCfg` per shank
  (filtered against the box specifically), since filtered contact reporting only works for a
  sensor matching a single primitive per environment.

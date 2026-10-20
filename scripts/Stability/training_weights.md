### Reward terms (positive)

| Term | Weight | What it rewards | Raising the weight | Lowering the weight |
|---|---|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | How close the real velocity (forward/lateral) is to the commanded one | Prioritizes fast/precise walking above everything else | Less urgency to follow the command, more conservative |
| `track_ang_vel_z_exp` | +0.5 | Same, for turning (yaw) | More priority on turning as commanded | Less priority on turning |
| `feet_air_time` | +0.5 | Each foot spending real time in the air before touching down (the anti-drag fix) | More pronounced/higher steps | Risk of dragging again if other rewards offset it |

### Penalty terms (negative)

| Term | Weight | What it penalizes | Raising the penalty (more negative) | Lowering the penalty |
|---|---|---|---|---|
| `flat_orientation_l2` | **-5.0** | The body NOT being level/upright — the largest of all | Prioritizes staying upright almost obsessively (can become stiff/rigid) | More freedom to wobble/absorb the arm's push with the body |
| `lin_vel_z_l2` | -2.0 | Bouncing vertically | Gait stays closer to the ground | Allows more bounce (may help cushion the push) |
| `undesired_contacts` | -1.0 | The THIGH touching the ground (sign of falling/dragging with the leg) | Stricter against fallen postures | More tolerant of that posture |
| `arm_leg_contact_penalty` | **-2.0** (was -1.0) | The arm/gripper actually touching a leg | More priority on avoiding the collision, even at the cost of speed | Tolerates more arm-leg contact |
| `ang_vel_xy_l2` | -0.05 | Unwanted angular wobble (roll/pitch) — exactly what the arm induces | Demands more angular stillness | More tolerance for wobble |
| `action_rate_l2` | -0.01 | The action changing a lot from one step to the next (jitter) | Smoother, more consistent actions | Allows faster/more erratic reactions to the arm's swing |
| `dof_torques_l2` | **-1.0e-05** (corrected; previously misstated as -2.5e-05) | Using a lot of torque (energy cost) | "Lazier" legs | More free force available to react |
| `dof_acc_l2` | -2.5e-07 | Abrupt joint acceleration | Smoother/slower movements | Allows more abrupt reactions |
| `dof_pos_limits` | **-2.0** (was 0.0/disabled) | Forcing a joint near its real mechanical limit | Already enabled — kept worsening in the 8000-iteration run despite -2.0, candidate to raise further if it doesn't improve | Going back to 0.0 would make it blind again, like before |

### Velocity command range (not a reward weight, but affects the task's demands all the same)

`lin_vel_x`, `lin_vel_y`, `ang_vel_z`: inherited at `(-1.0, 1.0)`, **halved to `(-0.5, 0.5)` on 2026-10-20** — not a curriculum (it doesn't widen back out later), it's a permanent ceiling for this task: the real goal is a stable platform for the arm, not fast locomotion. Root cause targeted: `dof_torques_l2`/`dof_acc_l2`/`action_rate_l2` kept worsening because the task demanded speeds more exigent than it actually needs.

**Current status (2026-10-20), summary of what changed since the original table:** `arm_leg_contact_penalty` and `dof_pos_limits` have already been raised/enabled; the velocity range has already been halved. Pending to see in the next run: whether `dof_pos_limits` keeps worsening despite -2.0, and whether the smaller range finally stops `dof_torques_l2`/`dof_acc_l2`/`action_rate_l2` from worsening. See `daily_log.md` for the detailed data behind each change.

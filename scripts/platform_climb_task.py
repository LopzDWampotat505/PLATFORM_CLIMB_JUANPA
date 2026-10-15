# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registers the "climb onto a platform and reach a point" task.

Separate file from anymal_d_custom_task.py on purpose (that file and its registered tasks are
left untouched) - this one only ADDS two new gym ids by building on top of the existing
AnymalDFlatCustomNoPayloadEnvCfg (our own USD + real ANYdrive 3.6 actuator net), reusing it
as-is via inheritance rather than editing it:
  - Isaac-PlatformClimb-Anymal-D-Custom-v0
  - Isaac-PlatformClimb-Anymal-D-Custom-Play-v0

What changes vs. the flat velocity-tracking task (see platform_climb_mdp.py for the new
pieces' implementation):
  - commands.base_velocity -> replaced by commands.goal_position (PlatformGoalCommand): a
    point on top of a box, instead of a random velocity to track.
  - observations.policy.velocity_commands -> replaced by goal_position (same slot, points at
    the new command).
  - rewards.track_lin_vel_xy_exp / track_ang_vel_z_exp -> removed (nothing to track anymore).
  - rewards.feet_air_time -> removed (its command_name="base_velocity" param would otherwise
    KeyError once that command is gone).
  - rewards.progress_to_goal / reached_goal_bonus -> added (potential-based distance reward +
    one-off success bonus).
  - terminations.goal_reached -> added (episode ends in success once on the platform).
  - scene.box -> added (a RigidObject, kinematic so it never falls/moves dynamically, but its
    pose CAN still be written every reset).
  - events.randomize_box -> added (randomizes the box's XY position and, by varying how much
    it's buried into the ground, its exposed height every reset).
  - curriculum.platform_height -> added (widens the box's exposed-height range as the success
    rate improves).
  - Everything else (torque/action-rate penalties, contact terminations, push/mass/friction
    randomization, PPO runner config) is inherited unchanged from AnymalDFlatCustomNoPayloadEnvCfg.

Usage: same as anymal_d_custom_task.py - this module's gym.register() calls only run once it's
imported. See the bottom of this file for the import line needed in train.py/play.py (in
ADDITION to the anymal_d_custom_task import already there - this is a separate module, so it
needs its own import line).
"""

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# Shank body names, in a fixed order shared by every place that needs to iterate per-leg
# (the foot_on_platform_bonus reward and the dedicated contact sensors below) - explicit list
# instead of a ".*SHANK" regex so the order is deterministic and guaranteed to line up.
SHANK_BODY_NAMES = ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import platform_climb_mdp  # noqa: E402
from anymal_d_custom_task import AGENT_CFG_ENTRY_POINT, AnymalDFlatCustomNoPayloadEnvCfg  # noqa: E402
from platform_climb_mdp import PlatformGoalCommandCfg  # noqa: E402

# The box is always this tall; only how much of it is buried changes (see the event below).
# Must be taller than the widest exposed-height target so there's always something to bury.
BOX_FULL_HEIGHT = 0.5

# Top-surface footprint (m). Was 1.0, then 2.0 - 1.0 had robots landing a foot on the vertical
# front/side face instead of clearing onto the flat top. 2.0 fixed that but created a NEW problem
# once success_radius grew to 0.5 below: goal_xy_jitter=0.7 only guarantees the goal is >=
# (BOX_FOOTPRINT/2 - 0.7) from the nearest edge, which was just 0.3m with a 2.0 footprint - less
# than the 0.5m success_radius, so the success circle poked past the edge for goals sampled near
# the jitter extreme, letting the robot "succeed" by leaning in from outside instead of actually
# climbing. 2.6 restores a real margin (1.3 - 0.7 = 0.6m, i.e. > success_radius) without touching
# goal_xy_jitter, so the walk-in distance once on top is unchanged - only the edge is pushed back.
BOX_FOOTPRINT = 2.6


@configclass
class AnymalDPlatformClimbEnvCfg(AnymalDFlatCustomNoPayloadEnvCfg):
    """Reach a goal point on top of a randomly-placed/randomly-"heighted" platform."""

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------
        # Platform
        # ---------------------------------------------------------------
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(BOX_FOOTPRINT, BOX_FOOTPRINT, BOX_FULL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, BOX_FULL_HEIGHT / 2.0)),
        )

        # One dedicated single-body contact sensor per shank, each filtered against just the
        # box. Per ContactSensorCfg's own docstring, filtered reporting (force_matrix_w) only
        # works when the sensor's prim_path matches a SINGLE primitive per environment - the
        # shared `contact_forces` sensor matches all ~13 robot bodies, so a filter on it
        # silently fails (this is what caused the earlier "filter pattern did not match"
        # PhysX error). The shared sensor itself is untouched - still used as-is for
        # undesired_contacts/base_contact.
        for shank_name in SHANK_BODY_NAMES:
            setattr(
                self.scene,
                f"{shank_name.lower()}_box_contact",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{shank_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
                ),
            )

        # ---------------------------------------------------------------
        # Commands: goal point instead of a velocity to track
        # ---------------------------------------------------------------
        self.commands.base_velocity = None
        self.commands.goal_position = PlatformGoalCommandCfg(
            asset_name="robot",
            box_name="box",
            box_half_height=BOX_FULL_HEIGHT / 2.0,
            # box half-extent minus a margin from every edge. The margin MUST stay >=
            # success_radius (0.5 below) + a safety buffer - otherwise the success circle pokes
            # past the edge for goals sampled near the jitter extreme, letting the robot "succeed"
            # by leaning in from outside instead of climbing (confirmed happening with the old
            # 0.3 margin once success_radius grew past it). 0.6 keeps goal_xy_jitter at 0.7
            # (unchanged) now that BOX_FOOTPRINT is 2.6 instead of 2.0 - same walk-in distance,
            # just computed against the bigger platform.
            goal_xy_jitter=(BOX_FOOTPRINT / 2.0) - 0.6,
            # Measured from the full 6000-iteration from-scratch run's CSV export:
            # foot_on_platform_bonus climbed steadily to ~0.71 and base_contact (falling)
            # terminations dropped from ~95% to ~25-30% - the climbing skill itself is solid.
            # But Metrics/goal_position/reached_goal never exceeded ~0.0007 (0.07%) the entire
            # run, ~1000x below the curriculum's 0.6 threshold - the robot reliably gets ON the
            # platform but essentially never lands precisely enough on the sampled point to
            # register as a success. Loosened from 0.3/0.1 so the success condition is reachable
            # without weakening what it actually requires: goal_xy_jitter is still 0.7, so the
            # robot still has to walk to a specific area of the platform, not just arrive anywhere.
            success_radius=0.5,
            success_height_tol=0.15,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        )

        # ---------------------------------------------------------------
        # Observations: same policy-group slot, now pointing at the goal command
        # ---------------------------------------------------------------
        self.observations.policy.velocity_commands = None
        self.observations.policy.goal_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "goal_position"}
        )

        # ---------------------------------------------------------------
        # Rewards: drop velocity-tracking terms, add goal-reaching ones.
        #
        # velocity_toward_goal is the main driver now - dense, every step, same mechanism
        # that actually taught the FLAT task to walk (track_lin_vel_xy_exp), just pointed at
        # the goal instead of a sampled command. progress_to_goal stays too (reduced weight)
        # as a secondary long-horizon refinement signal, since it directly rewards actually
        # closing the distance rather than just instantaneous heading.
        #
        # dof_torques_l2 / dof_acc_l2 / action_rate_l2 are the same smoothness penalties the
        # FLAT task used (tuned for small, gentle flat-walking leg swings) - halved here so
        # the bigger, more dynamic leg motion needed to clear a 15-20cm step isn't fighting
        # against them as hard.
        # ---------------------------------------------------------------
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None  # referenced command_name="base_velocity", now gone
        self.rewards.velocity_toward_goal = RewTerm(
            func=platform_climb_mdp.velocity_toward_goal, weight=1.5, params={"command_name": "goal_position"}
        )
        self.rewards.progress_to_goal = RewTerm(
            func=platform_climb_mdp.progress_to_goal, weight=1.0, params={"command_name": "goal_position"}
        )
        self.rewards.reached_goal_bonus = RewTerm(
            func=platform_climb_mdp.reached_goal_bonus, weight=50.0, params={"command_name": "goal_position"}
        )
        self.rewards.foot_on_platform_bonus = RewTerm(
            func=platform_climb_mdp.foot_on_platform_bonus,
            weight=0.5,
            params={
                "shank_sensor_names": [f"{name.lower()}_box_contact" for name in SHANK_BODY_NAMES],
                "asset_cfg": SceneEntityCfg("robot", body_names=SHANK_BODY_NAMES),
                "box_half_height": BOX_FULL_HEIGHT / 2.0,
                # Added after confirming visually (shank contact sensor markers) that robots were
                # collecting this bonus while standing on the GROUND, shank pressed against the
                # box's side wall - body_pos_w's height check alone can't tell a knee braced
                # against the side from a foot actually on top. This closes that exploit by also
                # requiring the contact to be within the box's own XY footprint.
                "box_half_footprint": BOX_FOOTPRINT / 2.0,
            },
        )
        self.rewards.dof_torques_l2.weight *= 0.5
        self.rewards.dof_acc_l2.weight *= 0.5
        self.rewards.action_rate_l2.weight *= 0.5
        # Three custom posture patches (standing_height, haa_deviation, hfe_kfe_deviation) were
        # tried here and pulled back out - each one chased the specific symptom of the moment
        # (crouching, then sprawled knees) without addressing why the policy could reach those
        # poses at all. Replaced with two OFFICIAL Isaac Lab reward terms that are already
        # present in the inherited config but disabled (weight=0.0) since plain flat-walking
        # never needs them - both are directly grounded in the real ANYdrive hardware limits
        # (LF_HAA's URDF <limit> is only -45deg/+35deg) rather than our own heuristics:
        #   - dof_pos_limits: costs something the moment a joint crosses its soft limit (95% of
        #     the real URDF range) - directly discourages riding joints to their mechanical
        #     end-stops, which is what let the sprawl happen in the first place.
        #   - roll_only_orientation_l2 (ours, but trivial - see platform_climb_mdp.py): only
        #     the ROLL half of the official flat_orientation_l2. Pitch is left completely free
        #     since the body legitimately needs to pitch up to climb - penalizing it the way we
        #     did with standing_height would just reintroduce the commitment-blocking problem.
        #     Roll has no legitimate reason to be large approaching a step head-on.
        self.rewards.dof_pos_limits.weight = -2.0
        self.rewards.roll_orientation = RewTerm(func=platform_climb_mdp.roll_only_orientation_l2, weight=-1.0)

        # ---------------------------------------------------------------
        # Termination: success ends the episode
        # ---------------------------------------------------------------
        self.terminations.goal_reached = DoneTerm(
            func=platform_climb_mdp.goal_reached, params={"command_name": "goal_position"}
        )

        # ---------------------------------------------------------------
        # Event: randomize platform XY position + exposed height every reset.
        # x in (2.5, 3.5): robot's own reset_base event (inherited) spawns it within
        # x in (-0.5, 0.5); with BOX_FOOTPRINT=2.6 (half-width 1.3), this still guarantees
        # >=0.7m clearance from the box's near edge (min 2.5-1.3=1.2) - no spawn overlap, same
        # margin-of-safety intent as before, just recomputed for the bigger box.
        # h_min=0.15: below this the height check barely discriminates "on the platform"
        # from "standing on the ground near it" (success_height_tol=0.1 would blur the two),
        # so the curriculum starts with a step that's unambiguous from the start.
        # ---------------------------------------------------------------
        h_min = 0.15
        # Back to 0.20 for this from-scratch run - a policy that hasn't learned to walk yet
        # needs the easy end of the curriculum, not wherever a previous checkpoint left off.
        # Reminder for next time: current_h_max lives only in the curriculum term's Python
        # instance (rebuilt fresh every launch), NOT in the rsl_rl checkpoint - so if/when
        # resuming THIS run later instead of training fresh again, bump this by hand to
        # whatever Curriculum/platform_height last showed, or every --resume silently restarts
        # the box height from scratch and burns iterations re-climbing mastered levels.
        initial_h_max = 0.20
        self.events.randomize_box = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "pose_range": {
                    "x": (2.5, 3.5),
                    "y": (-0.75, 0.75),
                    "z": (h_min - BOX_FULL_HEIGHT, initial_h_max - BOX_FULL_HEIGHT),
                },
                "velocity_range": {},
            },
        )

        # ---------------------------------------------------------------
        # Curriculum: widen the height range as the success rate improves.
        # (flat terrain has no terrain_levels term to begin with, nothing to remove there)
        # ---------------------------------------------------------------
        self.curriculum.platform_height = CurrTerm(
            func=platform_climb_mdp.PlatformHeightCurriculum,
            params={
                "event_term_name": "randomize_box",
                "command_name": "goal_position",
                "box_full_height": BOX_FULL_HEIGHT,
                "h_min": h_min,
                "initial_h_max": initial_h_max,
                "target_h_max": 0.35,
                "step": 0.01,
                # Was 0.6, then 0.3 - both turned out unreachable (0.6 because of the
                # foot_on_platform_bonus side-push exploit skewing everything; 0.3 because even
                # after fixing that exploit, the 8000-iter run's real success rate
                # (Episode_Termination/goal_reached) only peaked at 14.3% and mostly sustained
                # 2-4% through the back half. 0.08 sits above that sustained band (needs genuine
                # improvement, not just a lucky window) but inside reach of what's already being
                # demonstrated, unlike 0.3.
                "success_rate_threshold": 0.08,
                # advance only once 500 FRESH episode outcomes (not just whichever envs reset
                # on one lucky step) sustain a >8% success rate - a noisy small-sample spike
                # was triggering premature advances that then stalled for the rest of training.
                "success_window_size": 500,
            },
        )


class AnymalDPlatformClimbEnvCfg_PLAY(AnymalDPlatformClimbEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        # Was 2.5 (copied from the flat velocity task's PLAY config, where it never mattered). Here
        # the box's own x-offset from its env's local origin goes up to 3.5 + 1.3 (half footprint,
        # now 2.6/2=1.3) = 4.8m, while a neighboring env's origin is only env_spacing away - with
        # 2.5 that's a real overlap. Since envs have no cross-env collision filtering issue (Isaac
        # Lab isolates physics per env regardless of render distance), this was purely a play.py
        # visualization artifact - robots appearing to walk through a neighboring platform - but it
        # made every recorded video ambiguous to read. 6.0 gives clear separation (>2m clearance
        # even with the bigger footprint).
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


gym.register(
    id="Isaac-PlatformClimb-Anymal-D-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDPlatformClimbEnvCfg",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

gym.register(
    id="Isaac-PlatformClimb-Anymal-D-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:AnymalDPlatformClimbEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

# -----------------------------------------------------------------------------
# To make `--task Isaac-PlatformClimb-Anymal-D-Custom-v0` resolvable, this module needs to be
# imported too - add ONE more line to scripts/reinforcement_learning/rsl_rl/train.py (and
# play.py), right after the existing `import anymal_d_custom_task` line (no need to re-add the
# sys.path.insert, it's already there from that first import):
#
#     import platform_climb_task  # noqa: F401
# -----------------------------------------------------------------------------

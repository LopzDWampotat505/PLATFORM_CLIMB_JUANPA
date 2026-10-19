# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP pieces for the "climb onto a platform and reach a point" task.

Replaces velocity-command tracking with a goal-reaching formulation: each episode, a target
point is sampled on top of a box (whose position and exposed height are randomized every
reset), and the robot is rewarded for closing the distance to it (potential-based: previous
distance minus current distance), with a bonus + episode success when it actually gets there.

Blind by design: the goal's location is read directly from simulation state (privileged
information, like knowing your own GPS coordinates), not sensed via a height-scanner - a
height-scanner would help with *foot placement* on the ledge, not with *knowing where to go*,
which is all this command term needs. Can be added later as a fast-follow if climbing
reliability turns out to need it.

Built against this project's Isaac Lab 2.3.0 install - verified directly from the real source
of CommandTerm (isaaclab/managers/command_manager.py), the ObjectUniformPoseCommand reference
implementation (isaaclab_tasks .../dexsuite/mdp/commands/pose_commands.py), EventManager's
get_term_cfg/set_term_cfg (isaaclab/managers/event_manager.py), and the terrain_levels_vel /
DifficultyScheduler curriculum examples (isaaclab_tasks .../velocity/mdp/curriculums.py and
.../dexsuite/mdp/curriculums.py). The one inferred-but-unverified piece is
`isaaclab.utils.math.subtract_frame_transforms` - a standard utility, should exist, but flag
any ImportError here first if something breaks.
"""

from __future__ import annotations

import torch
from collections import deque
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# Own marker cfg instead of reusing isaaclab.markers.config.POSITION_GOAL_MARKER_CFG - that preset's
# spheres are radius=0.01 (built for close-up pose-alignment tasks), effectively invisible against a
# 2x2m platform viewed from the usual play.py camera distance. Also only 2 prototypes (not 3) since we
# never use a third "invisible" state. Index order matches reached_goal.int(): 0 -> far (red), 1 -> near
# (green), used as marker_indices directly.
GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_position",
    markers={
        "far": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "near": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Command: goal point on top of the platform
# =============================================================================


class PlatformGoalCommand(CommandTerm):
    """Samples a goal point on top of the (randomly placed/sized) platform each episode.

    The goal is static in the world for the whole episode (the platform doesn't move once
    placed), so it's stored and tracked in world frame, and re-expressed in the robot's
    current base frame every step for the observation (since the robot moves, the box does
    not). `resampling_time_range` should be set very large in the task cfg so the only
    resample is the implicit one `CommandTerm.reset()` does at episode start.
    """

    cfg: PlatformGoalCommandCfg

    def __init__(self, cfg: PlatformGoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.box: RigidObject = env.scene[cfg.box_name]

        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        # distance at the start of this step - used by the progress reward (prev - current)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

        self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reached_goal"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PlatformGoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tSuccess radius: {self.cfg.success_radius} m, height tol: {self.cfg.success_height_tol} m\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Goal position in the robot's current base frame. Shape is (num_envs, 3)."""
        return self.goal_pos_b

    def _update_metrics(self):
        distance_xy = torch.norm(self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1)
        height_ok = torch.abs(self.robot.data.root_pos_w[:, 2] - self.goal_pos_w[:, 2]) < self.cfg.success_height_tol
        self.metrics["distance_to_goal"] = distance_xy
        self.metrics["reached_goal"] = ((distance_xy < self.cfg.success_radius) & height_ok).float()

    def _resample_command(self, env_ids: Sequence[int]):
        box_pos_w = self.box.data.root_pos_w[env_ids]
        jitter = torch.empty(len(env_ids), 2, device=self.device).uniform_(
            -self.cfg.goal_xy_jitter, self.cfg.goal_xy_jitter
        )
        self.goal_pos_w[env_ids, 0] = box_pos_w[:, 0] + jitter[:, 0]
        self.goal_pos_w[env_ids, 1] = box_pos_w[:, 1] + jitter[:, 1]
        # box_pos_w is the box's CENTER - its current top surface is center_z + half_height,
        # regardless of how much the randomization event sank it into the ground. We store the
        # TARGET BASE HEIGHT here (surface + standing offset), not the bare surface height,
        # since that's what's actually compared against robot.data.root_pos_w[:, 2].
        self.goal_pos_w[env_ids, 2] = (
            box_pos_w[:, 2]
            + self.cfg.box_half_height
            + self.cfg.goal_height_clearance
            + self.cfg.nominal_stand_height
        )
        # seed prev_distance so the very first progress-reward step isn't a huge spurious spike
        self.prev_distance[env_ids] = torch.norm(
            self.goal_pos_w[env_ids, :2] - self.robot.data.root_pos_w[env_ids, :2], dim=-1
        )

    def _update_command(self):
        # re-express the (static) world-frame goal in the robot's current base frame
        goal_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.goal_pos_w
        )
        self.goal_pos_b[:, :] = goal_pos_b

    # Pattern mirrored directly from ObjectUniformPoseCommand (dexsuite/mdp/commands/pose_commands.py) -
    # CommandTerm only draws anything once these two hooks exist; setting debug_vis=True in the cfg alone
    # does nothing. Reusing POSITION_GOAL_MARKER_CFG's built-in red/green coloring to show, at a glance,
    # whether each env's robot currently satisfies reached_goal - exactly the thing no amount of staring at
    # TensorBoard numbers could answer: is it stopping short of the point, or sitting right on it?
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # "far" (red) while short of the goal, "near" (green) once reached_goal is true
        self.goal_visualizer.visualize(self.goal_pos_w, marker_indices=self.metrics["reached_goal"].int())


@configclass
class PlatformGoalCommandCfg(CommandTermCfg):
    """Configuration for :class:`PlatformGoalCommand`."""

    class_type: type = PlatformGoalCommand

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    box_name: str = "box"
    """Name of the platform/box rigid object in the scene."""

    box_half_height: float = MISSING
    """Half of the box's fixed full height (the box's own visual/collision size never changes -
    only its burial depth does - so this is a constant, not randomized)."""

    goal_xy_jitter: float = 0.2
    """Max +/- offset (m), per axis, for the goal point around the box's XY center.

    Choose this as (box_half_extent - min_edge_margin): sampling each axis independently in
    [-goal_xy_jitter, goal_xy_jitter] then guarantees the goal is always at least
    min_edge_margin away from every edge (including diagonally - the worst case, both axes at
    their max offset simultaneously, is still that far from the nearest edge along each axis).
    This matters for the task's intent: a goal too close to the lip could be satisfied by
    leaning/reaching instead of genuinely climbing and walking in - though in practice the
    height gate in `_update_metrics` already requires the robot's BASE to reach full
    platform-standing height, which leaning alone can't do regardless of how close the goal
    is to the edge. The margin is still worth keeping so the required walk-in distance looks
    like climbing over an obstacle, not just reaching for its rim."""

    goal_height_clearance: float = 0.0
    """Extra height (m) added above the box's top surface when placing the goal point."""

    nominal_stand_height: float = 0.6
    """How high the robot's BASE floats above whatever surface it's standing on (matches the
    z in ANYMAL_D_CUSTOM_NO_PAYLOAD_CFG's init_state.pos). Needed because success is checked
    against robot.data.root_pos_w[:, 2] (the base's height), not a foot/surface height - without
    adding this, the goal height would be the platform's surface height, off by ~0.6m from the
    base height even when the robot is standing perfectly on the platform, and the success
    check would never pass."""

    success_radius: float = 0.2 #changed from 0.3 to 0.2 on 19-10-2026
    """Horizontal distance (m) from the goal within which the robot counts as "there"."""

    success_height_tol: float = 0.05 #changed from 0.1 to 0.05 on 19-10-2026
    """Max base-height difference (m) from the goal height still counted as "there" (so the
    robot must actually be standing at platform height, not just nearby on the ground)."""

    goal_marker_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG
    """Marker shown at the goal point when debug_vis=True - red while unreached, green once
    reached_goal is true. Same _set_debug_vis_impl/_debug_vis_callback pattern as
    ObjectUniformPoseCommand's success_visualizer (dexsuite/mdp/commands/pose_commands.py), but with
    our own 0.15m-radius marker cfg (see GOAL_MARKER_CFG above) instead of the stock 0.01m preset."""


# =============================================================================
# Reward / termination functions
# =============================================================================


def progress_to_goal(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """Potential-based reward: how much closer the robot got to the goal this step."""
    term: PlatformGoalCommand = env.command_manager.get_term(command_name)
    current_distance = term.metrics["distance_to_goal"]
    progress = term.prev_distance - current_distance
    term.prev_distance = current_distance.clone()
    return progress


def reached_goal_bonus(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """One-off bonus on the step the robot reaches the goal (paired with the termination below)."""
    term: PlatformGoalCommand = env.command_manager.get_term(command_name)
    return term.metrics["reached_goal"]


def velocity_toward_goal(
    env: ManagerBasedRLEnv, command_name: str = "goal_position", max_speed: float = 1.0
) -> torch.Tensor:
    """Dense reward: how fast the robot is moving toward the goal RIGHT NOW, projected onto
    the current goal direction. Same spirit as track_lin_vel_xy_exp from the original
    velocity-tracking task (the thing that actually taught it to walk) - just pointed
    dynamically at the goal instead of a randomly sampled command, instead of relying only on
    progress_to_goal's much weaker per-step distance-delta signal. Clamped at max_speed so
    sprinting past it doesn't keep paying out unboundedly.
    """
    term: PlatformGoalCommand = env.command_manager.get_term(command_name)
    goal_dir_b = term.goal_pos_b[:, :2]
    goal_dist = torch.norm(goal_dir_b, dim=-1).clamp(min=1e-3)
    goal_dir_unit = goal_dir_b / goal_dist.unsqueeze(-1)
    vel_toward_goal = torch.sum(term.robot.data.root_lin_vel_b[:, :2] * goal_dir_unit, dim=-1)
    return vel_toward_goal.clamp(max=max_speed)


def feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward long steps - same mechanism as the official isaaclab_tasks' locomotion/velocity
    mdp.feet_air_time (verified against its real source: .../velocity/mdp/rewards.py), minus its
    zero-command gate. That gate exists there to not punish standing still when the velocity
    command asks for it; this task has no velocity command to check against, and the robot can
    never spawn already at platform height, so there's no legitimate "supposed to stand still"
    state here to protect - it should always be stepping until it reaches the goal.

    Dropping this term entirely (rather than just dropping its gate) when the task moved off
    `base_velocity` is the leading hypothesis for the drag-instead-of-step gait seen in play.py
    videos: nothing else in this task's rewards cares whether a foot actually left the ground
    between steps.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.sum((last_air_time - threshold) * first_contact, dim=1)


def foot_on_platform_bonus(
    env: ManagerBasedRLEnv,
    shank_sensor_names: list[str],
    asset_cfg: SceneEntityCfg,
    box_half_height: float,
    box_half_footprint: float,
    box_name: str = "box",
    height_tol: float = 0.05,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Small bonus per foot/shank that is (a) in contact with the box specifically (not the
    ground), (b) at or above the box's current top surface height, AND (c) horizontally within
    the box's top footprint - i.e. actually stepped on TOP of the platform, not just pressed
    against its front/side face.

    (b) alone is NOT enough to rule out the side face: `body_pos_w` for a shank is the link's
    own origin (up near the knee, not the foot tip), so a robot standing at the base of the box
    with its knee already above `box_top_z - height_tol` and its shank touching the vertical
    side wall satisfies both the old contact and height checks without ever climbing - confirmed
    visually (shank/foot sensor debug markers showed feet resting on the GROUND right at the
    box's base, not its top, while this reward was still paying out). (c) closes that gap: the
    contact point also has to sit within the box's own XY footprint, which a side-face push can't
    satisfy since the box's edge is exactly the footprint boundary.

    Needs one DEDICATED single-body ContactSensorCfg per shank (each with
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"], set in the task cfg) rather than the shared
    multi-body `contact_forces` sensor - per ContactSensorCfg's own docstring, filtered contact
    reporting only works when the sensor's prim_path matches a SINGLE primitive per
    environment; our shared sensor matches all ~13 robot bodies, so filtering silently fails
    there (which is exactly what caused the PhysX "filter pattern did not match" error).

    `shank_sensor_names` and `asset_cfg.body_ids` must be in the SAME shank order (e.g. both
    LF, RF, LH, RH) - asset_cfg should be built with an explicit body_names list, not a regex,
    so the order is deterministic and guaranteed to match.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    box: RigidObject = env.scene[box_name]
    box_top_z = box.data.root_pos_w[:, 2] + box_half_height
    box_pos_xy = box.data.root_pos_w[:, :2]

    total = torch.zeros(env.num_envs, device=env.device)
    for sensor_name, body_id in zip(shank_sensor_names, asset_cfg.body_ids):
        sensor = env.scene[sensor_name]
        # force_matrix_w: (num_envs, B=1, M=1, 3) - this one shank vs. just the box
        force = sensor.data.force_matrix_w[:, 0, 0, :]
        in_contact = torch.norm(force, dim=-1) > force_threshold
        body_pos = robot.data.body_pos_w[:, body_id, :]
        on_top = body_pos[:, 2] > (box_top_z - height_tol)
        within_footprint = torch.all(torch.abs(body_pos[:, :2] - box_pos_xy) < box_half_footprint, dim=-1)
        total += (in_contact & on_top & within_footprint).float()

    return total


def goal_reached(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """Termination condition: episode ends in success once the robot is on the platform."""
    term: PlatformGoalCommand = env.command_manager.get_term(command_name)
    return term.metrics["reached_goal"].bool()


def roll_only_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize ROLL only (sideways tilt), leaving pitch completely free.

    Same idea as the official `mdp.flat_orientation_l2` (penalizing the xy-components of
    projected_gravity_b), but split so only the roll component (index 1 - gravity's y in the
    body frame) counts. Pitch (index 0) is deliberately excluded: climbing a step head-on
    needs the body to pitch up/forward, and penalizing that the way we tried with
    standing_height (symmetric, not axis-aware) blocked commitment to the climb entirely.
    Roll, on the other hand, has no legitimate reason to be large when approaching a step
    straight-on - a robot that's rolling is sprawling sideways, not climbing.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


# =============================================================================
# Curriculum: widen the box's exposed-height range as the robot succeeds more
# =============================================================================


class PlatformHeightCurriculum(ManagerTermBase):
    """Grows the platform's exposed-height range once the success rate is high enough.

    Mirrors the shared-state style of isaaclab_tasks' DifficultyScheduler (dexsuite) and the
    direct-mutation style of terrain_levels_vel (official velocity task curriculum), simplified
    to a single shared scalar (current_h_max) instead of per-env adaptive difficulty, since we
    only need one global knob (box height), not per-env terrain levels.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.current_h_max = self.cfg.params.get("initial_h_max", 0.05)
        window_size = self.cfg.params.get("success_window_size", 500)
        # outcomes (1/0 per finished episode) across ALL envs, not just the ones that happened
        # to reset on one particular step - a handful of lucky simultaneous successes used to
        # cross the threshold on noise alone and advance the curriculum before it was actually
        # mastered, which is exactly why progress kept stalling right after each jump.
        self._outcomes: deque[float] = deque(maxlen=window_size)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        event_term_name: str = "randomize_box",
        command_name: str = "goal_position",
        box_full_height: float = 0.5,
        h_min: float = 0.03,
        initial_h_max: float = 0.05,
        target_h_max: float = 0.35,
        step: float = 0.01,
        success_rate_threshold: float = 0.6,
        success_window_size: int = 500,
    ) -> float:
        term: PlatformGoalCommand = env.command_manager.get_term(command_name)
        self._outcomes.extend(term.metrics["reached_goal"][env_ids].tolist())

        # only ever consider advancing once the window is completely full of FRESH outcomes -
        # a small/partial sample right after a jump (or at startup) can't trigger anything.
        if len(self._outcomes) == self._outcomes.maxlen:
            success_rate = sum(self._outcomes) / len(self._outcomes)
            if success_rate > success_rate_threshold:
                self.current_h_max = min(self.current_h_max + step, target_h_max)
                # next advance needs its own sustained evidence at the new (harder) height,
                # not leftover successes carried over from the easier level we just left.
                self._outcomes.clear()

        event_cfg = env.event_manager.get_term_cfg(event_term_name)
        event_cfg.params["pose_range"]["z"] = (h_min - box_full_height, self.current_h_max - box_full_height)
        env.event_manager.set_term_cfg(event_term_name, event_cfg)

        return self.current_h_max

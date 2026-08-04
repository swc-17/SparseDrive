"""NAVSIM ``AbstractAgent`` wrapper for original SparseDrive.

Importable only in an environment that has BOTH the navsim/nuplan stack and
the SparseDrive mm-stack (see docs/navsim_eval_openloop.md for the
two-process fallback used while no such combined local env exists).

The wrapper:

- requests all eight cameras on all four history frames;
- resets ALL SparseDrive temporal banks per scenario token (no state may
  leak between tokens, evaluation order, or two-stage frames);
- accepts synthetic-render stage-two frames exactly like real frames
  (in-memory RGB is converted to the training BGR layout before the same
  undistort/resize/normalize pipeline);
- outputs eight SE(2) poses at 0.5 s (SD->NAVSIM inverse
  ``x_nav = y_sd, y_nav = -x_sd``; headings derived from consecutive
  positions with a stationary fallback).
"""
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from .agent_input_adapter import frames_from_agent_input
from .coord import sd_plan_to_navsim_poses
from .runner import SparseDriveRunner


class SparseDriveNavsimAgent(AbstractAgent):
    """Original SparseDrive (detection+map+motion+planning) as a NAVSIM agent."""

    requires_scene = False

    def __init__(
        self,
        config_path,
        checkpoint_path,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
        device: str = "cuda:0",
    ):
        super().__init__(trajectory_sampling)
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._runner = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._runner = SparseDriveRunner(
            self._config_path, self._checkpoint_path, device=self._device
        )
        assert self._runner.ego_fut_ts == self._trajectory_sampling.num_poses, (
            f"config ego_fut_ts={self._runner.ego_fut_ts} != "
            f"trajectory_sampling.num_poses="
            f"{self._trajectory_sampling.num_poses}"
        )

    def get_sensor_config(self) -> SensorConfig:
        # all eight cameras on all four history frames, no lidar
        return SensorConfig(
            cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
            cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True,
            lidar_pc=False,
        )

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        assert self._runner is not None, "call initialize() first"
        frames = frames_from_agent_input(agent_input)
        # predict_scenario resets all temporal banks before the replay
        plan_sd = self._runner.predict_scenario(frames)
        poses = sd_plan_to_navsim_poses(plan_sd)
        return Trajectory(poses, self._trajectory_sampling)

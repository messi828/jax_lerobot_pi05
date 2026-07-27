#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_leader_spacebar_follow")
@dataclass
class SOLeaderSpacebarFollowTeleopConfig(TeleoperatorConfig):
    """Independent SO101 leader teleop with spacebar intervention events.

    This teleop mirrors follower joints to the leader and emits intervention events
    when spacebar toggles intervention mode. When intervention is on, leader motion
    is converted to ee delta and sent to the pipeline (requires urdf_path).
    """

    port: str = "/dev/so101_leader_blue"
    follower_port: str = "/dev/so101_follower_black"
    use_degrees: bool = True
    mirror_follower: bool = True
    use_gripper: bool = True
    # Leader bus needs calibration for sync_write; reuse so101_leader calibration file
    leader_calibration_path: str | None = None
    # For intervention: convert leader joints to ee delta (same robot as follower)
    urdf_path: str | None = None
    target_frame_name: str = "gripper_frame_link"
    end_effector_step_sizes: dict[str, float] | None = None  # e.g. {"x": 0.02, "y": 0.02, "z": 0.02}

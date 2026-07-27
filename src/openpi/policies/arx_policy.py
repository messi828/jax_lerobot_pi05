"""ARX Arm policy transforms for OpenPI.

This module provides transforms to convert ARX robot data to the format expected by Pi 0.5 models.

ARX Robot Observation Features (from LeRobot dataset):
- end_effector_pos.{x, y, z, roll, pitch, yaw}: 6D end-effector pose
- {joint_1..joint_6}.{pos, vel, cur}: 6 arm joints with position, velocity, current
- gripper.{pos, vel, cur}: gripper state

ARX Robot Action Features (delta actions):
- delta_{x, y, z, roll, pitch, yaw}.pos: 6D delta end-effector pose
- delta_gripper.pos: delta gripper position
"""

import dataclasses
import logging
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        # End-effector pose
        "observation.end_effector_pos.x": np.random.rand(),
        "observation.end_effector_pos.y": np.random.rand(),
        "observation.end_effector_pos.z": np.random.rand(),
        "observation.end_effector_pos.roll": np.random.rand(),
        "observation.end_effector_pos.pitch": np.random.rand(),
        "observation.end_effector_pos.yaw": np.random.rand(),
        # Joint positions
        "observation.joint_1.pos": np.random.rand(),
        "observation.joint_2.pos": np.random.rand(),
        "observation.joint_3.pos": np.random.rand(),
        "observation.joint_4.pos": np.random.rand(),
        "observation.joint_5.pos": np.random.rand(),
        "observation.joint_6.pos": np.random.rand(),
        # Joint velocities
        "observation.joint_1.vel": np.random.rand(),
        "observation.joint_2.vel": np.random.rand(),
        "observation.joint_3.vel": np.random.rand(),
        "observation.joint_4.vel": np.random.rand(),
        "observation.joint_5.vel": np.random.rand(),
        "observation.joint_6.vel": np.random.rand(),
        # Joint currents
        "observation.joint_1.cur": np.random.rand(),
        "observation.joint_2.cur": np.random.rand(),
        "observation.joint_3.cur": np.random.rand(),
        "observation.joint_4.cur": np.random.rand(),
        "observation.joint_5.cur": np.random.rand(),
        "observation.joint_6.cur": np.random.rand(),
        # Gripper
        "observation.gripper.pos": np.random.rand(),
        "observation.gripper.vel": np.random.rand(),
        "observation.gripper.cur": np.random.rand(),
        # Camera
        "observation.images.front": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        # Prompt
        "prompt": "ARX manipulation task",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ARXInputs(transforms.DataTransformFn):
    """Transform ARX robot data to model input format.
    
    This handles:
    - State extraction from observation.state (27D -> 7D -> 32D)
    - Image preprocessing
    - Action extraction (7D -> 32D)
    
    The 7D ARX data is padded to 32D and placed at indices 7-13 (right arm position):
    - Indices 0-6: Left arm (padded with zeros)
    - Indices 7-13: Right arm (ARX 7D data)
    - Indices 14-31: Unused (padded with zeros)
    
    Args:
        model_type: The type of model being used (PI0, PI05, or PI0_FAST)
        use_joint_state: If True, use joint positions instead of end-effector pose for state.
                        Default is False (use end-effector pose).
        state_names: Optional list of feature names in observation.state. If provided,
                    will use name-based mapping instead of hardcoded indices.
    """
    
    model_type: _model.ModelType
    use_joint_state: bool = False
    state_names: list[str] | None = None
    
    # ARX data indices in 32D space (right arm position)
    ARX_START_IDX: int = 7
    ARX_END_IDX: int = 14  # exclusive
    PADDED_DIM: int = 32
    
    # Track if warning has been issued (to avoid repeated warnings)
    _warned_no_names: ClassVar[bool] = False
    
    # Expected feature names for name-based mapping
    EXPECTED_EE_NAMES: ClassVar[list[str]] = [
        "end_effector_pos.x",
        "end_effector_pos.y",
        "end_effector_pos.z",
        "end_effector_pos.roll",
        "end_effector_pos.pitch",
        "end_effector_pos.yaw",
    ]
    EXPECTED_JOINT_NAMES: ClassVar[list[str]] = [
        "joint_1.pos", "joint_2.pos", "joint_3.pos",
        "joint_4.pos", "joint_5.pos", "joint_6.pos",
    ]
    EXPECTED_GRIPPER_NAME: ClassVar[str] = "gripper.pos"
    
    # If True, when actions are present (training), override gripper action target
    # with absolute gripper value from observation state instead of using dataset deltas.
    gripper_action_absolute: bool = True
    
    def _build_index_map(self) -> dict[str, int]:
        """Build a mapping from feature names to array indices.
        
        Returns:
            Dictionary mapping feature names to their indices in observation.state array.
        """
        if self.state_names is None:
            return {}
        
        # Create mapping: name -> index
        name_to_idx = {}
        for idx, name in enumerate(self.state_names):
            name_to_idx[name] = idx
        
        return name_to_idx
    
    def _extract_state_by_names(self, obs_state: np.ndarray) -> np.ndarray:
        """Extract 7D state using name-based mapping.
        
        Args:
            obs_state: Full observation state array (27D)
            
        Returns:
            7D state vector (end-effector pose + gripper or joint positions + gripper)
        """
        index_map = self._build_index_map()
        
        if self.use_joint_state:
            # Extract joint positions by name
            state_values = []
            for joint_name in self.EXPECTED_JOINT_NAMES:
                if joint_name not in index_map:
                    raise ValueError(
                        f"Required joint feature '{joint_name}' not found in state_names. "
                        f"Available names: {self.state_names}"
                    )
                state_values.append(obs_state[index_map[joint_name]])
            
            # Add gripper position
            if self.EXPECTED_GRIPPER_NAME not in index_map:
                raise ValueError(
                    f"Required gripper feature '{self.EXPECTED_GRIPPER_NAME}' not found in state_names. "
                    f"Available names: {self.state_names}"
                )
            state_values.append(obs_state[index_map[self.EXPECTED_GRIPPER_NAME]])
            
            return np.array(state_values, dtype=np.float32)
        else:
            # Extract end-effector pose by name
            state_values = []
            for ee_name in self.EXPECTED_EE_NAMES:
                if ee_name not in index_map:
                    raise ValueError(
                        f"Required end-effector feature '{ee_name}' not found in state_names. "
                        f"Available names: {self.state_names}"
                    )
                state_values.append(obs_state[index_map[ee_name]])
            
            # Add gripper position
            if self.EXPECTED_GRIPPER_NAME not in index_map:
                raise ValueError(
                    f"Required gripper feature '{self.EXPECTED_GRIPPER_NAME}' not found in state_names. "
                    f"Available names: {self.state_names}"
                )
            state_values.append(obs_state[index_map[self.EXPECTED_GRIPPER_NAME]])
            
            return np.array(state_values, dtype=np.float32)
    
    def _extract_state_by_indices(self, obs_state: np.ndarray) -> np.ndarray:
        """Extract 7D state using hardcoded indices (fallback method).
        
        This is kept for backward compatibility but should be avoided.
        Use name-based mapping by providing state_names instead.
        
        Args:
            obs_state: Full observation state array (27D)
            
        Returns:
            7D state vector (end-effector pose + gripper or joint positions + gripper)
        """
        if self.use_joint_state:
            # Hardcoded indices for joint positions
            # Assumes: [joint_1.pos(6), joint_1.vel(7), joint_1.cur(8), ...]
            return np.array([
                obs_state[6],   # joint_1.pos
                obs_state[9],   # joint_2.pos
                obs_state[12],  # joint_3.pos
                obs_state[15],  # joint_4.pos
                obs_state[18],  # joint_5.pos
                obs_state[21],  # joint_6.pos
                obs_state[24],  # gripper.pos
            ], dtype=np.float32)
        else:
            # Hardcoded indices for end-effector pose
            # Assumes: end_effector at indices 0-5, gripper.pos at index 24
            return np.concatenate([
                obs_state[0:6],   # end-effector: x, y, z, roll, pitch, yaw
                obs_state[24:25], # gripper position
            ], axis=0)

    def __call__(self, data: dict) -> dict:
        # Extract state from observation.state (27D tensor)
        obs_state = np.asarray(data["observation.state"], dtype=np.float32)
        
        # Build 7D state vector using name-based mapping if available
        if self.state_names is not None:
            arx_state = self._extract_state_by_names(obs_state)
        else:
            # Warn once if using hardcoded indices
            if not ARXInputs._warned_no_names:
                logging.warning(
                    "ARXInputs: No state_names provided. Using hardcoded indices for state extraction. "
                    "This may fail if dataset structure differs from expected format. "
                    "Consider providing state_names from dataset metadata for robust index mapping."
                )
                ARXInputs._warned_no_names = True
            
            # Fallback to hardcoded indices (based on verified dataset structure)
            arx_state = self._extract_state_by_indices(obs_state)
        
        # Pad 7D state to 32D: place ARX state at indices 7-13 (right arm position)
        state = np.zeros(self.PADDED_DIM, dtype=np.float32)
        state[self.ARX_START_IDX:self.ARX_END_IDX] = arx_state

        # Parse camera images
        front_image = _parse_image(data["observation.images.front"])
        wrist_image = _parse_image(data["observation.images.wrist"])

        # Organize images based on model type
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # Pi0/Pi0.5 expects 3 images: base_0, left_wrist, right_wrist
                # We use front camera as base_0 and wrist camera as left_wrist
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (front_image, wrist_image, np.zeros_like(front_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # Pi0-FAST expects: base_0, base_1, wrist_0
                # We use front camera as base_0 and wrist camera as wrist_0
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (front_image, np.zeros_like(front_image), wrist_image)
                image_masks = (np.True_, np.False_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Handle actions if present (for training)
        if "actions" in data:
            arx_actions = np.asarray(data["actions"])  # Shape: (action_horizon, 7)
            # Pad to 32D: place ARX actions at indices 7-13 (right arm position)
            padded_actions = np.zeros(
                (arx_actions.shape[0], self.PADDED_DIM), dtype=np.float32
            )
            padded_actions[:, self.ARX_START_IDX:self.ARX_END_IDX] = arx_actions
            
            # If using absolute gripper supervision, overwrite gripper column with absolute gripper from state
            if self.gripper_action_absolute:
                # arx_state holds the 7D state we extracted above; last element is gripper absolute
                gripper_abs = float(arx_state[-1])
                # Index 13 in 32D corresponds to ARX gripper (indices 7-13 are ARX)
                padded_actions[:, self.ARX_END_IDX - 1] = gripper_abs
            inputs["actions"] = padded_actions

        # Handle prompt
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PadARXActionsTo32D(transforms.DataTransformFn):
    """Pad 7D ARX actions to 32D for Pi0.5 compatibility.
    
    This transform is used when discrete_state_input=True (Pi0.5 mode).
    State is tokenized into the prompt, so only actions need padding.
    
    The 7D ARX actions are mapped to indices 7-13 (right arm position) in the 32D space:
    - Indices 0-6: Left arm (padded with zeros)
    - Indices 7-13: Right arm (ARX actions)
    - Indices 14-31: Unused (padded with zeros)
    """
    
    # ARX action indices in 32D space (right arm position)
    ARX_START_IDX: int = 7
    ARX_END_IDX: int = 14  # exclusive
    PADDED_DIM: int = 32
    
    def __call__(self, data: dict) -> dict:
        # Only pad actions (state is handled by tokenizer in Pi0.5 mode)
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            
            # Check if actions are already 32D (skip padding)
            if actions.shape[-1] == self.PADDED_DIM:
                # Already padded, no need to pad again
                return data
            elif actions.shape[-1] == 7:
                # Pad 7D to 32D
                actions_32d = np.zeros(
                    (actions.shape[0], self.PADDED_DIM), dtype=np.float32
                )
                actions_32d[:, self.ARX_START_IDX:self.ARX_END_IDX] = actions
                data["actions"] = actions_32d
            else:
                raise ValueError(
                    f"Expected actions to be 7D or 32D, got {actions.shape[-1]}D. "
                    f"Actions shape: {actions.shape}"
                )
        
        return data


@dataclasses.dataclass(frozen=True)
class PadARXTo32D(transforms.DataTransformFn):
    """Pad 7D ARX state/actions to 32D for Pi0 compatibility.
    
    This transform is used when discrete_state_input=False (Pi0 mode).
    Both state and actions need padding.
    
    The 7D ARX data is mapped to indices 7-13 (right arm position) in the 32D space:
    - Indices 0-6: Left arm (padded with zeros)
    - Indices 7-13: Right arm (ARX data)
    - Indices 14-31: Unused (padded with zeros)
    """
    
    # ARX action indices in 32D space (right arm position)
    ARX_START_IDX: int = 7
    ARX_END_IDX: int = 14  # exclusive
    PADDED_DIM: int = 32
    
    def __call__(self, data: dict) -> dict:
        # Pad state from 7D to 32D
        if "state" in data:
            state_7d = np.asarray(data["state"], dtype=np.float32)
            state_32d = np.zeros(self.PADDED_DIM, dtype=np.float32)
            state_32d[self.ARX_START_IDX:self.ARX_END_IDX] = state_7d
            data["state"] = state_32d
        
        # Pad actions from 7D to 32D
        if "actions" in data:
            actions_7d = np.asarray(data["actions"], dtype=np.float32)
            actions_32d = np.zeros(
                (actions_7d.shape[0], self.PADDED_DIM), dtype=np.float32
            )
            actions_32d[:, self.ARX_START_IDX:self.ARX_END_IDX] = actions_7d
            data["actions"] = actions_32d
        
        return data


@dataclasses.dataclass(frozen=True)
class DiscretizeARXGripper(transforms.DataTransformFn):
    """Discretize ARX gripper actions to {-1, 0, 1}.
    
    This transform converts continuous gripper values to discrete values:
    - If gripper > threshold: set to 1 (open)
    - If gripper < -threshold: set to -1 (close)
    - Otherwise: set to 0 (no change)
    
    This can be applied to both 7D actions (during training) and 32D actions (after padding).
    
    Args:
        threshold: Threshold value for discretization. Default is 0.02.
        gripper_idx_7d: Index of gripper in 7D action space. Default is 6.
        gripper_idx_32d: Index of gripper in 32D action space (ARX data at indices 7-13). Default is 13.
    """
    
    threshold: float = 0.02
    gripper_idx_7d: int = 6  # Gripper is the 7th dimension (index 6) in 7D ARX actions
    gripper_idx_32d: int = 13  # Gripper is at index 13 in 32D space (ARX data at indices 7-13)
    
    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        
        actions = np.asarray(data["actions"], dtype=np.float32)
        
        # Determine if actions are 7D or 32D
        if actions.shape[-1] == 7:
            # 7D actions: gripper is at index 6
            gripper_idx = self.gripper_idx_7d
        elif actions.shape[-1] == 32:
            # 32D actions: gripper is at index 13 (ARX data at indices 7-13)
            gripper_idx = self.gripper_idx_32d
        else:
            # Unknown action dimension, skip discretization
            logging.warning(
                f"DiscretizeARXGripper: Unknown action dimension {actions.shape[-1]}, skipping discretization."
            )
            return data
        
        # Discretize gripper values
        gripper_values = actions[..., gripper_idx]
        discretized = np.where(
            gripper_values > self.threshold,
            1.0,  # Open
            np.where(
                gripper_values < -self.threshold,
                -1.0,  # Close
                0.0  # No change
            )
        )
        actions[..., gripper_idx] = discretized
        data["actions"] = actions
        
        return data


@dataclasses.dataclass(frozen=True)
class DiscretizeARXGripperAbsolute(transforms.DataTransformFn):
    """Discretize absolute ARX gripper actions to {0, 1}.
    
    This transform converts continuous absolute gripper values (typically in [0,1])
    to discrete values:
    - If gripper > threshold: set to 1 (open)
    - Otherwise: set to 0 (closed)
    
    Works for both 7D and 32D action representations.
    
    Args:
        threshold: Threshold value for discretization. Default is 0.5 for absolute space.
        gripper_idx_7d: Index of gripper in 7D action space. Default is 6.
        gripper_idx_32d: Index of gripper in 32D action space (ARX data at indices 7-13). Default is 13.
    """
    
    threshold: float = 3.0
    gripper_idx_7d: int = 6
    gripper_idx_32d: int = 13
    
    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        
        actions = np.asarray(data["actions"], dtype=np.float32)
        
        # Determine action dimensionality
        if actions.shape[-1] == 7:
            gripper_idx = self.gripper_idx_7d
        elif actions.shape[-1] == 32:
            gripper_idx = self.gripper_idx_32d
        else:
            logging.warning(
                f"DiscretizeARXGripperAbsolute: Unknown action dimension {actions.shape[-1]}, skipping discretization."
            )
            return data
        
        gripper_values = actions[..., gripper_idx]
        discretized = (gripper_values > self.threshold).astype(np.float32)
        actions[..., gripper_idx] = discretized
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class DiscretizeARXGripperAbsoluteToRange(transforms.DataTransformFn):
    """Map absolute gripper logits/values to a target range {low, high} using a threshold.
    
    Typical usage at OUTPUT stage after unnormalize and ARXOutputs (7D):
    - threshold=0.5, low=0.0, high=5.0
    -> values > 0.5 become 5.0; else 0.0
    
    Can also operate on 32D (will use gripper at index 13).
    """
    threshold: float = 0.5
    low: float = 0.0
    high: float = 5.0
    gripper_idx_7d: int = 6
    gripper_idx_32d: int = 13

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 1:
            # Single step: make it (1, dim) for uniform logic
            actions = actions[None, ...]
            squeeze_back = True
        else:
            squeeze_back = False

        if actions.shape[-1] == 7:
            gi = self.gripper_idx_7d
        elif actions.shape[-1] == 32:
            gi = self.gripper_idx_32d
        else:
            raise ValueError(f"Unknown action dimension: {actions.shape[-1]}")

        mask = actions[..., gi] > self.threshold
        actions[..., gi] = np.where(mask, self.high, self.low)

        if squeeze_back:
            actions = actions[0]
        data["actions"] = actions
        return data

@dataclasses.dataclass(frozen=True)
class DiscretizeARXGripperStateToRange(transforms.DataTransformFn):
    """Map absolute gripper state values to a target range {low, high} using a threshold.
    
    This transform processes the gripper value in the observation state (not actions).
    Typical usage at INPUT stage after ARXInputs (32D state):
    - threshold: threshold value to determine open/closed
    - low: value to set if gripper <= threshold (closed)
    - high: value to set if gripper > threshold (open)
    
    The gripper is at index 13 in the 32D state (ARX data at indices 7-13).
    
    Args:
        threshold: Threshold value for discretization
        low: Value to set if gripper <= threshold (default: 0.0)
        high: Value to set if gripper > threshold (default: 1.0 for training consistency)
        gripper_idx_32d: Index of gripper in 32D state space (default: 13)
    """
    threshold: float = 3.0
    low: float = 0.0
    high: float = 1.0  # Changed to 1.0 to match action training targets
    gripper_idx_32d: int = 13  # Gripper is at index 13 in 32D space (ARX data at indices 7-13)
    
    def __call__(self, data: dict) -> dict:
        if "state" not in data:
            return data
        
        state = np.asarray(data["state"], dtype=np.float32)
        
        # Handle both 1D (single sample) and 2D (batched) states
        if state.ndim == 1:
            # Single sample: (32,)
            if state.shape[0] == 32:
                gripper_value = state[self.gripper_idx_32d]
                state[self.gripper_idx_32d] = self.high if gripper_value > self.threshold else self.low
            else:
                logging.warning(
                    f"DiscretizeARXGripperStateToRange: Expected 32D state, got {state.shape[0]}D. Skipping."
                )
        elif state.ndim == 2:
            # Batched: (batch, 32)
            if state.shape[1] == 32:
                gripper_values = state[:, self.gripper_idx_32d]
                state[:, self.gripper_idx_32d] = np.where(
                    gripper_values > self.threshold, self.high, self.low
                )
            else:
                logging.warning(
                    f"DiscretizeARXGripperStateToRange: Expected 32D state, got {state.shape[1]}D. Skipping."
                )
        else:
            logging.warning(
                f"DiscretizeARXGripperStateToRange: Unexpected state shape {state.shape}. Skipping."
            )
        
        data["state"] = state
        return data


@dataclasses.dataclass(frozen=True)
class ARXOutputs(transforms.DataTransformFn):
    """Transform model outputs back to ARX robot action format.
    
    Extracts 7D ARX actions from 32D model output (indices 7-13).
    Output format: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, delta_gripper]
    
    Args:
        freeze_gripper: If True, always set gripper action to 0 (no movement)
    """
    
    # ARX action indices in 32D space (right arm position)
    ARX_ACTION_START_IDX: int = 7
    ARX_ACTION_END_IDX: int = 14  # exclusive
    GRIPPER_IDX: int = 6  # Gripper is the 7th dimension (index 6) in 7D ARX actions
    
    freeze_gripper: bool = False  # Default: enable gripper
    
    def __call__(self, data: dict) -> dict:
        # Extract ARX actions from indices 7-13 (right arm position)
        actions_32d = np.asarray(data["actions"])  # Shape: (action_horizon, 32)
        arx_actions = actions_32d[:, self.ARX_ACTION_START_IDX:self.ARX_ACTION_END_IDX]  # Shape: (action_horizon, 7)
        
        # Freeze gripper if requested
        if self.freeze_gripper:
            arx_actions[:, self.GRIPPER_IDX] = 0.0
        
        return {"actions": arx_actions}

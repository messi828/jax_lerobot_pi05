"""SO101 (SO-ARM101) policy transforms for OpenPI.

SO101 LeRobot dataset format (produced by leader/record_with_leader.py):
- observation.state: (6,) float32, absolute joint positions in degrees.
  Motor order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
  (gripper is normalized to 0-100 range by the SO101 driver).
- action: (6,) float32, leader target joint positions (same order/units as state).
- observation.images.front: global camera (RealSense D435i color).
- observation.images.wrist: wrist camera.

The inference client must send observations with the same keys and units.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

SO101_ACTION_DIM = 6

SO101_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy."""
    return {
        "observation.state": np.random.rand(SO101_ACTION_DIM).astype(np.float32),
        "observation.images.front": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the cube",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C). LeRobot stores float32 (C, H, W) during training."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Convert SO101 observations to the model input format.

    Used for both training and inference. State/actions are kept at 6D here;
    padding to the model action dim (32 for pi0.5) is handled by
    `PadStatesAndActions` in the model transforms.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation.images.front"])
        wrist_image = _parse_image(data["observation.images.wrist"])

        inputs = {
            "state": np.asarray(data["observation.state"], dtype=np.float32),
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                # SO101 has no right wrist camera: pad with zeros.
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are only masked for pi0/pi0.5, not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Convert model outputs back to the SO101 action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # The model outputs padded (32D) actions; SO101 uses the first 6 dims.
        return {"actions": np.asarray(data["actions"][:, :SO101_ACTION_DIM])}

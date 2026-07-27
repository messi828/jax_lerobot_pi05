#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax
from openpi.training import config as _config
from openpi.policies import policy_config

# Check device
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Load config and policy
config = _config.get_config("pi05_arx")
checkpoint_dir = "./checkpoints/pi05_arx/arx_press_5k_test/2000"

print(f"\nLoading policy from: {checkpoint_dir}")
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print(f"Policy loaded. Is PyTorch: {policy._is_pytorch_model}")

# Create test observation
obs = {
    "observation.state": np.zeros(27, dtype=np.float32),
    "observation.images.front": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    "prompt": "press the button",
}

# Set some reasonable values for state
obs["observation.state"][0:6] = [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]  # ee pose
obs["observation.state"][24] = 0.5  # gripper

print("\nRunning first inference (may include JIT compilation)...")
result = policy.infer(obs)
print(f"First inference time: {result['policy_timing']['infer_ms']:.2f}ms")

print("\nRunning second inference (should be faster)...")
result = policy.infer(obs)
print(f"Second inference time: {result['policy_timing']['infer_ms']:.2f}ms")

print("\nRunning third inference...")
result = policy.infer(obs)
print(f"Third inference time: {result['policy_timing']['infer_ms']:.2f}ms")

print(f"\nSuccess!")
print(f"Actions shape: {result['actions'].shape}")
print(f"First action: {result['actions'][0]}")

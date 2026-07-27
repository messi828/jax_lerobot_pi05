#!/usr/bin/env python
"""
使用本地 pi0 微调权重（如 020000 步）进行推理示例。

权重目录应包含：
  - config.json
  - model.safetensors
  - policy_preprocessor.json（若训练时保存）
  - policy_postprocessor.json（若训练时保存）

用法：
  # 方式 1：用 lerobot-eval 在环境中跑多幕（推荐）
  lerobot-eval --policy.path=/path/to/checkpoints/020000/pretrained_model --env.type=... ...

  # 方式 2：本脚本仅加载权重并做单步推理示例
  python scripts/run_pi0_inference_so101.py /path/to/checkpoints/020000/pretrained_model
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Load pi0 (SO101) checkpoint and run inference example")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to pretrained_model dir, e.g. .../checkpoints/020000/pretrained_model",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")

    import torch
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    print(f"Loading policy from: {checkpoint_path}")
    policy = PI0FastPolicy.from_pretrained(str(checkpoint_path))
    policy.to(args.device)
    policy.eval()

    # 若训练时保存了 pre/post processor，推理时需一致使用
    preprocessor = postprocessor = None
    preprocessor_path = checkpoint_path / "policy_preprocessor.json"
    postprocessor_path = checkpoint_path / "policy_postprocessor.json"
    if preprocessor_path.exists() and postprocessor_path.exists():
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint_path), config_filename="policy_preprocessor.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint_path), config_filename="policy_postprocessor.json"
        )
        print("Pre/post processors loaded.")
    else:
        print("No policy_preprocessor/postprocessor.json in checkpoint; raw policy only.")

    print("Policy loaded. Example: single-step inference with dummy observation.")
    # 与 SO101 观测格式一致（key 与训练时一致），此处为形状示例
    dummy_obs = {
        "observation.images.front": torch.randn(1, 3, 224, 224),
        "observation.images.wrist": torch.randn(1, 3, 224, 224),
        "observation.images.overhead": torch.randn(1, 3, 224, 224),
        "observation.state": torch.randn(1, 14),
        "task": "pick up the object and place it in the target location",
    }
    if preprocessor is not None and postprocessor is not None:
        batch = preprocessor(dummy_obs)
        with torch.no_grad():
            action = policy.select_action(batch)
        action = postprocessor({"action": action})["action"]
    else:
        with torch.no_grad():
            action = policy.select_action(dummy_obs)
    print("Single-step action:", action.shape if hasattr(action, "shape") else type(action))
    print("Done.")


if __name__ == "__main__":
    main()

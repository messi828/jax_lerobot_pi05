#!/usr/bin/env python
"""
下载 gpudad/pi0fast-so101-pick-cube 模型权重

使用前请：
1. 运行 huggingface-cli login 登录
2. 在 https://huggingface.co/google/paligemma-3b-pt-224 接受模型条款
3. 如在国内网络，可设置: export HF_ENDPOINT=https://hf-mirror.com
"""
import os

def main():
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    import torch

    print("正在从 Hugging Face 下载 pi0fast-so101-pick-cube 模型...")
    policy = PI0FastPolicy.from_pretrained("gpudad/pi0fast-so101-pick-cube")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    policy.eval()
    print(f"模型已加载并移至 {device}")
    print("完成!")


if __name__ == "__main__":
    main()

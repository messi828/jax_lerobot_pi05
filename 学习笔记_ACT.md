# ACT 学习笔记：SO-ARM101 从采集到真机部署全流程

日期：2026-07-28
工作区：`/home/jaylen/桌面/jax_lerobot_pi05`
成果：ACT 策略在真机 SO-ARM101 上完成 `pick_screw_into_box` 任务

---

## 一、我做了什么（完整流程回顾）

### 1. 硬件准备与标定
- 用 udev 规则把串口和相机绑定成固定名称，插拔不乱序：
  - 从臂 `/dev/so101_follower2`（id=follower2）
  - 主臂 `/dev/so101_leader2`（id=leader2）
  - 腕部相机 `/dev/video-so101-wrist`（OpenCV）
  - 全局相机 RealSense D435i（serial 050522073017）
- 主/从臂各做一次关节标定（摆中位 → 各关节转到两端极限），标定文件存放在
  `~/.cache/huggingface/lerobot/calibration/` 下的 `follower2.json` / `leader2.json`。
- 每次采集/部署前跑遥操作自检：`bash lerobot/lerobot-main/new_arm_setup/04_teleop_test.sh`。

### 2. 遥操作采集数据
- 用自写脚本 `leader/record_with_leader.py` + 配置 `leader_record_config_v2.json`：
  主臂拖动示教，从臂跟随，同时录 front/wrist 两路 640x480@30fps 图像和 6 维关节状态。
- 采了 **50 条成功演示**，共 17968 帧，存为 LeRobot v3.0 格式：
  `dataset/jaylen/pick_lift_cube_so101_v2`
- 每条结束用 s/f/r/q 打标（成功/失败/重录/退出），本数据集 50 条全部为成功条。
- 数据结构：`data/`（parquet 逐帧数据）+ `videos/`（AV1 编码 mp4）+ `meta/`
  （info.json、stats.json 统计量、episodes 索引）。

### 3. 训练 ACT
- 脚本 `train_act_so101.sh`，核心命令 `lerobot-train --policy.type=act`。
- 关键参数：batch_size=8，steps=100000，save_freq=2000，单卡 RTX 3080。
- 归一化不用单独算：数据集 `stats.json` 的 MEAN_STD 直接打包进 checkpoint 的
  pre/postprocessor，推理时自动做（这是和 pi0.5 需要 compute_norm_stats 的区别）。
- 产物：`checkpoints/act/act_so101_pick/checkpoints/100000/pretrained_model/`
  （config.json + model.safetensors 约 200MB + 前后处理器）。
- 支持断点续训：`lerobot-train --config_path=.../last/pretrained_model/train_config.json --resume=true`。

### 4. 真机部署与评估
- 用 `lerobot-record` 挂 `--policy.path` 让策略直接控制从臂，同时把评估过程录成数据集。
- 必须与训练数据一致的项：`use_degrees=true`、相机 key 名 front/wrist、分辨率、fps=30。
- 安全项：`--robot.max_relative_target=15` 限制单步关节最大变化量（度），首跑必开。

### 5. 踩过的坑（重点复习）

| 现象 | 原因 | 解决 |
|---|---|---|
| `FileExistsError` | `--dataset.root` 目录已存在（上次运行已建目录） | 删目录或换 repo_id/root 名 |
| 机械臂只抖动不动作 | `--policy.n_action_steps=10` 覆盖了训练配置的 100，每 0.33s 重规划；且起始姿态偏离训练分布，肘关节目标被安全限幅反复截断 | 用默认 n_action_steps=100；起始姿态/场景/光照与采集时一致 |
| 日志刷 "Relative goal position clamped" | 模型目标与当前关节差超过 max_relative_target | 少量出现正常；持续出现说明姿态偏离或模型输出异常，不要靠调大限幅来掩盖 |
| 脚本找不到 | SOP 里的相对路径以 `lerobot/lerobot-main` 为基准 | 注意当前目录 |

经验总结：
- ACT 对视角/光照/物体摆放敏感，评估条件要尽量还原采集条件。
- 评估时 `single_task` 只是标注文案，ACT 没有语言理解，模型不看 prompt。
- 先小 `max_relative_target` 首跑确认安全，稳定后再放开。

---

## 二、ACT 算法原理

ACT = **Action Chunking with Transformers**，出自论文
*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ALOHA, Zhao et al., RSS 2023）。
约 52M 参数，是轻量的模仿学习（Behavior Cloning）基线。

### 1. 核心思想一：动作分块（Action Chunking）
传统 BC 每步预测 1 个动作，误差逐步累积（compounding error），且人类演示里
的停顿会让模型学到"原地不动"。ACT 一次预测**未来 k 步动作序列**（chunk_size=100，
即 30Hz 下约 3.3 秒），然后开环执行这段序列再重新推理：
- 有效决策频率降低 k 倍 → 误差累积大幅减少；
- 序列整体建模，动作平滑连贯。

这正是我踩的坑：把 `n_action_steps` 从 100 改成 10，等于每执行 10 步就重新
推理一次，chunk 之间衔接处目标跳变，表现为抖动。

### 2. 核心思想二：CVAE 建模演示的多样性
人类演示同一任务的轨迹不唯一（速度、路径都有随机性）。若用普通回归（MSE），
模型会输出多种模式的平均值，导致动作模糊。ACT 用 **条件变分自编码器（CVAE）**：
- **训练时**：一个额外的 VAE 编码器（4 层 Transformer encoder）把真实动作序列
  压缩成 32 维隐变量 z（可理解为"这条演示的风格"）；解码器以 z + 观测为条件
  重建动作序列。损失 = 动作重建 L1 损失 + KL 正则（kl_weight=10 把 z 的分布
  拉向标准正态）。
- **推理时**：直接令 z = 0（先验均值），输出确定性的"平均风格"动作。

### 3. 网络结构（对应 config.json 里的参数）

```text
观测输入
  ├─ front 相机 640x480 ──ResNet18──┐
  ├─ wrist 相机 640x480 ──ResNet18──┤  展平成视觉 token 序列
  └─ 关节状态 6 维 ──线性投影────────┤
                                     ▼
              Transformer Encoder（4 层, dim=512, 8 头）
                                     ▼
              Transformer Decoder（1 层）+ 100 个可学习查询
                                     ▼
              线性头 → 100 x 6 的动作序列（关节角度，度）
```

- 输入输出都用数据集统计量做 MEAN_STD 归一化/反归一化；
- 可选的 temporal ensembling（多次预测的重叠部分加权平均）本次未启用
  （`temporal_ensemble_coeff=null`）。

### 4. 与 pi0.5 的对比（本工作区两条技术路线）

| | ACT | pi0.5 |
|---|---|---|
| 类型 | 单任务模仿学习 | 视觉-语言-动作大模型（VLA） |
| 参数量 | ~52M | 数十亿 |
| 语言指令 | 不理解，task 仅是标注 | 理解 prompt |
| 训练 | 单卡数小时，从头训 | LoRA/全量微调预训练权重 |
| 框架 | PyTorch（lerobot） | JAX（openpi） |
| 推理 | 本机进程内直接跑 | 策略服务器 + 客户端 |
| 泛化 | 对视角/摆放敏感 | 较强 |

---

## 三、把训练好的模型分享给其他设备

`model.safetensors` 约 200MB，**超过 GitHub 单文件 100MB 硬限制**，直接
`git add` + `git push` 会被拒绝。两种方案：

### 方案 A（推荐）：Hugging Face Hub
模型托管的标准做法，lerobot 原生支持，其他设备一行代码就能加载。

```bash
# 1. 登录（一次性，需要在 huggingface.co 生成 write token）
hf auth login

# 2. 上传 pretrained_model 目录
hf upload <你的HF用户名>/act_so101_pick \
  checkpoints/act/act_so101_pick/checkpoints/100000/pretrained_model \
  --repo-type model
```

其他设备使用（无需手动下载，自动拉取缓存）：

```bash
lerobot-record ... --policy.path=<你的HF用户名>/act_so101_pick ...
```

私有模型加 `--private` 上传，使用方 `hf auth login` 即可。

### 方案 B：GitHub + Git LFS
如果坚持用 GitHub（当前仓库已有 backup 远程 `messi828/jax_lerobot_pi05`）：

```bash
# 1. 安装并启用 LFS（一次性）
sudo apt install git-lfs
git lfs install

# 2. 让 LFS 接管大文件
git lfs track "*.safetensors"
git add .gitattributes

# 3. 只提交部署必需的 pretrained_model（training_state 约 400MB 是优化器状态，不要传）
git add checkpoints/act/act_so101_pick/checkpoints/100000/pretrained_model/
git commit -m "Add ACT so101 pick checkpoint (100k steps)"
git push backup master

# 其他设备
git clone https://github.com/messi828/jax_lerobot_pi05.git   # LFS 文件自动拉取
```

注意：GitHub 免费版 LFS 只有 1GB 存储 + 1GB/月 带宽，多台设备频繁 clone 很快
用完，所以**首选方案 A**。

### 上传内容清单
部署只需要 `pretrained_model/` 目录（约 198MB）：
- `config.json` — 模型结构
- `model.safetensors` — 权重
- `policy_preprocessor*` / `policy_postprocessor*` — 归一化（已含数据集统计量）
- `train_config.json` — 训练配置（可选，供续训/复现）

`training_state/`（优化器状态，约 400MB）只用于断点续训，无需分享。

---

## 四、下一步可以做什么
- 用 `lerobot-record` 录 10-20 条评估数据，统计成功率作为 baseline。
- 对比 pi0.5 LoRA 微调在同一任务上的表现（数据已共用）。
- 尝试启用 temporal ensembling（`--policy.temporal_ensemble_coeff=0.01`）看动作是否更平滑。
- 增加物体摆放位置的多样性重新采集，检验 ACT 的泛化边界。

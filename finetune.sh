# Pi0 configs (7D actions, use Pi0 base model)
# TASK="pi0_arx"
# TASK="pi0_arx_lora"

# Pi0.5 configs (32D actions, use Pi0.5 base model)
TASK="pi05_arx"
TASK="pi05_arx_carrot"

# TASK="pi05_arx_lora"

python scripts/compute_norm_stats.py --config-name $TASK

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# NPROC_PER_NODE=4 \
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
# python scripts/train.py $TASK \
#     --exp-name pi0_arx_pick \
#     --overwrite


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py $TASK \
    --exp-name pi05_carrot \
    --overwrite
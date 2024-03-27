#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# GPUS_PER_NODE=8
# # Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
# NNODES=1
# NODE_RANK=0
# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=$1
# MASTER_PORT=6000
NNODES=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
JOBID=133

CHECKPOINT_PATH=/home/gulhane.2/Megatron-LM-MCR-DL/gpt_dataset_aws/release/mp_rank_00/model_optim_rng.pt
VOCAB_FILE=/home/gulhane.2/Megatron-LM-MCR-DL/gpt_dataset_aws/gpt2-vocab.json
MERGE_FILE=/home/gulhane.2/Megatron-LM-MCR-DL/gpt_dataset_aws/gpt2-merges.txt
DATA_PATH=/home/gulhane.2/Megatron-LM-MCR-DL/gpt_dataset_aws/my-gpt2_text_document

# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "
#  --rdzv_id $RANDOM \

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id $JOBID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:29500
"

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

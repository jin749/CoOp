#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
#bash scripts/coop/zeroshot.sh caltech101

# custom config
DATA=/hdd/hdd3/jsh/DATA
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
SUB=$3

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB}

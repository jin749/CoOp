#!/bin/bash

# custom config
DATA=/hdd/hdd2/sch/DATA
# TRAINER=CoCoOp
TRAINER=CoOp

DATASET=$1
SEED=$2

# CFG=vit_b16_c4_ep10_batch1_ctxv1
CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SHOTS=-1
# LOADEP=10
LOADEP=200
SUB=$3


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/all2all/train_base/${COMMON_DIR}
DIR=output/all2all/test_${SUB}/${COMMON_DIR}
# if [ -d "$DIR" ]; then
#     echo "Oops! The results exist at ${DIR} (so skip this job)"
# else
python train2.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
# fi
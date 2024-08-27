#!/bin/bash

#SBATCH -J coop_concept    # name of job
#SBATCH -c 8                        # number of cpus required per task
#SBATCH --gres=gpu:1                # number of gpus required
#SBATCH -D /home/jin749/Projects/coop_concept     # set working directory for batch script
#SBATCH -o /home/jin749/Projects/coop_concept/sbatch/slogs/%x_%A_%a.out    # file for batch script's standard output
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jin749@postech.ac.kr

#SBATCH --mem-per-gpu=12G           # memory required per allocated GPU
#SBATCH -t 0-10:00:00               # time limit
#SBATCH -p A5000                    # partition requested
#SBATCH -a 1-12                      # job array index values
source /home/jin749/.bashrc
conda activate coop
config=/home/jin749/Projects/coop_concept/sbatch/config.csv

echo JOB_ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} && echo
echo pwd: 
pwd && echo
echo which python: 
which python && echo

# custom config
DATA=/home/jin749/DATA
TRAINER=CoOp

DATASET=$(awk -F '[,]' -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $1}' $config)
CFG=vit_b32
CTP=end
NCTX=16
SHOTS=$(awk -F '[,]' -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $2}' $config)
CSC=True

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
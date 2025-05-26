dataset=$1
seed=$2
gpuid=$3
export CUDA_VISIBLE_DEVICES=${gpuid}
bash scripts/cocoop/all2all_full_train.sh ${dataset} ${seed}
bash scripts/cocoop/all2all_full_test.sh ${dataset} ${seed} base
bash scripts/cocoop/all2all_full_test.sh ${dataset} ${seed} new
bash scripts/cocoop/all2all_full_test.sh ${dataset} ${seed} all
dataset=imagenet
seed=$1
gpuid=$2
export CUDA_VISIBLE_DEVICES=${gpuid}
bash scripts/cocoop/all2all_full_train_imagenet.sh ${dataset} ${seed}
bash scripts/cocoop/all2all_full_test_imagenet.sh ${dataset} ${seed} base
bash scripts/cocoop/all2all_full_test_imagenet.sh ${dataset} ${seed} new
bash scripts/cocoop/all2all_full_test_imagenet.sh ${dataset} ${seed} all
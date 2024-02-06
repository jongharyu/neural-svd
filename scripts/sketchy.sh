#!/bin/bash

echo "USING GPU $CUDA_VISIBLE_DEVICES"

#export OMP_NUM_THREADS=1  # reference https://github.com/bcgsc/mavis/issues/185
export ROOT="/home/"
export SRCROOT="/home/src/neural-svd"
export PYTHONPATH="${PYTHONPATH}:${SRCROOT}"

# shellcheck disable=SC2054
args=(
    --root_dir ${ROOT}
    --exp_tag paper
    --overwrite

    --network_dims 8192,512
    --mu 16

    --num_epochs 10
    --warmup_epochs 0
    --batch_size 4096
    --optimizer sgd
    --momentum 0.9
    --base_lr 5e-3
    --use_lr_scheduler
    --clip_grad_norm

    --neigs 512
    --loss_name neuralsvd
    --neuralsvd.step 1
    --neuralsvd.sequential 0

    --sketchy_split $1
    --n_retrievals_to_save 20
    --trunc_dims -512 -448 -384 -320 -256 -192 -128 -64 -32 -16 -8 -4 -2 -1 1 2 4 8 16 32 64 128 192 256 320 384 448 512
    --ap_ver 1
)
for seed in {0..9}
do
    echo "Running Sketchy experiment (split $1) with seed $seed"
    $ROOT/.conda/envs/sketchy/bin/python $SRCROOT/examples/cdk/sketchy/main_sketchy.py "${args[@]}" --seed $seed
done

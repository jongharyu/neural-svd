#!/bin/bash

echo "USING GPU $CUDA_VISIBLE_DEVICES"

#export OMP_NUM_THREADS=1  # reference https://github.com/bcgsc/mavis/issues/185
export ROOT="/home/"
export SRCROOT="/home/src/neural-svd"
export PYTHONPATH="${PYTHONPATH}:${SRCROOT}"

# shellcheck disable=SC2054
args=(
    --optimizer rmsprop
    --use_lr_scheduler
    --ema_decay 0.995
    --batch_size $2
    --lr 1e-4
    --momentum 0.
    --num_iters 500000

    --laplacian_eps 0.01
    --eval_freq 10000
    --overwrite

    --potential_type hydrogen
    --ndim 2
    --lim 50
    --val_eps 0.1
    --neigs 36

    --apply_boundary 0
    --boundary_mode dir_box_sqrt

    --apply_exp_mask 0
    --exp_mask_init_scale 100

    --mlp_hidden_dims 128,128,128
    --hard_mul_const 1.
    --parallel 1
    --nonlinearity softplus

    --sampling_mode gaussian
    --sampling_scale 16

    --operator_scale 100
    # --operator_shift -3.0
    # --operator_inverse

    --use_fourier_feature
    # --fourier_append_raw
    --fourier_mapping_size 1024
    --fourier_scale 0.1

    --sort 0

    --neuralsvd.step 1
    --neuralsvd.sequential $3

    --neuralef.unbiased 1  # default: 0
    --neuralef.include_diag 0
    --neuralef.batchnorm_mode unbiased  # default: biased

    # --print_local_energies
    # --post_align
    --residual_weight 0.
)

for seed in {0..9}; do
    python $SRCROOT/examples/operator/pde/main_pde.py "${args[@]}" --loss_name $1 --seed $seed
done

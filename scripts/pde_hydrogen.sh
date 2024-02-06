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
    --batch_size 128
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

    --apply_exp_mask 1
    --exp_mask_init_scale 100

    --mlp_hidden_dims 128,128,128
    --hard_mul_const 1.
    --parallel 1
    --nonlinearity softplus

    --use_gaussian_sampling
    --sampling_scale 32

    --operator_scale 1000
    # --operator_shift -3.0
    # --operator_inverse

    --use_fourier_feature
    # --fourier_append_raw
    --fourier_mapping_size 256
    --fourier_scale 0.001

    --neuralsvd.step 1
    --neuralsvd.sequential 1
    --neuralsvd.separation 0
    --neuralsvd.separation_mode bn
    --neuralsvd.separation_init_scale 100
    --neuralsvd.separation_decompose_id 0.1

    --neuralef.unbiased 1
    --neuralef.include_diag 0
    --neuralef.normalize 1
    --neuralef.batchnorm_mode unbiased

    # --print_local_energies
    # --post_align
    --residual_weight 0. # 0.00001
)

for seed in {0..9}; do
    $ROOT/.conda/envs/pde/bin/python $SRCROOT/examples/operator/main_pde.py "${args[@]}" --loss_name $1 --seed $seed
done

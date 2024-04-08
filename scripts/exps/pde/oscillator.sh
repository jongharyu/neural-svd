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
    --num_iters 100000

    --laplacian_eps 0.01
    --eval_freq 100000
    --overwrite

    --potential_type harmonic_oscillator
    --ndim 2
    --lim 5
    --val_eps 0.1
    --neigs 55

    --apply_boundary 0
    --boundary_mode dir_box_sqrt

    --apply_exp_mask 1
    --exp_mask_init_scale 10

    --mlp_hidden_dims 128,128,128
    --hard_mul_const 1.
    --parallel 1
    --nonlinearity softplus

    --sampling_mode gaussian
    --sampling_scale 4

    --operator_scale 1
    --operator_shift 16.0
    # --operator_inverse

    --use_fourier_feature
    # --fourier_append_raw
    --fourier_mapping_size 256
    --fourier_scale 1

    --neuralsvd.step 1
    --neuralsvd.sequential $3
    --neuralsvd.separation 0
    --neuralsvd.separation_mode bn
    --neuralsvd.separation_init_scale 100
    --neuralsvd.separation_decompose_id 0.1

    --neuralef.unbiased 1  # default: 0
    --neuralef.include_diag 0
    --neuralef.normalize 1
    --neuralef.batchnorm_mode unbiased  # default: biased

    # --print_local_energies
    # --post_align
    --residual_weight 0.
)

for seed in {0..9}; do
    python $SRCROOT/examples/operator/pde/main_pde.py "${args[@]}" --loss_name $1 --seed $seed
done

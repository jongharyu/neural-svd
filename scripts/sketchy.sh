#!/bin/bash

# Slurm sbatch options
#SBATCH -o sketchy.sh.log-%j
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:1

# Loading the required module
module load cuda/11.6
module load anaconda/2022b

source activate frobenius

export OMP_NUM_THREADS=1  # reference https://github.com/bcgsc/mavis/issues/185
export PYTHONPATH="${PYTHONPATH}:~/working/frobenius-spectral-decomposition"


# /home/gridsan/jryu/.conda/envs/frobenius/bin/python cdk/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag test,step=1,sg=0 --mu 16 --use_amp --root_dir /home/gridsan/jryu/ --overwrite --num_epochs 10

# /home/gridsan/jryu/.conda/envs/frobenius/bin/python cdk/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag step=1,sg=0,relu,bn --mu 16 --use_amp --root_dir /home/gridsan/jryu/ --activation relu --use_bn --overwrite --num_epochs 10

# /home/gridsan/jryu/.conda/envs/frobenius/bin/python cdk/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag step=1,sg=0,relu,bn --mu 4 --use_amp --root_dir /home/gridsan/jryu/ --activation relu --use_bn --overwrite --num_epochs 10

export mu=0

# /home/gridsan/jryu/.conda/envs/frobenius/bin/python cdk/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag step=1,cntr=0,sg=0,adam1e-4,0.01_100. --mu $mu --root_dir /home/gridsan/jryu/ --frobenius.ratio_upper_bound 100. --frobenius.ratio_lower_bound 0.01 --overwrite --num_epochs 25 --use_amp --base_lr 1e-4 --frobenius.set_first_mode_const False

/home/gridsan/jryu/.conda/envs/frobenius/bin/python max_corr/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag step=1,cntr=0,sg=0,adam1e-4,0.01_100. --mu $mu --root_dir /home/gridsan/jryu/ --overwrite --num_epochs 25 --use_amp --base_lr 1e-4 --frobenius.set_first_mode_const False --network_dims 32,32 --frobenius.ratio_upper_bound 100. --frobenius.ratio_lower_bound 0.01

# /home/gridsan/jryu/.conda/envs/frobenius/bin/python cdk/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag step=1,sg=0,adam1e-4,joint,0.01_100. --mu $mu --root_dir /home/gridsan/jryu/ --frobenius.ratio_upper_bound 100. --frobenius.ratio_lower_bound 0.01 --overwrite --num_epochs 50 --use_amp --base_lr 1e-4 --frobenius.include_joint

# /home/gridsan/jryu/.conda/envs/frobenius/bin/python cdk/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag step=1,sg=0,adam1e-4,0.01_50. --mu $mu --root_dir /home/gridsan/jryu/ --frobenius.ratio_upper_bound 50. --frobenius.ratio_lower_bound 0.01 --overwrite --num_epochs 50 --use_amp --base_lr 1e-4

# /home/gridsan/jryu/.conda/envs/frobenius/bin/python cdk/sketchy/main_sketchy.py --loss_name frobenius --frobenius.step 1 --frobenius.stop_grad False --exp_tag step=1,sg=0,adam1e-4,relu,0.01_100. --mu $mu --root_dir /home/gridsan/jryu/ --frobenius.ratio_upper_bound 100. --frobenius.ratio_lower_bound 0.01 --overwrite --num_epochs 50 --use_amp --base_lr 1e-4 --activation relu


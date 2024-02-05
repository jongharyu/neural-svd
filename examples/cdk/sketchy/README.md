```bash
module load cuda/11.6
module load anaconda/2022b
export OMP_NUM_THREADS=1  # reference https://github.com/bcgsc/mavis/issues/185
export PYTHONPATH="${PYTHONPATH}:~/working/frobenius-spectral-decomposition"
python cdk/sketchy/main_classical.py --loss_name frobenius --frobenius.step 1 --frobenius.centering 1 --frobenius.stop_grad False --exp_tag step=1,cntr=1,sg=0 --mu 64 --use_amp --root_dir /home/gridsan/jryu/
```

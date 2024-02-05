export CONDA_ALWAYS_YES="true"
conda create -n pde python=3.9
conda activate pde
conda install -y numpy scikit-learn matplotlib tqdm
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
#pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu  # for pytorch 2.0 (cpu)
conda install -c conda-forge tensorboardx
conda install -c conda-forge notebook
conda install -y seaborn
pip install classy_vision
pip install torch-ema
pip install tensorflow==2.10.0
pip install termplotlib
pip install ConfigArgParse
pip install jsonpickle
# Disable yes to all
unset CONDA_ALWAYS_YES

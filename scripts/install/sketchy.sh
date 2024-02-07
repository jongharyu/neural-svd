export CONDA_ALWAYS_YES="true"
conda create -n sketchy python=3.7
conda activate sketchy
conda install -y numpy scikit-learn matplotlib tqdm
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # for pytorch 2.0
conda install -c faiss-gpu=1.7.3  # for Sketchy retrieval task
conda install -c conda-forge tensorboardx
conda install -c conda-forge notebook
conda install pyg -c pyg  # for graph
conda install -c conda-forge ogb  # for graph
conda install -c conda-forge timm  # for graph
conda install -y seaborn
pip install classy_vision
pip install torch-ema
pip install tensorflow==2.10.0
pip install termplotlib
pip install ConfigArgParse
pip install jsonpickle
# Disable yes to all
unset CONDA_ALWAYS_YES

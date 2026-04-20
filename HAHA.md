




## Create Conda Env 
- `conda create -n your_env_name python=3.10.13`

## Torch
- `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

## Requirements: vim_requirements.txt
  - `pip install -r vim/vim_requirements.txt`
  - `pip install -r vim_src/vim_requirements.txt`

## cousal-conv1d
  - `pip install --no-build-isolation -e ./causal-conv1d`

## mamba 1p1p1
  - `pip install -e mamba-1p1p1 --no-build-isolation`

## DET
  - `cd det`
  - `pip3 install -e . --no-build-isolation`
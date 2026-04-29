




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

## Conda List
# packages in environment at /home/marh/miniconda3/envs/vimdet:
#
# Name                     Version             Build            Channel
_libgcc_mutex              0.1                 main
_openmp_mutex              5.1                 1_gnu
absl-py                    2.4.0               pypi_0           pypi
addict                     2.4.0               pypi_0           pypi
aiohttp                    3.9.1               pypi_0           pypi
aiosignal                  1.3.1               pypi_0           pypi
alembic                    1.13.0              pypi_0           pypi
antlr4-python3-runtime     4.9.3               pypi_0           pypi
async-timeout              4.0.3               pypi_0           pypi
attrs                      23.1.0              pypi_0           pypi
black                      26.3.1              pypi_0           pypi
blinker                    1.7.0               pypi_0           pypi
bzip2                      1.0.8               h5eee18b_6
ca-certificates            2025.12.2           h06a4308_0
causal-conv1d              1.4.0               pypi_0           pypi
certifi                    2023.11.17          pypi_0           pypi
charset-normalizer         3.3.2               pypi_0           pypi
click                      8.1.7               pypi_0           pypi
cloudpickle                3.0.0               pypi_0           pypi
contourpy                  1.2.0               pypi_0           pypi
cycler                     0.12.1              pypi_0           pypi
databricks-cli             0.18.0              pypi_0           pypi
datasets                   2.15.0              pypi_0           pypi
detectron2                 0.6                 pypi_0           pypi
dill                       0.3.7               pypi_0           pypi
docker                     6.1.3               pypi_0           pypi
einops                     0.7.0               pypi_0           pypi
entrypoints                0.4                 pypi_0           pypi
filelock                   3.13.1              pypi_0           pypi
flask                      3.0.0               pypi_0           pypi
fonttools                  4.46.0              pypi_0           pypi
frozenlist                 1.4.0               pypi_0           pypi
fsspec                     2023.10.0           pypi_0           pypi
fvcore                     0.1.5.post20221221  pypi_0           pypi
gitdb                      4.0.11              pypi_0           pypi
gitpython                  3.1.40              pypi_0           pypi
greenlet                   3.0.2               pypi_0           pypi
grpcio                     1.78.0              pypi_0           pypi
gunicorn                   21.2.0              pypi_0           pypi
huggingface-hub            0.19.4              pypi_0           pypi
hydra-core                 1.3.2               pypi_0           pypi
idna                       3.6                 pypi_0           pypi
importlib-metadata         7.0.0               pypi_0           pypi
iopath                     0.1.9               pypi_0           pypi
itsdangerous               2.1.2               pypi_0           pypi
jinja2                     3.1.2               pypi_0           pypi
joblib                     1.3.2               pypi_0           pypi
kiwisolver                 1.4.5               pypi_0           pypi
ld_impl_linux-64           2.44                h153f514_2
libffi                     3.4.4               h6a678d5_1
libgcc                     15.2.0              h69a1729_7
libgcc-ng                  15.2.0              h166f726_7
libgomp                    15.2.0              h4751f2c_7
libstdcxx                  15.2.0              h39759b7_7
libstdcxx-ng               15.2.0              hc03a8fd_7
libuuid                    1.41.5              h5eee18b_0
libxcb                     1.17.0              h9b100fa_0
libzlib                    1.3.1               hb25bd0a_0
mako                       1.3.0               pypi_0           pypi
mamba-ssm                  1.1.1               pypi_0           pypi
markdown                   3.5.1               pypi_0           pypi
markupsafe                 2.1.3               pypi_0           pypi
matplotlib                 3.8.2               pypi_0           pypi
mlflow                     2.9.1               pypi_0           pypi
mmcv                       1.3.8               pypi_0           pypi
mmsegmentation             0.14.1              pypi_0           pypi
mpmath                     1.3.0               pypi_0           pypi
multidict                  6.0.4               pypi_0           pypi
multiprocess               0.70.15             pypi_0           pypi
mypy-extensions            1.1.0               pypi_0           pypi
ncurses                    6.5                 h7934f7d_0
networkx                   3.2.1               pypi_0           pypi
ninja                      1.11.1.1            pypi_0           pypi
numpy                      1.26.2              pypi_0           pypi
oauthlib                   3.2.2               pypi_0           pypi
omegaconf                  2.3.0               pypi_0           pypi
opencv-python              4.8.1.78            pypi_0           pypi
openssl                    3.5.5               h1b28b03_0
packaging                  23.2                pypi_0           pypi
pandas                     2.1.3               pypi_0           pypi
pathspec                   1.0.4               pypi_0           pypi
pillow                     10.1.0              pypi_0           pypi
pip                        26.0.1              pyhc872135_0
platformdirs               4.1.0               pypi_0           pypi
portalocker                3.2.0               pypi_0           pypi
prettytable                3.9.0               pypi_0           pypi
protobuf                   4.25.1              pypi_0           pypi
pthread-stubs              0.3                 h0ce48e5_1
pyarrow                    14.0.1              pypi_0           pypi
pyarrow-hotfix             0.6                 pypi_0           pypi
pycocotools                2.0.11              pypi_0           pypi
pyjwt                      2.8.0               pypi_0           pypi
pyparsing                  3.1.1               pypi_0           pypi
python                     3.10.13             h955ad1f_0
python-dateutil            2.8.2               pypi_0           pypi
python-hostlist            1.23.0              pypi_0           pypi
pytokens                   0.4.1               pypi_0           pypi
pytz                       2023.3.post1        pypi_0           pypi
pyyaml                     6.0.1               pypi_0           pypi
querystring-parser         1.2.4               pypi_0           pypi
readline                   8.3                 hc2a1206_0
regex                      2023.10.3           pypi_0           pypi
requests                   2.31.0              pypi_0           pypi
safetensors                0.4.1               pypi_0           pypi
scikit-learn               1.3.2               pypi_0           pypi
scipy                      1.11.4              pypi_0           pypi
setuptools                 80.10.2             py310h06a4308_0
six                        1.16.0              pypi_0           pypi
smmap                      5.0.1               pypi_0           pypi
sqlalchemy                 2.0.23              pypi_0           pypi
sqlite                     3.51.1              h3e8d24a_1
sqlparse                   0.4.4               pypi_0           pypi
sympy                      1.12                pypi_0           pypi
tabulate                   0.9.0               pypi_0           pypi
tensorboard                2.20.0              pypi_0           pypi
tensorboard-data-server    0.7.2               pypi_0           pypi
termcolor                  3.3.0               pypi_0           pypi
threadpoolctl              3.2.0               pypi_0           pypi
timm                       0.4.12              pypi_0           pypi
tk                         8.6.15              h54e0aa7_0
tokenizers                 0.15.0              pypi_0           pypi
tomli                      2.0.1               pypi_0           pypi
torch                      2.1.1+cu118         pypi_0           pypi
torchaudio                 2.1.1+cu118         pypi_0           pypi
torchvision                0.16.1+cu118        pypi_0           pypi
tqdm                       4.66.1              pypi_0           pypi
transformers               4.35.2              pypi_0           pypi
triton                     2.1.0               pypi_0           pypi
typing-extensions          4.15.0              pypi_0           pypi
tzdata                     2023.3              pypi_0           pypi
urllib3                    2.1.0               pypi_0           pypi
wcwidth                    0.2.12              pypi_0           pypi
websocket-client           1.7.0               pypi_0           pypi
werkzeug                   3.0.1               pypi_0           pypi
wheel                      0.46.3              py310h06a4308_0
xorg-libx11                1.8.12              h9b100fa_1
xorg-libxau                1.0.12              h9b100fa_0
xorg-libxdmcp              1.1.5               h9b100fa_0
xorg-xorgproto             2024.1              h5eee18b_1
xxhash                     3.4.1               pypi_0           pypi
xz                         5.8.2               h448239c_0
yacs                       0.1.8               pypi_0           pypi
yapf                       0.40.2              pypi_0           pypi
yarl                       1.9.4               pypi_0           pypi
zipp                       3.17.0              pypi_0           pypi
zlib                       1.3.1               hb25bd0a_0


Driver Version: 590.48.01      CUDA Version: 13.1 
torch cuda: 11.8
Python 3.10.13
pytorch 2.1.1+cu118
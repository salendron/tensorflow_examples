python3 -m venv env
source env/bin/activate
pip install -U pip
source ~/miniforge3/bin/activate
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
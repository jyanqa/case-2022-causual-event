#!/bin/bash
cd "$(dirname "$0")"
pip install --upgrade pip
pip install --upgrade packaging
pip install datasets
pip install seqeval
python evaluate.py $1 $2
# PYENV_VERSION=anaconda3-2.5.0 python evaluate.py $1 $2 || exit 1
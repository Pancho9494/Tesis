#!/bin/bash
apt-get update -y && apt-get upgrade -y
apt-get install -y libx11-6 libgl1
yes | pip install -U pip
yes | pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.4.0+cu101.html
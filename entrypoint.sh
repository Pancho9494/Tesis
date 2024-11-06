#!/bin/bash
apt-get update -y && apt-get upgrade -y
apt-get install libx11-6 libgl1 g++ -y
yes | pip install -U pip
yes | pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
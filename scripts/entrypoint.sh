#!/bin/bash
apt-get update -y && apt-get upgrade -y
apt-get install -y libx11-6 libgl1 mesa-utils libgl1-mesa-glx libglib2.0-0 wget
yes | pip install -U pip
yes | pip install -r requirements.txt

exec "$@"
#!/bin/bash

declare -A URLS=(
    ["PREDATOR"]="https://share.phys.ethz.ch/~gsg/Predator/weights.zip"
)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WEIGHTS_PATH=$SCRIPT_DIR/../customModels/weights

for model in "${!URLS[@]}"
do
    url="${URLS[$model]}"
    echo $model "->" $url
    mkdir -p $WEIGHTS_PATH/$model/
    wget $url --directory-prefix=$WEIGHTS_PATH/$model/ --continue -O $WEIGHTS_PATH/$model.zip
    unzip -qq $WEIGHTS_PATH/$model.zip -d $WEIGHTS_PATH/$model

    if [ -e  "$WEIGHTS_PATH/$model/weights" ]; then
        mv $WEIGHTS_PATH/$model/weights/* $WEIGHTS_PATH/$model/
        rm -rf $WEIGHTS_PATH/$model/weights/
    fi
    
    rm $WEIGHTS_PATH/$model.zip
done

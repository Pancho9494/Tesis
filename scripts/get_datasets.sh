#!/bin/bash

declare -A URLS=(
    ["3DLoMatch"]="https://share.phys.ethz.ch/~gsg/Predator/data.zip"
    ["WHU-TLS"]
)

for i in "${!URLS[@]}"
do
    echo "Downloading $i from ${URLS[$i]}"
    mkdir ../data/$i
    wget "${URLS[$i]}" --directory-prefix=../data/ --continue -O ../data/$i/$i.zip
    unzip -qq ../data/$i/$i.zip -d ../data/$i
    rm -f ../data/$i/$i.zip
    echo "" 
done

echo "Please visit https://github.com/WHU-USI3DV/WHU-TLS for the WHU-TLS dataset"
#!/bin/bash

# Compile cpp subsampling
cd subsampling
rm -rf ./build ./grid*.so
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd neighbors
rm -rf ./build ./radius_neighbors*.so
python3 setup.py build_ext --inplace
cd ..
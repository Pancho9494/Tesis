#!/bin/bash

# Compile cpp subsampling
cd subsampling
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd neighbors
python3 setup.py build_ext --inplace
cd ..
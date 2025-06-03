#!/bin/bash

# Compile cpp subsampling
cd subsampling
rm -rf ./build ./grid*.so
../../../../.venv/bin/python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd neighbors
rm -rf ./build ./radius_neighbors*.so
../../../../.venv/bin/python3 setup.py build_ext --inplace
cd ..

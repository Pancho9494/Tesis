# Tesis
Working on Low Overlap Point Cloud Registration (LOPCR)

## Datasets

Currently we're working with:

* [WHU-TLS](https://github.com/WHU-USI3DV/WHU-TLS)

* [3dLoMatch](https://3dmatch.cs.princeton.edu/)

The script `get_datasets.sh` downloads the files to the `data` folder, except for those stored in google drive, it was
too much of a pain to setup, so you'll have to manually download it.

## Models

To run the models you must first install the requirements, for that you can run the `install_dependencies` script, this
will install a local copy of `Python3.8.5` inside the `.localPython` directory, then it will create a `.tesis_venv`
directory for the virtual environment, based on the local python version, and it will install all the dependencies

We have implementations of:

* [PREDATOR](https://github.com/prs-eth/OverlapPredator)
# Tesis
Research code for experiments on **Low-Overlap Point Cloud Registration**.

This repository contains training and evaluation code used during my master's thesis work, with experiments around:

- **PREDATOR**
- **IAE**
- additional configuration files for **LIM** variants

The codebase also includes small sandbox experiments such as a synthetic **Stanford Bunny-style registration** setup used for controlled tests and debugging.

---

## Repository structure

```text
.
├── containers/          # Docker and Apptainer definitions
├── scripts/             # Helper scripts for training / running experiments
├── src/
│   ├── config/          # YAML experiment configurations
│   │   ├── IAE/
│   │   ├── PREDATOR/
│   │   └── LIM/
│   ├── LIM/             # Main library / implementation
│   ├── main.py          # Main training entrypoint
│   └── test_bunny.py    # Small synthetic bunny experiment
├── test/                # Tests
├── pyproject.toml
└── README.md
```

---

## Clone the repository

This repository uses git submodules, so clone it with:
```bash
git clone --recurse-submodules https://github.com/Pancho9494/Tesis.git
cd Tesis
```

If you already cloned it without submodules:
```bash
git submodule update --init --recursive
```

---

## Environment setup
The project is currently defined through `pyproject.toml` and requires Python 3.11.9.

### Option 1: using `uv` (recommended)
```bash
uv python install 3.11.9
uv venv --python 3.11.9
source .venv/bin/activate
uv sync
```

### Option 2: manual virtual environment 
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

>[!info]
Some experiments depend on GPU-enabled PyTorch and external system libraries. On HPC / cluster environments, the container-based workflow in `containers/` and `scripts/` may be more convenient.

---

## Available experiment configurations

The repository contains multiple YAML configuration files under src/config/, including:

* src/config/IAE/IAE_recreation.yaml
* src/config/PREDATOR/PREDATOR_recreation.yaml
* src/config/PREDATOR/PREDATOR_library.yaml
* several LIM configuration variants under src/config/LIM/

These files are the main way to select model, dataset, and training settings.

---

## Running experiments
The main training entry point is:
```bash
python src/main.py <path-to-config.yaml>
```

Example: run a PREDATOR experiment
```bash
uv run python src/main.py src/config/PREDATOR/PREDATOR_recreation.yaml
```

Example: run an IAE experiment
```bash
uv run python src/main.py src/config/IAE/IAE_recreation.yaml 
```

---
## Bunny toy example
For quick debugging and controlled experiments, the repository includes src/test_bunny.py.

This script contains:

* registration-error sanity checks
* synthetic bunny pair generation
* a small bunny-based training path

Example:
```bash
uv run python src/test_bunny.py
```

If your environment is already containerized, you may also find scripts/run_bunny.sh useful as a starting point.

---
## Containerized workflows
The repository includes both:

* `containers/docker/`
* `containers/apptainer/`

As well as helper scripts in `scripts/` for cluster / container execution.

Examples include:

* `scripts/train_IAE_recreation.sh`
* `scripts/train_PREDATOR_recreation.sh`
* `scripts/run_bunny.sh`

These scripts are useful references for reproducing experiments on GPU machines or HPC environments, although some paths are environment-specific and may need to be adapted locally.

---

## Datasets
Different experiments in this repository use different datasets depending on the selected configuration and script.

From the current codebase, the main experiment paths reference:

* 3DLoMatch / ThreeDLoMatch
* ScanNet
* a synthetic Bunny registration setup for controlled testing

Please check the corresponding YAML config and dataset class before running an experiment, since dataset paths are expected to be configured there.

---
## Notes on reproducibility
This repository was developed incrementally during thesis work, so some scripts are more polished than others and a few paths are tailored to my local or HPC environment.

To reproduce an experiment, the safest workflow is:

1. inspect the YAML file under `src/config/`
2. verify dataset paths
3. verify GPU / container requirements
4. launch through `src/main.py` or the corresponding helper script

---
## What to read first
If you are new to the repository, I recommend starting in this order:

1. src/main.py
2. src/config/
3. scripts/
4. src/test_bunny.py

This should give you a quick overview of how experiments are configured and launched.

---
## Acknowledgements
This repository includes or depends on several external submodules and research implementations, including projects such as:

* [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
* [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
* [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)

#!/bin/bash
uv run aim up &
uv run python main.py configs/train/indoor.yaml
kill $(jobs -p)

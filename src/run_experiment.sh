#!/usr/bin/env bash
# Convenience script to run the full experiment pipeline.
set -euo pipefail

python -m src.experiments "$@"

#!/bin/bash
# Sequential ablation runs. pipefail matters: without it the exit status of
# `python ... | grep` is grep's, so a training crash looks like success.
set -euo pipefail
cd "$(dirname "$0")"
run() {
  echo "=== $* ==="
  "$@"
}
for name in "$@"; do
  if [[ $name == eval:* ]]; then
    run .venv/bin/python scripts/evaluate.py --config "configs/${name#eval:}.yaml"
  else
    run .venv/bin/python scripts/train.py --config "configs/$name.yaml"
  fi
done
echo "=== ALL DONE ==="

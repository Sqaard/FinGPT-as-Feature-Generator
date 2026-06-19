#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R6C_ROOT="$REPO_ROOT/artifacts/r6c_stage0_1_text_baseline_20260530/rl_stage0_1_r6c_project"
EXPERIMENT_ROOT="$R6C_ROOT/artifacts/text_improvement_plan"
HELPER="$REPO_ROOT/scripts/14_prepare_and_collect_priority0.py"
TRAIN_MODULE="src.ppo.stage0_1_train"
R6C_VARIANT="R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"

PYTHON_BIN="${PYTHON_BIN:-/home/tianyi/miniconda3/envs/homework/bin/python}"
FOLDS="${FOLDS:-fold_2018 fold_2019 fold_2020 fold_2021}"
SEEDS="${SEEDS:-42 123 2026}"
SCREEN_FOLDS="${SCREEN_FOLDS:-fold_2020 fold_2021}"
MAX_JOBS="${MAX_JOBS:-1}"
FORCE="${FORCE:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export PYTHONUNBUFFERED=1

usage() {
  sed -n '1,180p' "$0" | sed -n '/^usage()/,/^}/p'
  printf '\nCommands:\n'
  printf '  check              Validate the Python environment and required files.\n'
  printf '  prepare-screen     Create Priority 0 configs/panels for folds 2020-2021.\n'
  printf '  prepare-full       Create Priority 0 configs/panels for all four folds.\n'
  printf '  smoke              Run one short baseline and one forward-earnings job.\n'
  printf '  run-screen         Run the two-fold, three-seed Priority 0 screen.\n'
  printf '  run-full           Run all prepared four-fold Priority 0 jobs.\n'
  printf '  collect            Build paired run and summary CSV files.\n'
  printf '  priority0-screen   check, prepare-screen, run-screen, collect.\n'
  printf '  priority0-full     check, prepare-full, run-full, collect.\n'
  printf '  status             Show completed versus prepared jobs.\n'
  printf '  next               Print commands and gates for Priorities 1-5.\n'
}

check() {
  test -x "$PYTHON_BIN"
  test -f "$HELPER"
  test -f "$R6C_ROOT/configs/stage0_1_r6c_deepseek_v2_text.yaml"
  test -f "$R6C_ROOT/artifacts/stage0_1/features/stage0_1_weight_features_raw_WITH_DEEPSEEK_V2_TEXT10.csv"
  "$PYTHON_BIN" - <<'PY'
import pandas
import yaml
import torch
import stable_baselines3
import gymnasium
print("Python dependencies are available.")
PY
}

prepare() {
  local folds="$1"
  read -r -a fold_args <<< "$folds"
  read -r -a seed_args <<< "$SEEDS"
  "$PYTHON_BIN" "$HELPER" prepare \
    --folds "${fold_args[@]}" \
    --seeds "${seed_args[@]}"
}

run_job() {
  local config="$1"
  local fold="$2"
  local smoke="${3:-0}"
  local force_arg=()
  local smoke_arg=()
  if [[ "$FORCE" == "1" ]]; then
    force_arg=(--force)
  fi
  if [[ "$smoke" == "1" ]]; then
    smoke_arg=(--smoke-test)
  fi
  (
    cd "$R6C_ROOT"
    "$PYTHON_BIN" -m "$TRAIN_MODULE" \
      --config "$config" \
      --variants "$R6C_VARIANT" \
      --folds "$fold" \
      "${smoke_arg[@]}" \
      "${force_arg[@]}"
  )
}

run_manifest() {
  local allowed_folds="$1"
  test -f "$EXPERIMENT_ROOT/priority0_manifest.csv"
  while IFS=$'\t' read -r config fold; do
    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
      wait -n
    done
    run_job "$config" "$fold" 0 &
  done < <("$PYTHON_BIN" - "$EXPERIMENT_ROOT/priority0_manifest.csv" "$allowed_folds" "$EXPERIMENT_ROOT" "$R6C_VARIANT" <<'PY'
import csv
import pathlib
import sys

manifest, allowed_raw, experiment_root, model = sys.argv[1:]
allowed = set(allowed_raw.split())
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        summary = (
            pathlib.Path(experiment_root)
            / "runs"
            / row["run_name"]
            / model
            / row["fold"]
            / "validation_summary.csv"
        )
        if row["fold"] in allowed and not summary.exists():
            print(row["config"] + "\t" + row["fold"])
PY
  )
  wait
}

smoke() {
  prepare "$SCREEN_FOLDS"
  local manifest="$EXPERIMENT_ROOT/priority0_manifest.csv"
  local baseline_config
  local text_config
  baseline_config="$("$PYTHON_BIN" - "$manifest" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
print(next(r["config"] for r in rows if r["variant"] == "r6c_base" and r["seed"] == "42" and r["fold"] == "fold_2020"))
PY
)"
  text_config="$("$PYTHON_BIN" - "$manifest" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
print(next(r["config"] for r in rows if r["variant"] == "r6c_forward_earnings_raw" and r["seed"] == "42" and r["fold"] == "fold_2020"))
PY
)"
  run_job "$baseline_config" "fold_2020" 1
  run_job "$text_config" "fold_2020" 1
}

collect() {
  "$PYTHON_BIN" "$HELPER" collect
}

status() {
  test -f "$EXPERIMENT_ROOT/priority0_manifest.csv"
  "$PYTHON_BIN" - "$EXPERIMENT_ROOT/priority0_manifest.csv" "$EXPERIMENT_ROOT" "$R6C_VARIANT" <<'PY'
import csv
import pathlib
import sys

manifest, root, model = sys.argv[1:]
rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
done = 0
for row in rows:
    summary = pathlib.Path(root) / "runs" / row["run_name"] / model / row["fold"] / "validation_summary.csv"
    done += summary.exists()
print(f"completed={done} prepared={len(rows)} remaining={len(rows) - done}")
PY
}

next_steps() {
  cat <<'EOF'
Priority 1:
  Implement event-aware earnings features, then repeat:
    FOLDS="fold_2020 fold_2021" SEEDS="42 123 2026" \
      bash scripts/run_text_improvement_plan.sh priority0-screen

Priority 2:
  Implement the zero-initialized residual adapter and register it as a policy
  variant. Compare it against r6c_forward_earnings_raw with the same harness.

Priority 3:
  Add separate market-text and stock-text encoders to the routed root-split
  policy. Promote only if it beats the residual adapter.

Priority 4:
  Add bounded text modifiers one at a time to risk_stress, sell score, and buy
  score. Evaluate each target metric using the paired fold-seed harness.

Priority 5:
  After one architecture passes the gates, run the predeclared PPO/reward grid.
  Lock the winner before any frozen 2022-2023 OOS evaluation.

The repository does not yet contain the Priority 1-5 model implementations.
The launcher intentionally stops at the executable Priority 0 validation gate.
EOF
}

command="${1:-help}"
case "$command" in
  check) check ;;
  prepare-screen) prepare "$SCREEN_FOLDS" ;;
  prepare-full) prepare "$FOLDS" ;;
  smoke) check; smoke ;;
  run-screen) run_manifest "$SCREEN_FOLDS" ;;
  run-full) run_manifest "$FOLDS" ;;
  collect) collect ;;
  priority0-screen) check; prepare "$SCREEN_FOLDS"; run_manifest "$SCREEN_FOLDS"; collect ;;
  priority0-full) check; prepare "$FOLDS"; run_manifest "$FOLDS"; collect ;;
  status) status ;;
  next) next_steps ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac

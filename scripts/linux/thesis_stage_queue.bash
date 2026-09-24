#!/usr/bin/env bash
# Queue and time the three thesis suites on the workstation's single GPU.

run_queued_thesis_stage() {
  local stage_name=$1
  shift
  local queue_lock="results/.locks/thesis_stages.lock"
  local timing_log="training_logs/thesis_stages/stage_timings.jsonl"
  local lock_fd queued_utc started_utc finished_utc
  local started_epoch finished_epoch status dry_run=false

  for argument in "$@"; do
    [[ "$argument" != "--dry-run" ]] || dry_run=true
  done

  mkdir -p "$(dirname "$queue_lock")"
  exec {lock_fd}>"$queue_lock"
  queued_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[QUEUE] stage=$stage_name waiting_utc=$queued_utc"
  flock "$lock_fd"
  started_epoch=$(date +%s)
  started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[QUEUE] stage=$stage_name started_utc=$started_utc"

  set +e
  "$@"
  status=$?
  set -e

  finished_epoch=$(date +%s)
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[QUEUE] stage=$stage_name finished_utc=$finished_utc"
  echo "[QUEUE] stage=$stage_name elapsed_seconds=$((finished_epoch - started_epoch)) exit_status=$status"
  if [[ "$dry_run" == false ]]; then
    mkdir -p "$(dirname "$timing_log")"
    printf '{"stage":"%s","queued_utc":"%s","started_utc":"%s","finished_utc":"%s","elapsed_seconds":%d,"exit_status":%d}\n' \
      "$stage_name" "$queued_utc" "$started_utc" "$finished_utc" \
      "$((finished_epoch - started_epoch))" "$status" >>"$timing_log"
    echo "[QUEUE] timing_log=$timing_log"
  fi
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  return "$status"
}

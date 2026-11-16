#!/bin/bash
# Tunnel-comm training: {PPO-Lag, vanilla PPO} x seeds, 3,000 fresh episodes each.
# Usage: analysis/tunnel_train.sh PORT "ALGO:SEED ALGO:SEED ..."      (ALGO = lag | ppo)
#   e.g. two CARLA servers, one job list each:
#     analysis/tunnel_train.sh 2000 "lag:42 lag:43 lag:44"
#     analysis/tunnel_train.sh 2010 "ppo:42 ppo:43 ppo:44"
# Each run writes only under outputs/tunnel/train_<algo>_s<seed>/ (log, checkpoint, CSVs).
# A run folder that already holds a finished train.log is skipped, so the script can be rerun.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$REPO/.venv/bin/python}   # project venv (analysis/setup_venv.sh); PYTHON=... overrides
[ -x "$PY" ] || { echo "no $PY: run analysis/setup_venv.sh first" >&2; exit 1; }
PORT=${1:?port}
JOBS=${2:?job list}
EPISODES=${EPISODES:-3000}
P_OUT=${P_OUT:-0.9}
P_TUNNEL=${P_TUNNEL:-0.1:0.9}
BURST=${BURST:-0}
NOISE_STD=${NOISE_STD:-0.1}
NOISE_TAU=${NOISE_TAU:-1.0}
BRAKE_PROB=${BRAKE_PROB:-0.5}
cd "$REPO"
for JOB in $JOBS; do
  ALGO=${JOB%%:*}; SEED=${JOB##*:}
  OUT=$REPO/outputs/tunnel/train_${ALGO}_s$SEED
  if [ -f "$OUT/train.log" ] && [ "$(grep -cE 'Episode [0-9]+ ended' "$OUT/train.log")" -ge "$EPISODES" ]; then
    echo "$JOB already finished, skipping"; continue
  fi
  mkdir -p "$OUT"
  EXTRA=""; [ "$ALGO" = ppo ] && EXTRA="--no-lagrangian"
  { echo "git: $(git describe --always --dirty)"; echo "started: $(date '+%F %T')";
    git diff -- "*.py" "*.json" > "$OUT/code.diff"
    echo "algo=$ALGO seed=$SEED episodes=$EPISODES p_success=$P_OUT p_tunnel=$P_TUNNEL burst_ticks=$BURST noise_std=$NOISE_STD noise_tau=$NOISE_TAU leader_brake_prob=$BRAKE_PROB port=$PORT"; } > "$OUT/code_version.txt"
  "$PY" episode_ppo_v2.py --controller ppo --ppo-fresh --no-rendering $EXTRA \
    --seed "$SEED" --train-episodes "$EPISODES" --p-success "$P_OUT" --p-tunnel "$P_TUNNEL" \
    --tunnel-burst-ticks "$BURST" --p-tunnel-noise-std "$NOISE_STD" --p-tunnel-noise-tau-sec "$NOISE_TAU" --leader-brake-prob "$BRAKE_PROB" --port "$PORT" --tm-port $((PORT + 6000)) \
    --ppo-checkpoint "$OUT/ppo_lag.pt" --output-dir "$OUT" --max-connect-wait-sec 300 \
    > "$OUT/train.log" 2>&1
  echo "$JOB exit=$? episodes=$(grep -cE 'Episode [0-9]+ ended' "$OUT/train.log") errors=$(grep -ciE 'traceback' "$OUT/train.log") $(date '+%F %T')"
done
echo "tunnel_train port $PORT done $(date '+%F %T')"

#!/bin/bash
# Tunnel-comm evaluation: PID and every trained checkpoint, same seed (same leader speeds and
# packet draws), over a sweep of in-tunnel success probabilities.
# Usage: analysis/tunnel_eval.sh [PORT]
# Reads outputs/tunnel/train_*/ppo_lag.pt, writes outputs/tunnel/eval/<controller>_t<p>/.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$REPO/.venv/bin/python}   # project venv (analysis/setup_venv.sh); PYTHON=... overrides
[ -x "$PY" ] || { echo "no $PY: run analysis/setup_venv.sh first" >&2; exit 1; }
PORT=${1:-2000}
N=${N:-100}
P_OUT=${P_OUT:-0.9}
SWEEP=${SWEEP:-"0.9 0.7 0.5 0.3 0.1"}
BURST=${BURST:-0}
NOISE_STD=${NOISE_STD:-0.1}
NOISE_TAU=${NOISE_TAU:-1.0}
BRAKE_PROB=${BRAKE_PROB:-0.5}
EVAL_SEED=${EVAL_SEED:-1000}
CKPTS=${CKPTS:-}      # optional: space-separated run names (e.g. "lag_s42 ppo_s42"); empty = every checkpoint
RUN_PID=${RUN_PID:-1} # 0 skips the PID cells
E=$REPO/outputs/tunnel/eval; mkdir -p "$E"
cd "$REPO"
COMMON="--no-rendering --train-episodes $N --seed $EVAL_SEED --p-success $P_OUT --tunnel-burst-ticks $BURST --p-tunnel-noise-std $NOISE_STD --p-tunnel-noise-tau-sec $NOISE_TAU --leader-brake-prob $BRAKE_PROB --port $PORT --tm-port $((PORT + 6000)) --max-connect-wait-sec 300"
for P in $SWEEP; do
  if [ "$RUN_PID" = 1 ]; then
    OUT=$E/pid_t$P; rm -rf "$OUT"
    timeout 1800 "$PY" episode_ppo_v2.py --controller pid $COMMON --p-tunnel "$P" --output-dir "$OUT" > "$OUT.log" 2>&1
    echo "pid t=$P exit=$? episodes=$(grep -cE 'Episode [0-9]+ ended' "$OUT.log") $(date '+%T')"
  fi
  for CK in "$REPO"/outputs/tunnel/train_*/ppo_lag.pt; do
    NAME=$(basename "$(dirname "$CK")"); NAME=${NAME#train_}
    [ -n "$CKPTS" ] && ! echo " $CKPTS " | grep -q " $NAME " && continue
    OUT=$E/${NAME}_t$P; rm -rf "$OUT"
    timeout 1800 "$PY" episode_ppo_v2.py --controller ppo --ppo-eval --ppo-checkpoint "$CK" $COMMON --p-tunnel "$P" --output-dir "$OUT" > "$OUT.log" 2>&1
    echo "$NAME t=$P exit=$? episodes=$(grep -cE 'Episode [0-9]+ ended' "$OUT.log") $(date '+%T')"
  done
done
echo "tunnel_eval done $(date '+%T')"

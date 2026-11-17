#!/bin/bash
# PID gain sweep for the tunnel scenario. No code change: each gain set gets its own copy of the
# scenario file (follower_control.kp/ki/kd replaced, h and d0 untouched) and is passed by --scenario.
# Usage: analysis/pid_retune_sweep.sh "PORT:kp,ki,kd[@p_tunnel] ..." (one queue per port, run in parallel)
#   default condition: no loss (p = 1.0 in and out, noise 0), leader hard brake prob 0.5, eval seed 1000.
#   with @p: in-tunnel p = that value, p 0.9 outside, noise std 0.1 (the tunnel_eval.sh condition).
# Writes outputs/tunnel/pid_retune/kp<kp>_ki<ki>_kd<kd>_t<p>[_seed<EVAL_SEED>]/ and starts CARLA on a port if it is not up.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$REPO/.venv/bin/python}
[ -x "$PY" ] || { echo "no $PY: run analysis/setup_venv.sh first" >&2; exit 1; }
CARLA=${CARLA:-$HOME/carla_ws/CARLA_0.9.16/CarlaUE4.sh}
N=${N:-100}
EVAL_SEED=${EVAL_SEED:-1000}
BRAKE_PROB=${BRAKE_PROB:-0.5}
R=$REPO/outputs/tunnel/pid_retune; mkdir -p "$R/scen"
cd "$REPO"
say(){ echo "$(date '+%F %T') $*"; }
start_carla(){
  local P=$1
  ss -ltn | grep -q ":$P " && return 0
  setsid nohup "$CARLA" -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$P > "$R/carla_$P.log" 2>&1 < /dev/null &
  for _ in $(seq 1 30); do sleep 4; ss -ltn | grep -q ":$P " && break; done
  sleep 45
  ss -ltn | grep -q ":$P " && say "carla $P started" || { say "carla $P failed to start"; return 1; }
}
run_cell(){
  local PORT=$1 SPEC=$2 GAINS P KP KI KD NAME SCEN OUT TRY EPS
  GAINS=${SPEC%@*}; P=1.0; [ "$SPEC" != "$GAINS" ] && P=${SPEC#*@}
  IFS=, read -r KP KI KD <<< "$GAINS"
  NAME=kp${KP}_ki${KI}_kd${KD}_t$P; [ "$EVAL_SEED" != 1000 ] && NAME=${NAME}_seed$EVAL_SEED
  SCEN=$R/scen/kp${KP}_ki${KI}_kd${KD}.json
  "$PY" - "$REPO/configs/scenario_town03.json" "$SCEN" "$KP" "$KI" "$KD" <<'EOF'
import json, sys
from pathlib import Path
src, dst, kp, ki, kd = sys.argv[1:]
c = json.loads(Path(src).read_text())
c["stack_file"] = str(Path(src).parent / c.get("stack_file", "stack.json"))
c["follower_control"].update(kp=float(kp), ki=float(ki), kd=float(kd))
Path(dst).write_text(json.dumps(c, indent=2))
EOF
  if [ "$P" = 1.0 ]; then COND="--p-success 1.0 --p-tunnel 1.0 --p-tunnel-noise-std 0"
  else COND="--p-success 0.9 --p-tunnel $P --p-tunnel-noise-std 0.1 --p-tunnel-noise-tau-sec 1.0"; fi
  OUT=$R/$NAME
  for TRY in 1 2 3; do
    rm -rf "$OUT"
    timeout 1800 "$PY" episode_ppo_v2.py --controller pid --scenario "$SCEN" --no-rendering \
      --train-episodes "$N" --seed "$EVAL_SEED" $COND --tunnel-burst-ticks 0 --leader-brake-prob "$BRAKE_PROB" \
      --port "$PORT" --tm-port $((PORT + 6000)) --max-connect-wait-sec 300 --output-dir "$OUT" > "$OUT.log" 2>&1
    EPS=$(grep -cE 'Episode [0-9]+ ended' "$OUT.log")
    say "$NAME port=$PORT try=$TRY episodes=$EPS"
    [ "$EPS" -ge "$N" ] && return 0
    sleep 20
  done
  say "CELL_FAILED $NAME"
}
declare -A QUEUE
for ITEM in $1; do QUEUE[${ITEM%%:*}]+=" ${ITEM#*:}"; done
# One server at a time, each with its client attached before the next server starts: an idle
# CARLA server holds about 4 GB of VRAM (2.4 GB once a no-rendering client is connected), and two
# idle servers do not fit in 8 GB (VK_ERROR_DEVICE_LOST).
for PORT in "${!QUEUE[@]}"; do
  start_carla "$PORT" || exit 1
  ( for SPEC in ${QUEUE[$PORT]}; do run_cell "$PORT" "$SPEC"; done ) &
  WORKERS+=" $!"
  sleep 40
done
wait $WORKERS   # the workers only: a CARLA server started above is also a child and never exits
say "SWEEP_DONE"

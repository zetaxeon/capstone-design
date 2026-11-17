#!/bin/bash
# Three-vehicle platoon evaluation (leader + 2 followers): PID and every tunnel checkpoint over
# in-tunnel p 1.0 / 0.9 / 0.5 / 0.3 / 0.1, 100 episodes per cell, two CARLA servers in parallel.
# p = 1.0 is the no-loss reference (p 1.0 outside as well, no noise); the other cells use p 0.9
# outside and noise std 0.1, as in the two-vehicle evaluation.
# Usage: analysis/platoon_eval.sh            (writes outputs/tunnel/eval3/<controller>_t<p>/)
# A cell that does not reach N episodes is retried up to twice after restarting its server.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
CARLA=${CARLA:-$HOME/carla_ws/CARLA_0.9.16/CarlaUE4.sh}
N=${N:-100}
OUT=$REPO/outputs/tunnel/eval3; mkdir -p "$OUT"
cd "$REPO"
say(){ echo "$(date '+%F %T') $*"; }
restart_carla(){
  local P=$1
  pkill -f "Shipping.*carla-rpc-port=$P"; sleep 8; pkill -9 -f "Shipping.*carla-rpc-port=$P"; sleep 4
  setsid nohup "$CARLA" -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$P > "$OUT/carla_$P.log" 2>&1 < /dev/null &
  for _ in $(seq 1 30); do sleep 4; ss -ltn | grep -q ":$P " && break; done
  sleep 60
  say "carla $P restarted"
}
cell_ok(){ [ -f "$OUT/$1.log" ] && [ "$(grep -cE 'Episode [0-9]+ ended' "$OUT/$1.log")" -ge "$N" ] && [ "$(grep -ci traceback "$OUT/$1.log")" = 0 ]; }
run_cell(){   # PORT CONTROLLER P     (CONTROLLER: pid | lag_s42 | ppo_s43 ...)
  local PORT=$1 C=$2 P=$3 TRY COND
  if [ "$P" = 1.0 ]; then COND="P_OUT=1.0 NOISE_STD=0"; else COND="P_OUT=0.9 NOISE_STD=0.1"; fi
  cell_ok "${C}_t$P" && { say "${C}_t$P already done"; return 0; }
  for TRY in 1 2 3; do
    [ $TRY -gt 1 ] && restart_carla "$PORT"
    if [ "$C" = pid ]; then
      env $COND FOLLOWERS=2 EVAL_DIR="$OUT" N=$N SWEEP="$P" RUN_PID=1 CKPTS=none bash analysis/tunnel_eval.sh "$PORT" > /dev/null
    else
      env $COND FOLLOWERS=2 EVAL_DIR="$OUT" N=$N SWEEP="$P" RUN_PID=0 CKPTS="$C" bash analysis/tunnel_eval.sh "$PORT" > /dev/null
    fi
    if cell_ok "${C}_t$P"; then say "${C}_t$P port=$PORT try=$TRY ok"; return 0; fi
    say "${C}_t$P port=$PORT try=$TRY incomplete"
  done
  say "CELL_FAILED ${C}_t$P"
}
queue(){ local PORT=$1; shift; for ITEM in "$@"; do run_cell "$PORT" "${ITEM%@*}" "${ITEM#*@}"; done; }
ALL="pid lag_s42 lag_s43 lag_s44 ppo_s42 ppo_s43 ppo_s44"
QA=(); QB=()
for C in $ALL; do QA+=("$C@1.0" "$C@0.9"); QB+=("$C@0.3" "$C@0.1"); done
for C in pid lag_s42 lag_s43 lag_s44; do QA+=("$C@0.5"); done
for C in ppo_s42 ppo_s43 ppo_s44; do QB+=("$C@0.5"); done
for PORT in 2000 2010; do ss -ltn | grep -q ":$PORT " || { restart_carla $PORT; sleep 30; }; done
queue 2000 "${QA[@]}" & A=$!
queue 2010 "${QB[@]}" & B=$!
wait $A $B
say "PLATOON_EVAL_DONE failed=$(grep -c CELL_FAILED "$OUT/pipeline.log" 2>/dev/null)"

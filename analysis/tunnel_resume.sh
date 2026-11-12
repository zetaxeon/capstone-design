#!/bin/bash
# Finish the tunnel experiment from the state of 2026-11-12 (see docs/RESUME.md):
#   1. retrain vanilla PPO seed 44 from scratch on CARLA port 2000, with a stall watchdog
#      (a silent train.log for STALL_SEC seconds = frozen CARLA tick: kill, restart CARLA, retrain; 3 attempts)
#   2. meanwhile evaluate PID, PPO-Lag s42-44 and PPO s42-43 on port 2010 (finished cells are skipped)
#   3. evaluate PPO s44 on both servers
# Usage (detached, from the repo root):
#   setsid nohup bash analysis/tunnel_resume.sh > outputs/tunnel/pipeline_resume.log 2>&1 < /dev/null &
# Progress markers in that log: TRAIN_DONE | TRAIN_FAILED, then EVAL_DONE failed_cells=N.
# The script starts and restarts its own CARLA servers. Two servers only: a third does not fit in 8 GB VRAM.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$REPO/outputs/tunnel
CARLA=${CARLA:-$HOME/carla_ws/CARLA_0.9.16/CarlaUE4.sh}
N=100
STALL_SEC=600
say(){ echo "$(date '+%F %T') $*"; }
restart_carla(){
  local P=$1 TRY
  for TRY in 1 2 3 4 5; do
    pkill -f "Shipping.*carla-rpc-port=$P"; sleep 8; pkill -9 -f "Shipping.*carla-rpc-port=$P"; sleep 4
    setsid nohup "$CARLA" -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$P > "$T/carla_$P.log" 2>&1 < /dev/null &
    for _ in $(seq 1 30); do sleep 4; ss -ltn | grep -q ":$P " && break; done
    sleep 45
    if ss -ltn | grep -q ":$P "; then say "carla $P (re)started (try $TRY)"; return 0; fi
  done
  say "carla $P could not be restarted"; return 1
}
ensure_carla(){ ss -ltn | grep -q ":$1 " || restart_carla "$1"; }
cell_done(){ [ "$(grep -cE 'Episode [0-9]+ ended' "$T/eval/$1.log" 2>/dev/null)" -ge $N ] && [ "$(grep -ci traceback "$T/eval/$1.log" 2>/dev/null)" = 0 ]; }
worker(){  # port, cells "name:p ..."
  local PORT=$1 C NAME P TRY
  for C in $2; do
    NAME=${C%%:*}; P=${C##*:}
    for TRY in 1 2 3; do
      cell_done "${NAME}_t$P" && break
      [ $TRY -gt 1 ] && restart_carla "$PORT"
      ensure_carla "$PORT" || return 1
      if [ "$NAME" = pid ]; then SWEEP=$P RUN_PID=1 CKPTS=none analysis/tunnel_eval.sh "$PORT"
      else SWEEP=$P RUN_PID=0 CKPTS=$NAME analysis/tunnel_eval.sh "$PORT"; fi | grep -v 'tunnel_eval done'
    done
    cell_done "${NAME}_t$P" || say "CELL_FAILED ${NAME}_t$P"
  done
}
train_s44(){
  local ATT L=$T/train_ppo_s44/train.log TB AGE NE E
  for ATT in 1 2 3; do
    [ -d "$T/train_ppo_s44" ] && mv "$T/train_ppo_s44" "$T/failed_ppo_s44_$(date +%m%d_%H%M%S)"
    restart_carla 2000 || return 1
    analysis/tunnel_train.sh 2000 "ppo:44" > "$T/queue_ppo_2000d_$ATT.log" 2>&1 & TB=$!
    say "ppo:44 training attempt $ATT started"
    while kill -0 $TB 2>/dev/null; do
      sleep 60
      [ -f "$L" ] || continue
      AGE=$(( $(date +%s) - $(stat -c %Y "$L") ))
      if [ $AGE -gt $STALL_SEC ]; then
        say "ppo:44 stalled (log silent ${AGE}s at episode $(grep -cE 'Episode [0-9]+ ended' "$L")), killing"
        pkill -9 -P $TB; sleep 3; kill $TB 2>/dev/null; break
      fi
    done
    wait $TB 2>/dev/null
    NE=$(grep -cE 'Episode [0-9]+ ended' "$L"); E=$(grep -ci traceback "$L")
    say "ppo:44 attempt $ATT episodes=$NE tracebacks=$E"
    [ "$NE" -ge 3000 ] && [ "$E" -eq 0 ] && [ -f "$T/train_ppo_s44/ppo_lag.pt" ] && return 0
  done
  return 1
}
cd "$REPO"
say "RESUME start"
train_s44 > "$T/train_s44_watch.log" 2>&1 & TR=$!
sleep 240   # let the new CARLA server finish loading before another CUDA client starts
B=""
for P in 0.9 0.7 0.5 0.3 0.1; do for NAME in pid lag_s42 lag_s43 lag_s44 ppo_s42 ppo_s43; do B="$B $NAME:$P"; done; done
worker 2010 "$B" > "$T/eval_2010.log" 2>&1 & WB=$!
wait $TR; RC=$?
cat "$T/train_s44_watch.log"
if [ $RC -ne 0 ]; then say "TRAIN_FAILED"; wait $WB; say "EVAL_PART_DONE"; exit 1; fi
say "TRAIN_DONE"
worker 2000 "ppo_s44:0.9 ppo_s44:0.7 ppo_s44:0.5" > "$T/eval_2000.log" 2>&1 & WA=$!
wait $WB
worker 2010 "ppo_s44:0.1 ppo_s44:0.3" >> "$T/eval_2010.log" 2>&1 & WC=$!
wait $WA $WC
say "EVAL_DONE failed_cells=$(cat $T/eval_2000.log $T/eval_2010.log | grep -c CELL_FAILED)"

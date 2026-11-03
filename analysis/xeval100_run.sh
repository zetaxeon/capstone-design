#!/bin/bash
# Cross-evaluation, 100 deterministic episodes per cell:
# {run1026 (4d, p=1.0 trained), comm09 (7d, p=0.9), emcost (7d, p=0.9, entry cost)} x p_success {1.0, 0.9, 0.5}.
# Waits for the emcost training (pid given as $1) to exit; checkpoints are copied into outputs/xeval_100 first; the run folders are only read. Reruns delete and rewrite each cell folder.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
X=$REPO/outputs/xeval_100
mkdir -p $X
N=100
TRAIN_PID=${1:-}
if [ -n "$TRAIN_PID" ]; then while ps -p "$TRAIN_PID" >/dev/null 2>&1; do sleep 30; done; fi
echo "training exited $(date '+%F %T')"
ENDED=$(grep -cE 'Episode [0-9]+ ended' $REPO/outputs/run_emcost_3000/train.log)
ERRS=$(grep -ciE 'traceback|error' $REPO/outputs/run_emcost_3000/train.log)
echo "training episodes ended=$ENDED errors=$ERRS"
if [ "$ENDED" -lt 3000 ]; then echo "ABORT: training did not reach 3000 episodes"; exit 2; fi
cp $REPO/outputs/run1026_3000/ppo_lag.pt $X/ckpt_run1026.pt
cp $REPO/outputs/run_comm09_3000/ppo_lag.pt $X/ckpt_comm09.pt
cp $REPO/outputs/run_emcost_3000/ppo_lag.pt $X/ckpt_emcost.pt
sha256sum $REPO/outputs/run1026_3000/ppo_lag.pt $REPO/outputs/run_comm09_3000/ppo_lag.pt $REPO/outputs/run_emcost_3000/ppo_lag.pt $X/ckpt_*.pt
cd $REPO
for CK in run1026 comm09 emcost; do
  for P in 1.0 0.9 0.5; do
    OUT=$X/${CK}_p$P
    rm -rf $OUT
    timeout 1800 python3 episode_ppo_v2.py --controller ppo --ppo-eval --no-rendering \
      --train-episodes $N --p-success $P --ppo-checkpoint $X/ckpt_$CK.pt \
      --output-dir $OUT --max-connect-wait-sec 180 > $OUT.log 2>&1
    echo "$CK p=$P exit=$? episodes=$(grep -cE 'Episode [0-9]+ ended' $OUT.log) $(date '+%T')"
  done
done
echo "xeval100 done $(date '+%T')"

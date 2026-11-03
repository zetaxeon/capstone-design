#!/bin/bash
# Cross-evaluation: {run1026 4d, comm09 7d} x p_success {1.0, 0.9, 0.5}, 20 deterministic episodes each.
# Checkpoints are copied into the output folder first; outputs/ is only read.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
S=$REPO/outputs/xeval_20
mkdir -p $S
N=20
while ps -p 20187 >/dev/null 2>&1; do sleep 30; done
echo "training exited $(date '+%T')"
cp $REPO/outputs/run1026_3000/ppo_lag.pt $S/ckpt_run1026.pt
cp $REPO/outputs/run_comm09_3000/ppo_lag.pt $S/ckpt_comm09.pt
sha256sum $REPO/outputs/run1026_3000/ppo_lag.pt $S/ckpt_run1026.pt $REPO/outputs/run_comm09_3000/ppo_lag.pt $S/ckpt_comm09.pt
cd $REPO
for CK in run1026 comm09; do
  for P in 1.0 0.9 0.5; do
    OUT=$S/${CK}_p$P
    rm -rf $OUT
    timeout 1800 python3 episode_ppo_v2.py --controller ppo --ppo-eval --no-rendering \
      --train-episodes $N --p-success $P --ppo-checkpoint $S/ckpt_$CK.pt \
      --output-dir $OUT --max-connect-wait-sec 120 > $OUT.log 2>&1
    echo "$CK p=$P exit=$? episodes=$(grep -cE 'Episode [0-9]+ ended' $OUT.log) $(date '+%T')"
  done
done
echo "xeval done $(date '+%T')"

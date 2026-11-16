#!/bin/bash
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$REPO/.venv/bin/python}   # project venv (analysis/setup_venv.sh); PYTHON=... overrides
[ -x "$PY" ] || { echo "no $PY: run analysis/setup_venv.sh first" >&2; exit 1; }
X=$REPO/outputs/xeval_100
mkdir -p $X
cd $REPO
for P in 1.0 0.9 0.5; do
  OUT=$X/emcost_p$P; rm -rf $OUT
  timeout 1800 "$PY" episode_ppo_v2.py --controller ppo --ppo-eval --no-rendering --train-episodes 100 --p-success $P \
    --ppo-checkpoint $X/ckpt_emcost.pt --output-dir $OUT --max-connect-wait-sec 180 > $OUT.log 2>&1
  echo "emcost p=$P exit=$? episodes=$(grep -cE 'Episode [0-9]+ ended' $OUT.log) $(date '+%T')"
done

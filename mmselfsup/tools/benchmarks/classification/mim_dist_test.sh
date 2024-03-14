#!/usr/bin/env bash

set -x

CFG=$1
CHECKPOINT=$2
GPUS=${GPUS:-8}
PY_ARGS=${@:3}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/dssg/home/acct-medftn/medftn/.conda/envs/mmselfsup_yzc/bin/python -m mim test mmcls \
    $CFG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    -G 1 \
    $PY_ARGS

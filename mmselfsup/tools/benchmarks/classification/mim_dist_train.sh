#!/usr/bin/env bash

set -x

CFG=$1
PRETRAIN=$2
GPUS=${GPUS:-4}
PY_ARGS=${@:3}




/dssg/home/acct-medftn/medftn/.conda/envs/mmselfsup_yzc/bin/python -m mim train mmcls $CFG \
    --launcher pytorch -G 1 \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
    model.backbone.init_cfg.prefix="backbone." \
    $PY_ARGS

#!/bin/bash
set -xe

HOME_PATH=$(cd "$(dirname "$0")"; pwd)

devices=0
filename=${HOME_PATH}/../data/feature/3jvr.h5
checkpoint=${HOME_PATH}/../checkpoint/model.pkl

CUDA_VISIBLE_DEVICES=${devices} python ${HOME_PATH}/inference.py \
    --filename=${filename} \
    --checkpoint=${checkpoint} \
    --cuda

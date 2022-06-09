#!/bin/bash
set -xe

HOME_PATH=$(cd "$(dirname "$0")"; pwd)

prot_filename=${HOME_PATH}/../data/dataset/3jvr/3jvr_protein.pdb
lig_filename=${HOME_PATH}/../data/dataset/3jvr/3jvr_ligand.mol2
feature_filename=${HOME_PATH}/../data/feature/3jvr.h5

python ${HOME_PATH}/data.py \
    --prot_filename=${prot_filename} \
    --lig_filename=${lig_filename} \
    --feature_filename=${feature_filename}

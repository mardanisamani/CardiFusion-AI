#!/usr/bin/env bash
# AI-assisted helper for the ARCADE interview project.
# This script DOES NOT auto-download. It prints the exact commands so you can
# review disk/space/credentials first, then run them yourself.
set -euo pipefail

DATA_DIR="${1:-data/raw}"
echo "ARCADE download plan (record: Zenodo 10390295, ~ a few GB)."
echo "Target: ${DATA_DIR}"
echo
cat <<EOF
# 1) Create the target dir
mkdir -p ${DATA_DIR}

# 2) Option A — Zenodo (recommended). Replace the URL with the file links shown
#    on https://zenodo.org/records/10390295
#    (the record bundles the 'syntax' and 'stenosis' COCO datasets):
# wget -P ${DATA_DIR} "https://zenodo.org/records/10390295/files/<FILE>.zip?download=1"

# 3) Option B — GitHub mirror:
# git clone https://github.com/cmctec/ARCADE ${DATA_DIR}/ARCADE

# 4) Unzip; you should end up with two sub-datasets, each split 1000/200/300:
#    ${DATA_DIR}/syntax/{train,val,test}/{images,annotations}
#    ${DATA_DIR}/stenosis/{train,val,test}/{images,annotations}
# unzip '${DATA_DIR}/*.zip' -d ${DATA_DIR}
EOF
echo
echo "After download, run scripts/prepare_data.sh to convert to masks + YOLO labels."

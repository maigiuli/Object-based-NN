#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running $1 on: $(hostname)"
nvidia-smi

echo "Starting job..."

source /eos/user/g/gimainer/myenv_py39/bin/activate

#cd /afs/cern.ch/user/g/gimainer/QT/



SAMPLE=$1
TRAIN_FILES=$2
TEST_FILES=$3
BETA=$4

# Converte stringhe separate da ; in array
IFS=';' read -r -a TRAIN_ARRAY <<< "$TRAIN_FILES"
IFS=';' read -r -a TEST_ARRAY <<< "$TEST_FILES"

echo "Running sample: $SAMPLE"
echo "Training files: ${TRAIN_ARRAY[@]}"
echo "Testing files: ${TEST_ARRAY[@]}"
echo "Testing beta: $BETA"

python3 /afs/cern.ch/user/g/gimainer/QT/noMET_NPV_standard/NORM-Multi-8020-withCST-withGaussian-noMETtarget.py  \
    --sample "$SAMPLE" \
    --train_files "${TRAIN_ARRAY[@]}" \
    --test_files "${TEST_ARRAY[@]}" \
    --beta $BETA

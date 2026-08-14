#!/bin/bash
set -euo pipefail
MERGED_ROOT="$1"
BASE="$2"
JAN_NANO="$3"
ABLITERATED="$4"
PYTHAGORAS_PROVER="$5"
QVIKHR_INSTRUCTION="$6"
CHINESE_ERROR_CORRECTOR="$7"
MET_D="$8"
OSIM="${9}"

echo "=== Disk layout ==="
df -h

# rw_mount writes ALSO buffer through the same local AZ_BATCH_NODE_ROOT_DIR
# quota (~64GB) before flushing to blob. Only ever keep the current + next
# stage's output resident (delete the previous stage as soon as the next
# has consumed it) -- same discipline as the working combined job.
mkdir -p ./work
cp configs/arcee-fusion-stage1-mounted.yml ./work/stage1.yml
sed -i "s|{{BASE}}|$BASE|g; s|{{JAN_NANO}}|$JAN_NANO|g" ./work/stage1.yml
cp configs/arcee-fusion-stage2-mounted.yml ./work/stage2.yml
sed -i "s|{{STAGE1}}|$MERGED_ROOT/fusion-stage1|g; s|{{ABLITERATED}}|$ABLITERATED|g" ./work/stage2.yml
cp configs/arcee-fusion-stage3-mounted.yml ./work/stage3.yml
sed -i "s|{{STAGE2}}|$MERGED_ROOT/fusion-stage2|g; s|{{PYTHAGORAS_PROVER}}|$PYTHAGORAS_PROVER|g" ./work/stage3.yml
cp configs/arcee-fusion-stage4-mounted.yml ./work/stage4.yml
sed -i "s|{{STAGE3}}|$MERGED_ROOT/fusion-stage3|g; s|{{QVIKHR_INSTRUCTION}}|$QVIKHR_INSTRUCTION|g" ./work/stage4.yml
cp configs/arcee-fusion-stage5-mounted.yml ./work/stage5.yml
sed -i "s|{{STAGE4}}|$MERGED_ROOT/fusion-stage4|g; s|{{CHINESE_ERROR_CORRECTOR}}|$CHINESE_ERROR_CORRECTOR|g" ./work/stage5.yml
cp configs/arcee-fusion-stage6-mounted.yml ./work/stage6.yml
sed -i "s|{{STAGE5}}|$MERGED_ROOT/fusion-stage5|g; s|{{MET_D}}|$MET_D|g" ./work/stage6.yml
cp configs/arcee-fusion-stage7-mounted.yml ./work/stage7.yml

echo "=== Merging: arcee-fusion (7 sequential stages) ==="
mergekit-yaml ./work/stage1.yml "$MERGED_ROOT/fusion-stage1" --cuda --lazy-unpickle --allow-crimes
df -h
stage=2
while [ $stage -le 6 ]; do
  prev=$((stage - 1))
  mergekit-yaml "./work/stage${stage}.yml" "$MERGED_ROOT/fusion-stage${stage}" --cuda --lazy-unpickle --allow-crimes
  rm -rf "$MERGED_ROOT/fusion-stage${prev}"
  df -h
  stage=$((stage + 1))
done
sed -i "s|{{STAGE6}}|$MERGED_ROOT/fusion-stage6|g; s|{{OSIM}}|$OSIM|g" ./work/stage7.yml
mergekit-yaml ./work/stage7.yml "$MERGED_ROOT/arcee-fusion-7way" --cuda --lazy-unpickle --allow-crimes
rm -rf "$MERGED_ROOT/fusion-stage6"

echo "=== Final disk layout ==="
df -h
echo "MERGE_COMPLETE"

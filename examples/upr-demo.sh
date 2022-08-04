#! /bin/bash

export TRANSFORMERS_CACHE="/mnt/disks/project/transformers-cache"

BASE_DIR="downloads/data/"
DATASET="nq"
SPLIT="test"

MODEL="T0_3B"
HF_MODEL="bigscience/${MODEL}"
# Other possible options are MODEL="t5-v1_1-xl / t5-xl-lm-adapt and HF_MODEL="google/${MODEL}"

RETRIEVER="bm25"
TOPK=1000
EVIDENCE_DATA_PATH="${BASE_DIR}/wikipedia-split/psgs_w100.tsv"

WORLD_SIZE=8
DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${WORLD_SIZE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"


ARGS=" \
  --num-workers 2 \
  --log-interval 1 \
  --topk-passages ${TOPK} \
  --shard-size 32 \
  --hf-model-name ${HF_MODEL} \
  --use-gpu \
  --use-bf16 \
  --report-topk-accuracies 1 5 20 100 \
  --evidence-data-path ${EVIDENCE_DATA_PATH} \
  --retriever-topk-passages-path ${BASE_DIR}/retriever-outputs/${RETRIEVER}/${DATASET}-${SPLIT}.json \
  --reranker-output-dir ${BASE_DIR}/retriever-outputs/${RETRIEVER}/reranked/ \
  --merge-shards-and-save \
  --special-suffix ${DATASET}-${SPLIT}-plm-${MODEL}-topk-${TOPK} "

# `--use-bf16` option provides speed ups and memory savings on Ampere GPUs such as A100 or A6000.
# However, when working with V100 GPUs, this argument should be removed.


COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} upr.py ${ARGS}"
eval "${COMMAND}"
exit

#! /bin/bash

USE_SLURM="false"
NPROC=8
CONFIG="base"
TOPK=100
DATASET="nq"
BATCH_SIZE=64

# It can be either "train" or "inference"
MODE="inference"

# Please set the absolute path of the BASE_DIR
BASE_DIR="/mnt/disks/project/downloads/data"

DATA_DIR="${BASE_DIR}/retriever-outputs/mss-dpr"

EVIDENCE_DATA_PATH="${BASE_DIR}/wikipedia-split/psgs_w100.tsv"
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"
MMAP_INDEXED_EVIDENCE_PATH="${BASE_DIR}/evidence-wikipedia-indexed-mmap/wikipedia-evidence_text_document"
MMAP_INDEXED_TITLE_PATH="${BASE_DIR}/evidence-wikipedia-indexed-mmap/wikipedia-evidence_title_document"


CHECKPOINT_PATH="${BASE_DIR}/checkpoints/fid-mss-dpr-${DATASET}-${CONFIG}-topk${TOPK}-bsize${BATCH_SIZE}"

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${NPROC} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"
NAME="fid-${DATASET}-${CONFIG}-top${TOPK}-bsize${BATCH_SIZE}"
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs/fid

# For training FiD, please point to the paths of the correct VALID_DATA and TEST_DATA.
# For example:
# During training, VALID_DATA="${DATA_DIR}/nq-dev.json"
# During inference, VALID_DATA="${DATA_DIR}/reranked/nq-dev.json"

if [ ${DATASET} == "nq" ]; then
    TRAIN_DATA="${DATA_DIR}/nq-train.json"
    VALID_DATA="${DATA_DIR}/reranked/nq-dev.json"
    TEST_DATA="${DATA_DIR}/reranked/nq-test.json"
    EPOCHS=10

elif [ ${DATASET} == "trivia" ]; then
    TRAIN_DATA="${DATA_DIR}/trivia-train.json"
    VALID_DATA="${DATA_DIR}/reranked/trivia-dev.json"
    TEST_DATA="${DATA_DIR}/reranked/trivia-test.json"
    EPOCHS=10

elif [ ${DATASET} == "squad1" ]; then
    TRAIN_DATA="${DATA_DIR}/squad1-train.json"
    VALID_DATA="${DATA_DIR}/reranked/squad1-dev.json"
    TEST_DATA="${DATA_DIR}/reranked/squad1-test.json"
    EPOCHS=10

else
    echo "Unsupported dataset"
    exit 1
fi


function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--kv-channels 64 \
--ffn-hidden-size 3072 \
--model-parallel-size 1"
}

function config_large() {
    export CONFIG_ARGS="--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 16 \
--kv-channels 64 \
--ffn-hidden-size 4096 \
--model-parallel-size 1"
}


if [ ${CONFIG} == "base" ]; then
    config_base
    READER_CHKPT_PATH="${BASE_DIR}/checkpoints/mss-emdr2-reader-base-steps82k"

elif [ ${CONFIG} == "large" ]; then
    config_large
    READER_CHKPT_PATH="${BASE_DIR}/checkpoints/t5-large"

else
    echo "Invalid model configuration"
    exit 1
fi


if [ ${MODE} == "train" ]; then
    MODE_OPTIONS="--fid-training"

elif [ ${MODE} == "inference" ]; then
    MODE_OPTIONS="--no-load-optim"

else
    echo "Invalid mode option ${MODE}"
    exit 1
fi


OPTIONS=" \
          --train-data $TRAIN_DATA \
          --valid-data $VALID_DATA \
          --test-data $TEST_DATA \
          --evidence-data-path ${EVIDENCE_DATA_PATH} \
          --indexed-evidence-data-path ${MMAP_INDEXED_EVIDENCE_PATH} \
          --indexed-title-data-path ${MMAP_INDEXED_TITLE_PATH} \
          --save-interval 500 \
          --save ${CHECKPOINT_PATH} \
          --load ${CHECKPOINT_PATH} \
          --pretrained-t5-load ${READER_CHKPT_PATH} \
          --log-interval 20 \
          --eval-interval 500 \
          --eval-iters 10 \
          --weight-decay 1.0e-1 \
          --seq-length 512 \
          --seq-length-ret 256 \
          --decoder-seq-length 64 \
          --max-decode-len 64 \
          --max-position-embeddings 512 \
          --fp16 \
          --vocab-file ${VOCAB_FILE} \
          --model-parallel-size 1 \
          --num-workers 2 \
          --distributed-backend nccl \
          --checkpoint-activations \
          --task OPENQA \
          --tokenizer-type BertWordPieceLowerCase \
          --epochs ${EPOCHS} \
          --sample-rate 1.0 \
          --batch-size 1 \
          --eval-batch-size 2 \
          --beam-size 1 \
          --lr 2e-5 \
          --warmup 0.01 \
          --DDP-impl local \
          --lr-decay-style linear \
          --max-training-rank ${NPROC} \
          --topk-retrievals ${TOPK} \
          --allow-trivial-doc "


if [ ${USE_SLURM} == "false" ]; then
    COMMAND="WORLD_SIZE=${NPROC} python ${DISTRIBUTED_ARGS} tasks/run.py ${OPTIONS} ${CONFIG_ARGS} ${MODE_OPTIONS}"
    eval "${COMMAND}"

elif [ ${USE_SLURM} == "true" ]; then
    module purge
    module load anaconda3/2021.05
    module load cuda/10.2
    module load gcc/7.3.0
    source activate mypyt

    run_cmd="python -u ${DIR}/tasks/run.py ${OPTIONS} ${CONFIG_ARGS} ${MODE_OPTIONS} --use-slurm"
    srun -l --output=$DIR/logs/fid/${NAME}_%x_%j_$DATETIME.log sh -c "${run_cmd}"
    set +x

fi

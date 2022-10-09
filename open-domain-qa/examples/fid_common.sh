#! /bin/bash

#SBATCH -p devlab -t 02:00:00 --nodes=4 --exclusive --gres=gpu:8 --mem=450G --overcommit --ntasks-per-node=8 --dependency=singleton --constraint=volta32gb --job-name=squad1-mss-dpr-fid-base-bs64-topk100-inference-ll

USE_SLURM="true"
NPROC=8
CONFIG="base"
TOPK=100
DATASET="squad1"
BATCH_SIZE=64
PER_GPU_BATCH_SIZE=1

# It can be either train or inference
MODE="inference"

BASE_DIR="/mnt/disks/project/data"
DATA_DIR="${BASE_DIR}/retriever-outputs/dpr-emdr2/top-1000-outputs-mss-dpr"
EVIDENCE_DATA_PATH="${BASE_DIR}/evidence-en/psgs_w100.tsv" 
#psgs_w100.tsv"

VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"

CHECKPOINT_PATH="${BASE_DIR}/checkpoints/fid-mss-dpr-${DATASET}-${CONFIG}-topk${TOPK}-bsize${BATCH_SIZE}"
#rm -rf ${CHECKPOINT_PATH}

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${NPROC} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"
NAME="fid-${DATASET}-${CONFIG}-top${TOPK}-bsize${BATCH_SIZE}"
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs/fid


if [ ${DATASET} == "nq" ]; then
    TRAIN_DATA="${DATA_DIR}/nq-train.json"
    VALID_DATA="${DATA_DIR}/reranking/nq-dev.json"
    TEST_DATA="${DATA_DIR}/reranking/nq-test.json"
    EPOCHS=10

elif [ ${DATASET} == "trivia" ]; then
    TRAIN_DATA="${DATA_DIR}/trivia-train.json"
    VALID_DATA="${DATA_DIR}/reranking/trivia-dev.json"
    TEST_DATA="${DATA_DIR}/reranking/trivia-test.json"
    EPOCHS=10

elif [ ${DATASET} == "squad1" ]; then
    TRAIN_DATA="${DATA_DIR}/squad1-train.json"
    VALID_DATA="${DATA_DIR}/squad1-dev.json"
    #VALID_DATA="${BASE_DIR}/retriever-outputs/sparse-dense-hybrid/dpr-bm25/reranking/squad1-test.json"
    TEST_DATA="${DATA_DIR}/reranking/squad1-dev.json"
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
    READER_CHKPT_PATH="${BASE_DIR}/checkpoints/mss-t5-base"

elif [ ${CONFIG} == "large" ]; then
    config_large
    READER_CHKPT_PATH="${BASE_DIR}/checkpoints/mss-t5-large"

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
          --indexed-evidence-data-path ${BASE_DIR}/evidence-wikipedia-indexed-mmap/wikipedia-evidence_text_document \
          --indexed-title-data-path ${BASE_DIR}/evidence-wikipedia-indexed-mmap/wikipedia-evidence_title_document \
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

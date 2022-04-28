#! /bin/bash

#SBATCH -p learnlab -t 15:00:00 --nodes=4 --exclusive --gres=gpu:8 --mem=450G --overcommit --ntasks-per-node=8 --dependency=singleton --constraint=volta32gb --job-name=nq-dev-bm25-top900-hf-T0_3B-lf

BASE_DIR="/mnt/disks/project/data/"
DATASET="nq"
SPLIT="test"

MODEL="T0_3B"
HF_MODEL="bigscience/${MODEL}"

# Other options are
#MODEL="t5-xl-lm-adapt"
#MODEL="t5-v1_1-xl"
#HF_MODEL="google/${MODEL}"

RETRIEVER="bm25"
TOPK=1000

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 16 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"
#DISTRIBUTED_ARGS="-m pdb"

WORLD_SIZE=16

#NAME="reranking-${DATASET}-${SPLIT}-${RETRIEVER}-${MODEL}-top${TOPK}"
#DIR=`pwd`
#DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
#mkdir -p $DIR/logs/reranking


if [ ${RETRIEVER} == "bm25" ]
then
    RETRIEVED_DATA_PATH="bm25-dprformat"

elif [ ${RETRIEVER} == "mss-emdr2" ]
then
    RETRIEVED_DATA_PATH="mss-emdr2/top-1000-outputs"

elif [ ${RETRIEVER} == "hybrid" ]
then
    RETRIEVED_DATA_PATH="sparse-dense-hybrid"

elif [ ${RETRIEVER} == "dpr-emdr2" ]
then
    RETRIEVED_DATA_PATH="dpr-emdr2/top-1000-outputs-dpr"
    #RETRIEVED_DATA_PATH="dpr-emdr2/top-1000-outputs-mss-dpr"

elif [ ${RETRIEVER} == "emdr2" ]
then
    RETRIEVED_DATA_PATH="emdr2/top-2000-outputs"

elif [ ${RETRIEVER} == "dpr-hybrid" ]
then
    RETRIEVED_DATA_PATH="sparse-dense-hybrid/dpr-bm25"

elif [ ${RETRIEVER} == "contriever" ]
then
    RETRIEVED_DATA_PATH="contriever/top-1000-outputs"

elif [ ${RETRIEVER} == "contriever-bm25-hybrid" ]
then
    RETRIEVED_DATA_PATH="sparse-dense-hybrid/contriever-bm25"
fi

#export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1

ARGS=" \
	--num-workers 2 \
	--batch-size 1 \
  --log-interval 1 \
  --topk-contexts ${TOPK} \
  --shard-size 20 \
  --sample-rate 1.0 \
  --task-name reranking \
  --hf-model-name ${HF_MODEL} \
  --use-gpu \
  --use-nccl-reduce \
  --report-topk-accuracies 1 5 10 20 50 100 \
  --evidence-data-path ${BASE_DIR}/dpr/wikipedia_split/psgs_w100.tsv \
  --retriever-output-path ${BASE_DIR}/retriever-outputs/${RETRIEVED_DATA_PATH}/${DATASET}-${SPLIT}.json \
  --output-path ${BASE_DIR}/retriever-outputs/${RETRIEVED_DATA_PATH}/reranking/ \
  --merge-shards-and-save \
  --special-suffix ${DATASET}-${SPLIT}-T03B-${TOPK} "


#run_cmd="python -u ${DIR}/reranking_t5_hf.py ${ARGS}"

COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} reranking_t5_hf.py ${ARGS}"
eval "${COMMAND}"
exit

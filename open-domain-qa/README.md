
<a id="Open-Domain QA Experiments"></a>
# Contents
<!-- MarkdownTOC -->

- [Downloading Data and Checkpoints](#downloading-data-and-checkpoints)
- [Usage](#usage)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)

<!-- /MarkdownTOC -->

This repository contains the implementation and results of Fusion-in-Decoder (FiD) algorithm for the task of open-domain question answering. 

We demonstrate that performing inference using UPR re-ranked passages and a pre-trained FiD checkpoint leads to an improved answer generation performance.  

<a id="downloading-data-and-checkpoints"></a>
## Downloading Data and Checkpoints
We've provided pretrained checkpoints and datasets on Dropbox for use to train models for dense retrieval and open-domain QA tasks. 
This data can be downloaded here:

*Required data files to be downloaded*
- [Pre-tokenized memory-mapped (indexed) evidence passages and titles](https://www.dropbox.com/s/nc49dkno8o3pgb3/evidence-wikipedia-indexed-mmap.tar.gz)
- [BERT-large vocabulary file](https://www.dropbox.com/s/ttblv1uggd4cijt/bert-large-uncased-vocab.txt)

The pre-tokenized evidence file(s) can be obtained with this command.
```bash
python tools/create_evidence_indexed_dataset.py --input wikipedia-split/psgs_w100.tsv --tsv-keys text title --tokenizer-type BertWordPieceLowerCase --vocab-file bert-vocab/bert-large-uncased-vocab.txt --output-prefix wikipedia-evidence --workers 25
```

*T5 checkpoints for training FiD*
- [T5 pre-trained reader (base configuration)](https://www.dropbox.com/s/33lm2685ifpei4l/mss-emdr2-reader-base-steps82k.tar.gz)
- [T5 pre-trained reader (large configuration)](https://www.dropbox.com/s/wjul4xgkgiuli6s/t5_large.tar.gz)

*Finetuned FiD checkpoints on individual datasets*
* Please download the FiD models for each dataset using their URLs/links provided in the tables below.

*Dataset-specific files*
* Train, dev, and test datasets along with retrieved passages and UPR re-ranked passages can be downloaded as described in the [README.md](../README.md) of the landing page.

<a id="usage"></a>
# Usage

We have provided a demo script for training an FiD model for open-domain QA tasks in [`examples`](./examples) directory.

Please ensure to change the data, config, and checkpoint paths in this scripts.

To train or do inference using a pre-trained model, please see the options and run the script as
```
bash examples/fid_common.sh
```
The default settings in this script are useful for doing inference with pre-trained FiD checkpoint(s).

To train FiD models, please set the paths of the VALID_DATA and TEST_DATA in the above script accordingly.

<a id="pre-trained-checkpoints"></a>
# Pre-trained FiD Checkpoints


## SQuAD-Open

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS | base | 36.2 | 39.6 | [link](https://www.dropbox.com/s/36zhaxim0lax9uv/fid-mss-squad1-base-topk100-bsize64.tar.gz)
| MSS + UPR | base | **43.7** | **50.1** |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| DPR | base | 48.8 | 45.8 | [link](https://www.dropbox.com/s/sb10urzlzs0jcmb/fid-dpr-squad1-base-topk100-bsize64.tar.gz)
| DPR + UPR | base | **51.5** | **54.0** |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS-DPR | base | 50.1 | 52.2 | [link](https://www.dropbox.com/s/7a61gt3zuivxto8/fid-mss-dpr-squad1-base-topk100-bsize64.tar.gz)
| MSS-DPR + UPR | base | **51.9** | **55.6**  |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS-DPR | large | 51.9 | 54.4 | [link](https://www.dropbox.com/s/nx4ahlhuywdvjg8/fid-mss-dpr-squad1-large-topk100-bsize64.tar.gz)
| MSS-DPR + UPR | large | **53.1** | **58.1** | |


## TriviaQA

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS | base | 60.9 | 60.3 | [link](https://www.dropbox.com/s/s92am9wcwunv1q5/fid-mss-trivia-base-topk100-bsize64.tar.gz)
| MSS + UPR | base | **68.5** | **68.9** | |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| DPR | base | 67.9 | 68.5 | [link](https://www.dropbox.com/s/dl0tb4at6090h00/fid-dpr-trivia-base-topk100-bsize64.tar.gz)
| DPR + UPR | base | **70.1** | **71.2** |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS-DPR | base | 69.9 | 70.2 | [link](https://www.dropbox.com/s/7qpnndn3cf0rwkp/fid-mss-dpr-trivia-base-topk100-bsize64.tar.gz)
| MSS-DPR + UPR | base | **71.5** | **71.8** |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS-DPR | large | 71.5 | 71.6 | [link](https://www.dropbox.com/s/2xlemikbnq86yf9/fid-mss-dpr-trivia-large-topk100-bsize64.tar.gz)
| MSS-DPR + UPR | large | **72.7** | **73.2** | |


## Natural Questions

 | Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
 |-----------|---------------|--------|---------|:-----------:
 | MSS | base | 43.7 | 44.5 | [link](https://www.dropbox.com/s/2lbphzcac4w2tfp/fid-mss-nq-base-topk100-bsize64.tar.gz)
 | MSS + UPR | base | **45.8** | **47.3** |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| DPR | base | 49.4 | 50.8 | [link](https://www.dropbox.com/s/sno9ms6zphtajxq/fid-dpr-nq-base-topk100-bsize64.tar.gz)
| DPR + UPR | base | **49.8** | **51.3** |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS-DPR | base | 49.7 | 50.8 | [link](https://www.dropbox.com/s/fo9vgophu81yjm3/fid-mss-dpr-nq-base-topk100-bsize64.tar.gz)
| MSS-DPR + UPR | base | **49.9** | **51.5** |  |

| Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
|-----------|---------------|--------|---------|:-----------:
| MSS-DPR | large | **51.8** | 53.6 | [link](https://www.dropbox.com/s/wqbyq7b96hlaahr/fid-mss-dpr-nq-large-topk100-bsize64.tar.gz)
| MSS-DPR + UPR | large | 51.5 | **54.5** |  | 

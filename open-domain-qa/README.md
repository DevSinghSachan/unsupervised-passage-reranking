
<a id="Open-Domain QA Experiments"></a>
# Contents
<!-- MarkdownTOC -->

TODO: This README file is in the process of being updated.

- [Downloading Data and Checkpoints](#downloading-data-and-checkpoints)
- [Usage](#usage)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)

<!-- /MarkdownTOC -->

This repository contains the implementation of Fusion-in-Decoder (FiD) algorithm for the task of open-domain question answering. 

We demonstrate that performing inference using UPR re-ranked passages and a pre-trained FiD checkpoint leads to an improved answer generation performance.  

<a id="downloading-data-and-checkpoints"></a>
## Downloading Data and Checkpoints
We've provided pretrained checkpoints and datasets on Dropbox for use to train models for dense retrieval and open-domain QA tasks. 
This data can be downloaded here:

Required training files
- [Wikipedia Evidence Documents from DPR paper](https://www.dropbox.com/s/bezryc9win2bha1/psgs_w100.tar.gz)
- [Indexed evidence documents and titles](https://www.dropbox.com/s/nc49dkno8o3pgb3/evidence-wikipedia-indexed-mmap.tar.gz)
- [Dataset-specific question-answer pairs](https://www.dropbox.com/s/gm0y3lx1wv0uxx2/qas.tar.gz)
- [BERT-large vocabulary file](https://www.dropbox.com/s/ttblv1uggd4cijt/bert-large-uncased-vocab.txt)

Required checkpoints
- [Masked Salient Span (MSS) T5 pre-trained reader (base configuration)](https://www.dropbox.com/s/33lm2685ifpei4l/mss-emdr2-reader-base-steps82k.tar.gz)


<a id="usage"></a>
# Usage

We've provided a demo scripts for training an FiD model for open-domain QA tasks in [`examples`](./examples) directory.

Please ensure to change the data and checkpoint paths in these scripts.

To train or do inference using a pre-trained model, please see the options and run the script as
```
bash examples/fid_common.sh
```

<a id="pre-trained-checkpoints"></a>
# Pre-trained Checkpoints

Dataset | Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
--------|-----------|---------------|--------|---------|:-----------:
Natural Questions | MSS + UPR | base | 45.8 | 47.3 | [link](https://www.dropbox.com/s/2lbphzcac4w2tfp/fid-mss-nq-base-topk100-bsize64.tar.gz)
Natural Questions | DPR + UPR | base | 49.8 | 51.3 | [link](https://www.dropbox.com/s/sno9ms6zphtajxq/fid-dpr-nq-base-topk100-bsize64.tar.gz)
Natural Questions | MSS-DPR + UPR | base | 49.9 | 51.5 | [link](https://www.dropbox.com/s/fo9vgophu81yjm3/fid-mss-dpr-nq-base-topk100-bsize64.tar.gz)
Natural Questions | MSS-DPR + UPR | large | 51.5 | 54.5 | [link](https://www.dropbox.com/s/wqbyq7b96hlaahr/fid-mss-dpr-nq-large-topk100-bsize64.tar.gz) 

Dataset | Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
--------|-----------|---------------|--------|---------|:-----------:
TriviaQA | MSS + UPR | base | 68.5 | 68.9 | [link](https://www.dropbox.com/s/s92am9wcwunv1q5/fid-mss-trivia-base-topk100-bsize64.tar.gz)
TriviaQA | DPR + UPR | base | 70.1 | 71.2 | [link](https://www.dropbox.com/s/dl0tb4at6090h00/fid-dpr-trivia-base-topk100-bsize64.tar.gz)
TriviaQA | MSS-DPR + UPR | base | 71.5 | 71.8 | [link](https://www.dropbox.com/s/7qpnndn3cf0rwkp/fid-mss-dpr-trivia-base-topk100-bsize64.tar.gz)
TriviaQA | MSS-DPR + UPR | large | 72.7 | 73.2 | [link](https://www.dropbox.com/s/2xlemikbnq86yf9/fid-mss-dpr-trivia-large-topk100-bsize64.tar.gz)

Dataset | Retriever | Reader Config | Dev EM | Test EM | Checkpoint 
--------|-----------|---------------|--------|---------|:-----------:
SQuAD-Open | MSS + UPR | base | 43.7 | 50.1 | [link](https://www.dropbox.com/s/36zhaxim0lax9uv/fid-mss-squad1-base-topk100-bsize64.tar.gz)
SQuAD-Open | DPR + UPR | base | 51.5 | 54.0 | [link](https://www.dropbox.com/s/sb10urzlzs0jcmb/fid-dpr-squad1-base-topk100-bsize64.tar.gz)
SQuAD-Open | MSS-DPR + UPR | base | 51.9 | 55.6  | [link](https://www.dropbox.com/s/7a61gt3zuivxto8/fid-mss-dpr-squad1-base-topk100-bsize64.tar.gz)
SQuAD-Open | MSS-DPR + UPR | large | 53.1 | 58.1 | [link](https://www.dropbox.com/s/nx4ahlhuywdvjg8/fid-mss-dpr-squad1-large-topk100-bsize64.tar.gz)

# Credits: The design of the source code follows from the DPR download_data.py script


"""
 Command line tool to download various preprocessed data sources for UPR.
"""
import argparse
import tarfile
import os
import pathlib
from subprocess import Popen, PIPE


RESOURCES_MAP = {
    # Wikipedia
    "data.wikipedia-split.psgs_w100": {
        "dropbox_url": "https://www.dropbox.com/s/bezryc9win2bha1/psgs_w100.tar.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },

    # BM25
    "data.retriever-outputs.bm25.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/yp3zp0brotckzlz/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for WebQuestions test set.",
    },
    "data.retriever-outputs.bm25.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/ml2lnt34ktjgft6/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for Natural Questions Open test set.",
    },
    "data.retriever-outputs.bm25.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/2gx8mwj58ifxwm2/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for Natural Questions Open development set.",
    },
    "data.retriever-outputs.bm25.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/dd04rdrk85fj6kz/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for TriviaQA development set.",
    },
    "data.retriever-outputs.bm25.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/2cf4v77bay9cwnm/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for TriviaQA test set.",
    },
    "data.retriever-outputs.bm25.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/29kjd71wn1fs9ca/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for Squad1-Open test set.",
    },
    "data.retriever-outputs.bm25.entity-questions": {
        "dropbox_url": "https://www.dropbox.com/s/8qjmyhnd8wt4b7s/entity-questions.tar.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for the Entity Questions test set.",
    },

    # Contriever data
    "data.retriever-outputs.contriever.entity-questions": {
        "dropbox_url": "https://www.dropbox.com/s/jmvs0lc6u03ydbu/entity-questions.tar.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "Top-1000 passages from Contriever for the Entity Questions test set.",
    },
    "data.retriever-outputs.contriever.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/lcmaiqh3wq9tprr/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from Contriever for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.contriever.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/5bciavqx2j43a7s/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from Contriever for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.contriever.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/x9uq40ub5t0gsoz/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from Contriever for the TriviaQA test set.",
    },
    "data.retriever-outputs.contriever.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/4ldkx74pv2jo7a9/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from Contriever for the TriviaQA dev set.",
    },
    "data.retriever-outputs.contriever.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/rq3961cij8qq7v6/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from Contriever for the WebQuestions test set.",
    },
    "data.retriever-outputs.contriever.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/hkub97p14jx0uuh/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from Contriever for the Squad1 test set.",
    },

    # DPR data
    "data.retriever-outputs.dpr.entity-questions": {
        "dropbox_url": "https://www.dropbox.com/s/2ngexghb2zzjdie/entity-questions.tar.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Entity Questions test set.",
    },
    "data.retriever-outputs.dpr.nq-train": {
        "dropbox_url": "https://www.dropbox.com/s/6g4erof4ifg8xea/nq-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Natural Questions Open train set.",
    },
    "data.retriever-outputs.dpr.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/257quanu64w9sh0/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.dpr.reranked.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/osolohjruv3dw2y/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.dpr.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/c7ooi5fgy658cri/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.dpr.reranked.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/x1nxpf0uz5lapz6/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.dpr.trivia-train": {
        "dropbox_url": "https://www.dropbox.com/s/3onjkogwkc2gk4u/trivia-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the TriviaQA train set.",
    },
    "data.retriever-outputs.dpr.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/50wx42yquqvbbgx/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the TriviaQA test set.",
    },
    "data.retriever-outputs.dpr.reranked.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/s7g76bkftwinozw/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the TriviaQA test set.",
    },
    "data.retriever-outputs.dpr.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/7t7czgqmxyz1ddt/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the TriviaQA dev set.",
    },
    "data.retriever-outputs.dpr.reranked.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/zz3btm8bhaw1c7c/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the TriviaQA dev set.",
    },
    "data.retriever-outputs.dpr.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/n8v2o00231e9lkl/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the WebQuestions test set.",
    },
    "data.retriever-outputs.dpr.squad1-train": {
        "dropbox_url": "https://www.dropbox.com/s/i4loxz4k1squ3az/squad1-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Squad1-Open train set.",
    },
    "data.retriever-outputs.dpr.squad1-dev": {
        "dropbox_url": "https://www.dropbox.com/s/0r8k4cqtt61ep3e/squad1-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Squad1-Open dev set.",
    },
    "data.retriever-outputs.dpr.reranked.squad1-dev": {
        "dropbox_url": "https://www.dropbox.com/s/tbbm9s1jksw31fk/squad1-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Squad1-Open dev set.",
    },
    "data.retriever-outputs.dpr.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/91vf2nqmzfvyyx7/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Squad1-Open test set.",
    },
    "data.retriever-outputs.dpr.reranked.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/taitdxquvhqc0da/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Squad1-Open test set.",
    },

    # MSS-DPR data
    "data.retriever-outputs.mss-dpr.entity-questions": {
        "dropbox_url": "https://www.dropbox.com/s/hvtmn7sjbk1y1po/entity-questions.tar.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the Entity Questions test set.",
    },
    "data.retriever-outputs.mss-dpr.reranked.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/rbbnx0vfaz5un0z/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR + UPR (T0-3B) for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.mss-dpr.reranked.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/xxnqp97i2cb2cv0/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR + UPR (T0-3B) for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.mss-dpr.reranked.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/lf8xz4vnqqosz1t/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR + UPR (T0-3B) for the TriviaQA test set.",
    },
    "data.retriever-outputs.mss-dpr.reranked.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/t0cy9ohxwno2af0/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR + UPR (T0-3B) for the TriviaQA dev set.",
    },
    "data.retriever-outputs.mss-dpr.reranked.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/43if28wgry73xu3/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR + UPR (T0-3B) for the WebQuestions test set.",
    },
    "data.retriever-outputs.mss-dpr.reranked.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/y1q6kg44z8pm2uw/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR + UPR (T0-3B) for the Squad1 test set.",
    },
    "data.retriever-outputs.mss-dpr.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/suwr38hkldd8ly0/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.mss-dpr.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/jd42ibph352be5a/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.mss-dpr.nq-train": {
        "dropbox_url": "https://www.dropbox.com/s/nh9ee844qml7ecg/nq-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the Natural Questions Open training set.",
    },
    "data.retriever-outputs.mss-dpr.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/56ohs7w9d1bsxh8/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the TriviaQA test set.",
    },
    "data.retriever-outputs.mss-dpr.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/abslu8ib0pg5nlw/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the TriviaQA dev set.",
    },
    "data.retriever-outputs.mss-dpr.trivia-train": {
        "dropbox_url": "https://www.dropbox.com/s/bdn0npbfi2dnr49/trivia-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the TriviaQA training set.",
    },
    "data.retriever-outputs.mss-dpr.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/s2poplmjfubkcwt/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the WebQuestions test set.",
    },
    "data.retriever-outputs.mss-dpr.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/7zsds2ddqh6wsys/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the Squad1 test set.",
    },
    "data.retriever-outputs.mss-dpr.squad1-dev": {
        "dropbox_url": "https://www.dropbox.com/s/ia2804uo75qr8em/squad1-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the Squad1 dev set.",
    },
    "data.retriever-outputs.mss-dpr.squad1-train": {
        "dropbox_url": "https://www.dropbox.com/s/qf223ens44bmcxy/squad1-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS-DPR for the Squad1-Open training set.",
    },

    # MSS (Masked Salient Spans) Retriever Data
    "data.retriever-outputs.mss.entity-questions": {
        "dropbox_url": "https://www.dropbox.com/s/y9adtb3cjxbpd0x/entity-questions.tar.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the Entity Questions test set.",
    },
    "data.retriever-outputs.mss.nq-train": {
        "dropbox_url": "https://www.dropbox.com/s/1g9vmqrxacjg8u9/nq-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the Natural Questions Open train set.",
    },
    "data.retriever-outputs.mss.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/oyop7a6kgbsm9nw/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.mss.reranked.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/6a4acadq3uf7vwm/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS + UPR (T0-3B) for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.mss.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/11uviokjqkbgg1x/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.mss.reranked.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/io1pjspj1kb4h7n/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS + UPR (T0-3B) for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.mss.trivia-train": {
        "dropbox_url": "https://www.dropbox.com/s/n839yywsy1tk8ep/trivia-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the TriviaQA train set.",
    },
    "data.retriever-outputs.mss.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/y8hskth77q53pvy/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the TriviaQA test set.",
    },
    "data.retriever-outputs.mss.reranked.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/cke6ahx2gpz0ldq/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS + UPR (T0-3B) for the TriviaQA test set.",
    },
    "data.retriever-outputs.mss.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/5qhapied39k59ix/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the TriviaQA dev set.",
    },
    "data.retriever-outputs.mss.reranked.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/bj0ite79epziztx/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS + UPR (T0-3B) for the TriviaQA dev set.",
    },
    "data.retriever-outputs.mss.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/ljf3rwfo9bgdi5s/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the WebQuestions test set.",
    },
    "data.retriever-outputs.mss.squad1-train": {
        "dropbox_url": "https://www.dropbox.com/s/i5qpdvbbmi4y2zj/squad1-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the Squad1-Open train set.",
    },
    "data.retriever-outputs.mss.squad1-dev": {
        "dropbox_url": "https://www.dropbox.com/s/covkgszc7ttt1l6/squad1-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the Squad1-Open dev set.",
    },
    "data.retriever-outputs.mss.reranked.squad1-dev": {
        "dropbox_url": "https://www.dropbox.com/s/yhl5qxp19zi16tk/squad1-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS + UPR (T0-3B) for the Squad1-Open dev set.",
    },
    "data.retriever-outputs.mss.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/e7pe9d2u93sre6l/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS for the Squad1-Open test set.",
    },
    "data.retriever-outputs.mss.reranked.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/zi6p63pvcdpan4o/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from MSS + UPR (T0-3B) for the Squad1-Open test set.",
    },

    # Union of MSS-BM25 data
    "data.retriever-outputs.mss-bm25-union.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/n2s1hu6pudn0dki/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-2000 passages from the union of MSS + BM25 for the NQ-Open dev set.",
    },
}


def unpack(tar_file: str, out_path: str):
    print("Uncompressing %s", tar_file)
    input = tarfile.open(tar_file, "r:gz")
    input.extractall(out_path)
    input.close()
    print(" Saved to %s", out_path)


def download_resource(
    dropbox_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str
) -> None:
    print("Requested resource from %s", dropbox_url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    print("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, "downloads", *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    print("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        print("File already exist %s", local_file_uncompressed)
        return

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] + (".tar.gz" if compressed else original_ext)))

    process = Popen(['wget', dropbox_url, '-O', local_file], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    # print(stderr.decode("utf-8"))
    print("Downloaded to %s", local_file)

    if compressed:
        # uncompressed_path = os.path.join(save_root, path_names[-1])
        unpack(local_file, save_root)
        os.remove(local_file)
    return



def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        print("matched by prefix resources: %s", resources)
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    dropbox_url = download_info["dropbox_url"]

    if isinstance(dropbox_url, list):
        for i, url in enumerate(dropbox_url):
            download_resource(
                url,
                download_info["original_ext"],
                download_info["compressed"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
    else:
        download_resource(
            dropbox_url,
            download_info["original_ext"],
            download_info["compressed"],
            resource_key,
            out_dir,
        )
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print("Resource key=%s  :  %s", k, v["desc"])


if __name__ == "__main__":
    main()

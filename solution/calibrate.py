#!/usr/bin/env python3
"""Calibrate the nDCG@10 bars against the fixtures this task actually ships.

Authoring-side only: it uses the ONNX export of the same checkpoint the task pins, so the
numbers here are the same retrieval quality the Rust crate reaches, without needing a Rust
toolchain to re-derive a bar. Point MINILM_DIR at a directory holding model.onnx and
tokenizer.json for sentence-transformers/all-MiniLM-L6-v2 at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41.

Arms, in the order an implementer is likely to reach for them:

  trunc256   embed each issue once, truncated at the checkpoint's 256-token limit
  win510     fixed 510-token windows, score = best window
  win254     fixed 254-token windows, score = best window   (the common chunk size)
  win126     126-token windows, stride 63, score = best window
  item_max   split each issue on its own item boundaries, score = best item
  item_mean  split on item boundaries, score = mean over items
  bm25       lexical baseline, no model at all
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import Embedder

root = Path(__file__).resolve().parent.parent
CHUNK_ARMS = [("win510", 510, 510), ("win254", 254, 254), ("win126", 126, 63)]


def ndcg(scores, rel, k=10):
    order = list(np.argsort(-scores)[:k])
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(order) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / idcg if idcg else 0.0


def main():
    files = sorted((root / "environment" / "corpus").glob("issue_*.md"))
    ids = [f.stem for f in files]
    index = {name: i for i, name in enumerate(ids)}
    docs = [f.read_text(encoding="utf-8") for f in files]
    N = len(docs)

    e = Embedder()
    doc_tokens = [e.token_ids(d) for d in docs]

    def windows(size, stride):
        seqs, owner = [], []
        for i, ids_ in enumerate(doc_tokens):
            for s in range(0, max(1, len(ids_)), stride):
                seqs.append(ids_[s:s + size])
                owner.append(i)
                if s + size >= len(ids_):
                    break
        return seqs, np.array(owner)

    items, item_owner = [], []
    for i, d in enumerate(docs):
        for part in d.split("\n\n")[1:]:
            items.append(e.token_ids(part))
            item_owner.append(i)
    item_owner = np.array(item_owner)

    print(f"issues={N} items={len(items)} median_tokens="
          f"{int(np.median([len(t) for t in doc_tokens]))}", flush=True)

    mats = {"trunc256": (e.embed_ids(doc_tokens, batch=64, cap=256), np.arange(N)),
            "item_max": (e.embed_ids(items, batch=128), item_owner)}
    for name, size, stride in CHUNK_ARMS:
        seqs, owner = windows(size, stride)
        mats[name] = (e.embed_ids(seqs, batch=16 if size > 300 else 64, cap=size + 2), owner)

    mean = np.zeros((N, 384), dtype=np.float32)
    for i in range(N):
        v = mats["item_max"][0][item_owner == i].mean(0)
        mean[i] = v / max(np.linalg.norm(v), 1e-12)

    words = [re.sub(r"[^a-z ]", " ", d.lower()).split() for d in docs]
    df = Counter()
    for w in words:
        df.update(set(w))
    idf = {w: math.log(1 + (N - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    tf = [Counter(w) for w in words]
    dl = np.array([len(w) for w in words], dtype=float)
    avgdl = dl.mean()

    # BM25 over the items rather than the issues, an issue scored by its best item. This is the
    # lexical route that matters: over whole issues BM25 is diluted by the other items, over
    # single items it is not, and a query that shares its wording with its item is found.
    item_texts, item_owner = [], []
    for i, d in enumerate(docs):
        for it in d.split("\n\n")[1:]:
            if it.strip():
                item_texts.append(re.sub(r"[^a-z ]", " ", it.lower()).split())
                item_owner.append(i)
    item_tf = [Counter(t) for t in item_texts]
    item_dl = np.array([len(t) for t in item_texts], dtype=float)
    item_df = Counter()
    for t in item_texts:
        item_df.update(set(t))
    item_avgdl = item_dl.mean()

    def bm25_item(query, k1=1.2, b=0.75):
        out = np.full(N, -1e9)
        M = len(item_texts)
        for i, counts in enumerate(item_tf):
            total = 0.0
            for w in query:
                f = counts.get(w, 0)
                if f:
                    d = item_df.get(w, 0)
                    idf_w = np.log((M - d + 0.5) / (d + 0.5) + 1.0)
                    total += idf_w * f * (k1 + 1) / (f + k1 * (1 - b + b * item_dl[i] / item_avgdl))
            out[item_owner[i]] = max(out[item_owner[i]], total)
        return out

    def bm25(query, k1=1.5, b=0.75):
        out = np.zeros(N)
        for i, counts in enumerate(tf):
            total = 0.0
            for w in query:
                f = counts.get(w, 0)
                if f:
                    total += idf.get(w, 0) * f * (k1 + 1) / (
                        f + k1 * (1 - b + b * dl[i] / avgdl))
            out[i] = total
        return out

    sets = {
        "dev": (root / "environment" / "queries" / "dev.json", None),
        "graded": (root / "environment" / "queries" / "graded.json",
                   root / "tests" / "golden" / "graded_labels.json"),
        "hidden": (root / "tests" / "hidden" / "queries.json",
                   root / "tests" / "hidden" / "labels.json"),
    }
    names = ["trunc256", "win510", "win254", "win126", "item_max", "item_mean", "bm25", "bm25_item"]
    for split, (qp, lp) in sets.items():
        rows = json.loads(qp.read_text())
        labels = ({r["query_id"]: r["issues"] for r in rows} if lp is None
                  else json.loads(lp.read_text()))
        Q = e.embed_texts([r["query"] for r in rows])
        got = defaultdict(list)
        for j, r in enumerate(rows):
            q = Q[j]
            rel = {index[x] for x in labels[r["query_id"]]}

            def best(pair, q=q):
                mat, owner = pair
                out = np.full(N, -1e9)
                for oi, sc in zip(owner, mat @ q):
                    out[oi] = max(out[oi], sc)
                return out

            scores = {n: best(mats[n]) for n in mats}
            scores["item_mean"] = mean @ q
            scores["bm25"] = bm25(re.sub(r"[^a-z ]", " ", r["query"].lower()).split())
            scores["bm25_item"] = bm25_item(re.sub(r"[^a-z ]", " ", r["query"].lower()).split())
            for n in names:
                got[n].append(ndcg(scores[n], rel))
        print(f"  {split:7s} n={len(rows):3d}  " +
              "  ".join(f"{n}={np.mean(got[n]):.3f}" for n in names), flush=True)


if __name__ == "__main__":
    main()

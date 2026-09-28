# main/retrieval/run_bm25_grouped_frag_pool.py
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from .simplemono_rel import load_decoding_cfg, infer_used_len_events, events_to_rel_tokens
from .ngram_bm25 import ngram_hashes_uint64, NGramBM25Index, BM25Params


def load_queries_meta(meta_jsonl: str) -> List[Dict[str, Any]]:
    out = []
    with Path(meta_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    out.sort(key=lambda x: int(x["query_id"]))
    return out


def _norm_factor(mode: str, *, n_notes: int, n_terms: int) -> float:
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return 1.0
    if mode == "notes":
        return 1.0 / float(max(1, n_notes))
    if mode == "terms":
        return 1.0 / float(max(1, n_terms))
    if mode == "sqrt_terms":
        return 1.0 / math.sqrt(float(max(1, n_terms)))
    raise ValueError(f"Unknown score_norm: {mode}")


class _MaxPool:
    def __init__(self):
        self.best: Dict[int, float] = {}

    def update(self, doc_id: int, score: float):
        cur = self.best.get(doc_id)
        if cur is None or score > cur:
            self.best[doc_id] = float(score)

    def finalize(self) -> Dict[int, float]:
        return self.best


class _LSEPool:
    """
    pooled(doc) = tau * logsumexp_j(score_j / tau)
    """
    def __init__(self, tau: float):
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError("lse_tau must be > 0")
        # doc -> (m, s) in y-space where y=score/tau
        self.state: Dict[int, Tuple[float, float]] = {}

    def update(self, doc_id: int, score: float):
        y = float(score) / self.tau
        st = self.state.get(doc_id)
        if st is None:
            self.state[doc_id] = (y, 1.0)
            return
        m, s = st
        if y <= m:
            s = s + math.exp(y - m)
            self.state[doc_id] = (m, s)
        else:
            # new max
            s = s * math.exp(m - y) + 1.0
            self.state[doc_id] = (y, s)

    def finalize(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for doc_id, (m, s) in self.state.items():
            out[doc_id] = self.tau * (m + math.log(max(s, 1e-30)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_npy", type=str, required=True)
    ap.add_argument("--frags_npy", type=str, required=True)
    ap.add_argument("--frag_offsets_npy", type=str, required=True)
    ap.add_argument("--queries_meta_jsonl", type=str, required=True)
    ap.add_argument("--simplemono_pkl", type=str, required=True)

    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--topM", type=int, default=200)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--hash_base", type=int, default=1315423911)

    ap.add_argument("--pooling", type=str, default="max", choices=["max", "lse"])
    ap.add_argument("--lse_tau", type=float, default=2.0)

    ap.add_argument("--score_norm", type=str, default="none", choices=["none", "notes", "terms", "sqrt_terms"],
                    help="Normalization to make scores comparable across different fragment lengths.")

    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--max_queries", type=int, default=0)
    args = ap.parse_args()

    cfg_dec = load_decoding_cfg(args.simplemono_pkl)

    docs = np.load(args.docs_npy, mmap_mode="r")
    frags = np.load(args.frags_npy, mmap_mode="r")
    offsets = np.load(args.frag_offsets_npy)

    if docs.ndim != 3 or docs.shape[-1] != 3:
        raise ValueError(f"Bad docs shape: {docs.shape}")
    if frags.ndim != 3 or frags.shape[-1] != 3:
        raise ValueError(f"Bad frags shape: {frags.shape}")
    if offsets.ndim != 1:
        raise ValueError(f"Bad offsets shape: {offsets.shape}")

    qmeta = load_queries_meta(args.queries_meta_jsonl)

    N_docs = int(docs.shape[0])
    if int(args.max_docs) > 0:
        N_docs = min(N_docs, int(args.max_docs))

    N_q = int(offsets.shape[0]) - 1
    N_q = min(N_q, len(qmeta))
    if int(args.max_queries) > 0:
        N_q = min(N_q, int(args.max_queries))

    # ---------- build index ----------
    n = int(args.n)
    idx = NGramBM25Index(
        n=n,
        bm25=BM25Params(k1=float(args.k1), b=float(args.b)),
        hash_base=int(args.hash_base),
        verbose=True,
    )

    def iter_doc_terms():
        for doc_id in tqdm(range(N_docs), desc="[Index] building", dynamic_ncols=True):
            ev = np.asarray(docs[doc_id])
            used = infer_used_len_events(ev, cfg_dec)
            rel = events_to_rel_tokens(ev, cfg_dec, used_len_events=used)
            terms = ngram_hashes_uint64(rel, n, base=int(args.hash_base))
            yield doc_id, terms

    t0 = time.time()
    idx.build(n_docs=N_docs, iter_doc_terms=iter_doc_terms())
    t1 = time.time()
    print(f"[Index] build time: {t1 - t0:.2f}s")

    # ---------- retrieval + eval ----------
    Ks = [1, 5, 10, 20]
    Ks = [k for k in Ks if k <= int(args.topM)]
    hit = {k: 0 for k in Ks}
    mrr_sum = 0.0
    miss_empty = 0

    t_q0 = time.time()

    pbar = tqdm(range(N_q), desc="[Query] grouped frag bm25", dynamic_ncols=True)
    for qid in pbar:
        st = int(offsets[qid])
        ed = int(offsets[qid + 1])
        if ed <= st:
            miss_empty += 1
            continue

        tgt = int(qmeta[qid]["target_doc_id"])

        pool = _MaxPool() if args.pooling == "max" else _LSEPool(tau=float(args.lse_tau))

        for fi in range(st, ed):
            evq = np.asarray(frags[fi])
            used_q = infer_used_len_events(evq, cfg_dec)
            n_notes = max(0, int(used_q) - 2)

            relq = events_to_rel_tokens(evq, cfg_dec, used_len_events=used_q)
            q_terms = ngram_hashes_uint64(relq, n, base=int(args.hash_base))

            nf = _norm_factor(args.score_norm, n_notes=n_notes, n_terms=int(q_terms.size))

            res = idx.query(q_terms, topM=int(args.topM))
            for d, s in res:
                pool.update(int(d), float(s) * nf)

        agg_scores = pool.finalize()
        if not agg_scores:
            continue

        if tgt not in agg_scores:
            rank = None
        else:
            items = sorted(agg_scores.items(), key=lambda x: -x[1])
            rank = None
            for r, (d, _) in enumerate(items, start=1):
                if int(d) == tgt:
                    rank = r
                    break

        if rank is not None:
            mrr_sum += 1.0 / float(rank)
            for k in Ks:
                if rank <= k:
                    hit[k] += 1

        pbar.set_postfix(mrr=f"{mrr_sum/max(qid+1,1):.3f}")

    t_q1 = time.time()

    print("\n========== Results (Grouped Fragment Pooling) ==========")
    for k in Ks:
        print(f"Recall@{k}: {hit[k] / float(max(N_q,1)):.4f}  ({hit[k]}/{N_q})")
    print(f"MRR: {mrr_sum / float(max(N_q,1)):.4f}")
    print(f"Empty queries (0 fragments): {miss_empty}")
    print(f"[Query] time: {t_q1 - t_q0:.5f}s | per-query: {(t_q1 - t_q0)/max(N_q,1):.7f}s")
    print("=======================================================")


if __name__ == "__main__":
    main()
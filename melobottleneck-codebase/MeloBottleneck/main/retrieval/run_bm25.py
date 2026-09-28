# main/retrieval/run_bm25.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .simplemono_rel import (
    load_decoding_cfg,
    infer_used_len_events,
    events_to_rel_tokens,
    RelTokenConfig,
)
from .ngram_bm25 import ngram_hashes_uint64, NGramBM25Index, BM25Params


def load_queries_meta(meta_jsonl: str) -> List[Dict]:
    out = []
    with Path(meta_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    out.sort(key=lambda x: int(x["query_id"]))
    return out


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_npy", type=str, required=True)
    ap.add_argument("--queries_npy", type=str, required=True)
    ap.add_argument("--queries_meta_jsonl", type=str, required=True)
    ap.add_argument("--simplemono_pkl", type=str, required=True)

    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--topM", type=int, default=100)

    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--hash_base", type=int, default=1315423911)

    ap.add_argument("--max_docs", type=int, default=0, help="0 => use all")
    ap.add_argument("--max_queries", type=int, default=0, help="0 => use all")

    ap.add_argument(
        "--rel_mode",
        type=str,
        default="dp_dur_ratio",
        choices=["dp_dur_ratio", "dp_only", "dp_coarse_dur"],
        help=(
            "Retrieval representation mode. "
            "dp_dur_ratio: old behavior; "
            "dp_only: pitch interval only; "
            "dp_coarse_dur: dp + coarse duration bins."
        ),
    )
    ap.add_argument(
        "--coarse_dur_bin_edges",
        type=str,
        default="1,2,3,4,6,8,12,16,24,32,48,64",
        help="Comma-separated duration upper bounds (in pos units) for rel_mode=dp_coarse_dur.",
    )

    args = ap.parse_args()

    cfg_dec = load_decoding_cfg(args.simplemono_pkl)

    docs = np.load(args.docs_npy, mmap_mode="r")
    queries = np.load(args.queries_npy, mmap_mode="r")
    if docs.ndim != 3 or docs.shape[-1] != 3:
        raise ValueError(f"Bad docs shape: {docs.shape}")
    if queries.ndim != 3 or queries.shape[-1] != 3:
        raise ValueError(f"Bad queries shape: {queries.shape}")

    N_docs = int(docs.shape[0])
    N_q = int(queries.shape[0])

    if args.max_docs and args.max_docs > 0:
        N_docs = min(N_docs, int(args.max_docs))
    if args.max_queries and args.max_queries > 0:
        N_q = min(N_q, int(args.max_queries))

    qmeta = load_queries_meta(args.queries_meta_jsonl)
    if len(qmeta) < N_q:
        N_q = len(qmeta)

    coarse_edges = tuple(_parse_int_list(args.coarse_dur_bin_edges))
    if str(args.rel_mode) == "dp_coarse_dur" and len(coarse_edges) == 0:
        raise ValueError("--coarse_dur_bin_edges must not be empty when --rel_mode=dp_coarse_dur")

    rel_cfg = RelTokenConfig(
        mode=str(args.rel_mode),
        coarse_dur_bin_edges=coarse_edges if len(coarse_edges) > 0 else RelTokenConfig().coarse_dur_bin_edges,
    )

    if rel_cfg.mode == "dp_coarse_dur":
        print(f"[Rel] mode={rel_cfg.mode}, coarse_dur_bin_edges={list(rel_cfg.coarse_dur_bin_edges)}")
    else:
        print(f"[Rel] mode={rel_cfg.mode}")

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
            rel = events_to_rel_tokens(ev, cfg_dec, used_len_events=used, rel_cfg=rel_cfg)
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

    t_q0 = time.time()

    empty = 0

    pbar = tqdm(range(N_q), desc="[Query] bm25", dynamic_ncols=True)
    for qi in pbar:
        evq = np.asarray(queries[qi])
        used_q = infer_used_len_events(evq, cfg_dec)
        relq = events_to_rel_tokens(evq, cfg_dec, used_len_events=used_q, rel_cfg=rel_cfg)
        q_terms = ngram_hashes_uint64(relq, n, base=int(args.hash_base))

        if q_terms.size == 0:
            empty += 1
            continue

        res = idx.query(q_terms, topM=int(args.topM))
        ranked_docs = [d for d, _ in res]

        tgt = int(qmeta[qi]["target_doc_id"])
        rank = None
        for r, d in enumerate(ranked_docs, start=1):
            if int(d) == tgt:
                rank = r
                break

        if rank is not None:
            mrr_sum += 1.0 / float(rank)
            for k in Ks:
                if rank <= k:
                    hit[k] += 1

        pbar.set_postfix(mrr=f"{mrr_sum/max(qi+1,1):.3f}")

    t_q1 = time.time()

    print("\n========== Results ==========")
    for k in Ks:
        print(f"Recall@{k}: {hit[k] / float(N_q):.4f}  ({hit[k]}/{N_q})")
    print(f"MRR: {mrr_sum / float(N_q):.4f}")
    print(f"[Query] time: {t_q1 - t_q0:.5f}s | per-query: {(t_q1 - t_q0)/max(N_q,1):.7f}s")
    print(f"empty queries: {empty}")
    print("================================")


if __name__ == "__main__":
    main()
    
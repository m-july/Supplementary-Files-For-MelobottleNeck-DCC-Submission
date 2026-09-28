# main/retrieval/run_random_rank_baseline.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def load_queries_meta(meta_jsonl: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(meta_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    out.sort(key=lambda x: int(x.get("query_id", 0)))
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


def _harmonic(n: int) -> float:
    n = int(n)
    if n <= 0:
        return 0.0
    a = np.arange(1, n + 1, dtype=np.float64)
    return float((1.0 / a).sum())


def _harmonic2(n: int) -> float:
    n = int(n)
    if n <= 0:
        return 0.0
    a = np.arange(1, n + 1, dtype=np.float64)
    return float((1.0 / (a * a)).sum())


def _fmt(x: float) -> str:
    x = float(x)
    if abs(x) < 1e-4:
        return f"{x:.3e}"
    return f"{x:.6f}"


def analytic_random_baseline(
    *,
    N_docs: int,
    N_q: int,
    n_valid: int,
    Ks: List[int],
    topM: int,
) -> Tuple[Dict[int, Tuple[float, float]], Tuple[float, float], Dict[str, float]]:
    """
    Returns:
      - recall_stats[k] = (mean, std_of_mean_over_queries_approx)
      - mrr_stats = (mean, std_of_mean_over_queries_approx)
      - extra = dict(...) (harmonic numbers etc.)
    """
    N_docs = int(N_docs)
    N_q = int(N_q)
    n_valid = int(n_valid)
    topM_eff = min(int(topM), N_docs)

    if N_docs <= 0 or N_q <= 0:
        recall_stats = {int(k): (0.0, 0.0) for k in Ks}
        return recall_stats, (0.0, 0.0), {"topM_eff": float(topM_eff), "valid_ratio": 0.0}

    # valid queries contribute; invalid targets always miss
    valid_ratio = n_valid / float(N_q)

    recall_stats: Dict[int, Tuple[float, float]] = {}
    for k in Ks:
        k = int(k)
        p = k / float(N_docs)  # P(rank<=k) for a valid query
        mean = valid_ratio * p

        # hits_count ~ Binomial(n_valid, p), recall = hits_count / N_q
        var_recall = (n_valid * p * (1.0 - p)) / (float(N_q) ** 2)
        std = math.sqrt(max(var_recall, 0.0))
        recall_stats[k] = (mean, std)

    # MRR (truncated at topM_eff)
    H = _harmonic(topM_eff)
    H2 = _harmonic2(topM_eff)

    mu_valid = H / float(N_docs)
    ex2_valid = H2 / float(N_docs)
    var_valid = max(ex2_valid - mu_valid * mu_valid, 0.0)

    mean_mrr = valid_ratio * mu_valid
    var_mrr = (n_valid * var_valid) / (float(N_q) ** 2)
    std_mrr = math.sqrt(max(var_mrr, 0.0))

    extra = {
        "topM_eff": float(topM_eff),
        "H_topM": float(H),
        "H2_topM": float(H2),
        "valid_ratio": float(valid_ratio),
    }
    return recall_stats, (mean_mrr, std_mrr), extra


def mc_random_baseline(
    *,
    N_docs: int,
    N_q: int,
    n_valid: int,
    Ks: List[int],
    topM: int,
    trials: int,
    seed: int,
    chunk_trials: int = 1024,
) -> Tuple[Dict[int, Tuple[float, float]], Tuple[float, float], Dict[str, float]]:
    """
    Monte-Carlo:
      sample ranks ~ Uniform{1..N_docs} for each valid query and each trial.
    Output: mean/std over trials (NOT std of mean over queries).
    """
    N_docs = int(N_docs)
    N_q = int(N_q)
    n_valid = int(n_valid)
    topM_eff = min(int(topM), N_docs)
    trials = int(trials)

    if N_docs <= 0 or N_q <= 0 or trials <= 0:
        recall_stats = {int(k): (0.0, 0.0) for k in Ks}
        return recall_stats, (0.0, 0.0), {"topM_eff": float(topM_eff), "valid_ratio": 0.0}

    rng = np.random.default_rng(int(seed))

    Ks = [int(k) for k in Ks]
    sum_rec = np.zeros((len(Ks),), dtype=np.float64)
    sum2_rec = np.zeros((len(Ks),), dtype=np.float64)
    sum_mrr = 0.0
    sum2_mrr = 0.0

    done = 0
    while done < trials:
        t = min(int(chunk_trials), trials - done)
        # ranks: [t, n_valid] in [1..N_docs]
        ranks = rng.integers(1, N_docs + 1, size=(t, n_valid), dtype=np.int32)

        # recall@k per trial
        for i, k in enumerate(Ks):
            rec = (ranks <= k).sum(axis=1, dtype=np.int64).astype(np.float64) / float(N_q)  # [t]
            sum_rec[i] += rec.sum()
            sum2_rec[i] += (rec * rec).sum()

        # mrr per trial
        rr = np.where(ranks <= topM_eff, 1.0 / ranks.astype(np.float64), 0.0).sum(axis=1) / float(N_q)  # [t]
        sum_mrr += float(rr.sum())
        sum2_mrr += float((rr * rr).sum())

        done += t

    mean_rec = sum_rec / float(trials)
    var_rec = np.maximum(sum2_rec / float(trials) - mean_rec * mean_rec, 0.0)
    std_rec = np.sqrt(var_rec)

    mean_mrr = sum_mrr / float(trials)
    var_mrr = max(sum2_mrr / float(trials) - mean_mrr * mean_mrr, 0.0)
    std_mrr = math.sqrt(var_mrr)

    recall_stats = {int(k): (float(mean_rec[i]), float(std_rec[i])) for i, k in enumerate(Ks)}
    extra = {
        "topM_eff": float(topM_eff),
        "valid_ratio": float(n_valid / float(N_q)),
        "trials": float(trials),
    }
    return recall_stats, (float(mean_mrr), float(std_mrr)), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_npy", type=str, required=True)
    ap.add_argument("--queries_meta_jsonl", type=str, required=True)
    ap.add_argument("--queries_npy", type=str, default=None, help="Optional: to clamp N_q to queries.npy length")
    ap.add_argument("--topM", type=int, default=400)
    ap.add_argument("--ks", type=str, default="1,5,10,20")

    ap.add_argument("--max_docs", type=int, default=0, help="0 => use all")
    ap.add_argument("--max_queries", type=int, default=0, help="0 => use all")

    ap.add_argument("--mc_trials", type=int, default=0, help="0 => skip Monte-Carlo; e.g. 10000")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mc_chunk_trials", type=int, default=1024)

    args = ap.parse_args()

    docs = np.load(args.docs_npy, mmap_mode="r")
    if docs.ndim != 3 or docs.shape[-1] != 3:
        raise ValueError(f"Bad docs shape: {docs.shape}")
    N_docs = int(docs.shape[0])
    if int(args.max_docs) > 0:
        N_docs = min(N_docs, int(args.max_docs))

    qmeta = load_queries_meta(args.queries_meta_jsonl)

    N_q = len(qmeta)
    if args.queries_npy:
        qarr = np.load(args.queries_npy, mmap_mode="r")
        N_q = min(N_q, int(qarr.shape[0]))

    if int(args.max_queries) > 0:
        N_q = min(N_q, int(args.max_queries))

    qmeta = qmeta[:N_q]

    # count valid targets
    n_valid = 0
    for m in qmeta:
        tgt = int(m["target_doc_id"])
        if 0 <= tgt < N_docs:
            n_valid += 1

    Ks = _parse_int_list(args.ks)
    if not Ks:
        Ks = [1, 5, 10, 20]
    topM = int(args.topM)

    # match run_bm25 behavior: only report Ks <= topM
    Ks = sorted(set(int(k) for k in Ks if int(k) > 0 and int(k) <= topM))
    if not Ks:
        raise ValueError(f"All Ks are > topM={topM}. Please increase topM or change --ks.")

    print("\n========== Random Ranking Baseline ==========")
    print(f"N_docs: {N_docs}")
    print(f"N_queries: {N_q}")
    print(f"valid_targets: {n_valid}  (ratio={n_valid/max(N_q,1):.4f})")
    print(f"topM: {topM}  (effective topM=min(topM,N_docs)={min(topM, N_docs)})")
    print(f"Ks: {Ks}")

    # -------- analytic (exact expectation) --------
    rec_a, (mrr_a, mrr_a_std), extra_a = analytic_random_baseline(
        N_docs=N_docs, N_q=N_q, n_valid=n_valid, Ks=Ks, topM=topM
    )
    print("\n--- Analytic (exact expectation; std is std-of-mean over queries approx) ---")
    for k in Ks:
        mean, std = rec_a[k]
        lo = max(mean - 1.96 * std, 0.0)
        hi = min(mean + 1.96 * std, 1.0)
        print(f"Recall@{k}: {_fmt(mean)}   (approx 95% CI [{_fmt(lo)}, {_fmt(hi)}])")
    lo = max(mrr_a - 1.96 * mrr_a_std, 0.0)
    hi = min(mrr_a + 1.96 * mrr_a_std, 1.0)
    print(f"MRR: {_fmt(mrr_a)}   (approx 95% CI [{_fmt(lo)}, {_fmt(hi)}])")

    # -------- MC (optional) --------
    if int(args.mc_trials) > 0:
        rec_m, (mrr_m, mrr_m_std), extra_m = mc_random_baseline(
            N_docs=N_docs,
            N_q=N_q,
            n_valid=n_valid,
            Ks=Ks,
            topM=topM,
            trials=int(args.mc_trials),
            seed=int(args.seed),
            chunk_trials=int(args.mc_chunk_trials),
        )
        print("\n--- Monte-Carlo (mean/std over trials) ---")
        for k in Ks:
            mean, std = rec_m[k]
            print(f"Recall@{k}: {_fmt(mean)}   (std_over_trials={_fmt(std)})")
        print(f"MRR: {_fmt(mrr_m)}   (std_over_trials={_fmt(mrr_m_std)})")

    print("===========================================")


if __name__ == "__main__":
    main()
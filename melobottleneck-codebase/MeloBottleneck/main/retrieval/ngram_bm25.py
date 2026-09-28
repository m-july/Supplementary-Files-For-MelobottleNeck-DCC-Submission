# main/retrieval/ngram_bm25.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ngram_hashes_uint64(seq_u32: np.ndarray, n: int, *, base: int = 1315423911) -> np.ndarray:
    """
    seq_u32: [T] uint32
    return: [T-n+1] uint64
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    s = np.asarray(seq_u32, dtype=np.uint64)
    if s.size < n:
        return np.zeros((0,), dtype=np.uint64)

    w = sliding_window_view(s, n)  # [M,n]

    # powers[i] = base^(n-1-i) (mod 2^64 via uint64 overflow)
    b = np.uint64(base)
    powers = np.empty((n,), dtype=np.uint64)
    powers[-1] = np.uint64(1)
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * b

    h = (w * powers).sum(axis=1, dtype=np.uint64)  # overflow is ok
    return h.astype(np.uint64, copy=False)


@dataclass
class BM25Params:
    k1: float = 1.2
    b: float = 0.75


class NGramBM25Index:
    """
    term = hashed n-gram (uint64)
    postings[term] = list[(doc_id, tf_in_doc)]
    """
    def __init__(
        self,
        *,
        n: int = 5,
        bm25: BM25Params = BM25Params(),
        hash_base: int = 1315423911,
        verbose: bool = True,
    ):
        self.n = int(n)
        self.bm25 = bm25
        self.hash_base = int(hash_base)
        self.verbose = bool(verbose)

        self.N: int = 0
        self.doc_len: Optional[np.ndarray] = None
        self.avgdl: float = 0.0

        self.postings: Dict[int, List[Tuple[int, int]]] = {}   # term -> [(doc,tf)]
        self.idf: Dict[int, float] = {}                        # term -> idf

    def build(
        self,
        *,
        n_docs: int,
        iter_doc_terms: Iterable[Tuple[int, np.ndarray]],
    ) -> None:
        """
        iter_doc_terms yields (doc_id, term_occurrences_uint64)
        where term_occurrences may contain duplicates (one per occurrence).
        """
        self.N = int(n_docs)
        self.doc_len = np.zeros((self.N,), dtype=np.int32)

        # build postings
        for doc_id, terms in iter_doc_terms:
            doc_id = int(doc_id)
            t = np.asarray(terms, dtype=np.uint64)
            self.doc_len[doc_id] = int(t.size)

            if t.size == 0:
                continue

            uniq, cnt = np.unique(t, return_counts=True)
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                # u: python int, c: python int
                self.postings.setdefault(int(u), []).append((doc_id, int(c)))

        # compute avgdl
        dl = self.doc_len.astype(np.float64)
        self.avgdl = float(dl.mean()) if dl.size > 0 else 0.0
        if self.avgdl <= 0:
            self.avgdl = 1.0

        # compute idf
        N = float(self.N)
        self.idf = {}
        for term, plist in self.postings.items():
            df = float(len(plist))
            # BM25+ style idf to keep non-negative
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[int(term)] = float(idf)

        if self.verbose:
            print(f"[BM25Index] built: N={self.N}, avgdl={self.avgdl:.2f}, "
                  f"unique_terms={len(self.postings)}")

    def query(
        self,
        q_terms_occ: np.ndarray,
        *,
        topM: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        q_terms_occ: query term occurrences (uint64, may contain duplicates)
        return: list[(doc_id, score)] sorted desc
        """
        topM = int(topM)
        if topM <= 0:
            return []

        q = np.asarray(q_terms_occ, dtype=np.uint64)
        if q.size == 0:
            return []

        q_uniq, q_cnt = np.unique(q, return_counts=True)
        q_uniq = q_uniq.tolist()
        q_cnt = q_cnt.tolist()

        k1 = float(self.bm25.k1)
        b = float(self.bm25.b)
        avgdl = float(self.avgdl)

        scores: Dict[int, float] = {}

        for term_u64, qf in zip(q_uniq, q_cnt):
            term = int(term_u64)
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = float(self.idf.get(term, 0.0))
            qf = float(qf)

            for doc_id, tf in plist:
                dl = float(self.doc_len[doc_id])
                tf = float(tf)
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                s = idf * (tf * (k1 + 1.0) / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + s * qf

        if not scores:
            return []

        # topM
        # 小技巧：先转 list 再 partial sort；这里用 argsort 简单实现
        doc_ids = np.fromiter(scores.keys(), dtype=np.int32)
        sc = np.fromiter(scores.values(), dtype=np.float64)
        if sc.size <= topM:
            order = np.argsort(-sc)
        else:
            # argpartition for topM then sort those
            idx = np.argpartition(-sc, topM - 1)[:topM]
            order = idx[np.argsort(-sc[idx])]

        out = [(int(doc_ids[i]), float(sc[i])) for i in order]
        return out
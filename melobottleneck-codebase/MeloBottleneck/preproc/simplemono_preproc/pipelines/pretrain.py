# simplemono_preproc/pipelines/pretrain.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import os
import multiprocessing as mp

import numpy as np

from ..constants import DEFAULT_SPLIT, WINDOW_MAX_NOTES, POS_RESOLUTION
from ..encoder import (
    encode_triples_to_events,
    iter_piece_triple_windows,
    pad_or_truncate_events,
    triples_to_token_line,
)
from ..npy_stream import NpyAppendWriter
from ..score_readers import iter_score_files
from ..splitting import split_files
from ..utils import ensure_dir_empty
from ..vocab import SimpleMonoVocab

from tqdm import tqdm

from preproc.dataset_stats import (
    DEFAULT_STATS_BPM,
    SeqExportStatsAccumulator,
    compute_span_pos_from_triples,
    write_json,
    RunningIntStats
)


def default_max_len_tokens() -> int:
    """
    Your old pipeline (window=2048 triples) => max tokens, e.g.:
      WINDOW_MAX_NOTES=2048 : BOS(3) + 2048*3 + EOS(3) = 6150
      WINDOW_MAX_NOTES=512 : BOS(3) + 512*3 + EOS(3) = 1542
    """
    return 3 * (WINDOW_MAX_NOTES + 2)


# ----------------------------
# multiprocessing worker state
# ----------------------------
_W_INPUT_DIR: Optional[Path] = None
_W_SEED: int = 0
_W_XML_GROUP: str = "staff_voice"
_W_MAX_EVENTS: int = 0
_W_WRITE_TXT: bool = False
_W_DO_COUNTS: bool = False
_W_VOCAB: Optional[SimpleMonoVocab] = None
_W_MAX_WINDOWS_PER_FILE: int = 0
_W_MAX_WINDOWS_PER_VOICE: int = 0


def _init_pretrain_worker(
    input_dir: str,
    seed: int,
    xml_group: str,
    max_events: int,
    write_txt_debug: bool,
    do_counts: bool,
    max_windows_per_file: int,      # NEW
    max_windows_per_voice: int,     # NEW
):
    # 注意：spawn 下每个子进程会重新 import 模块，然后跑 initializer
    global _W_INPUT_DIR, _W_SEED, _W_XML_GROUP, _W_MAX_EVENTS, _W_WRITE_TXT, _W_DO_COUNTS, _W_VOCAB
    global _W_MAX_WINDOWS_PER_FILE, _W_MAX_WINDOWS_PER_VOICE
    _W_INPUT_DIR = Path(input_dir)
    _W_SEED = int(seed)
    _W_XML_GROUP = str(xml_group)
    _W_MAX_EVENTS = int(max_events)
    _W_WRITE_TXT = bool(write_txt_debug)
    _W_DO_COUNTS = bool(do_counts)
    _W_VOCAB = SimpleMonoVocab.build()
    _W_MAX_WINDOWS_PER_FILE = int(max_windows_per_file)
    _W_MAX_WINDOWS_PER_VOICE = int(max_windows_per_voice)


def _process_piece_task(task: tuple[str, str]) -> dict[str, Any]:
    """
    task: (split, piece_path_str)

    Returns dict:
      {
        "split": str,
        "piece_path": str,
        "error": Optional[str],
        "batch": Optional[np.ndarray],   # (n, max_events, 3) int32
        "meta": List[dict],              # length n
        "txt": List[str],                # length n (optional)
        "counts": Optional[np.ndarray],  # (vocab_size,) int64 (optional)
      }
    """
    from preproc.dataset_stats import compute_span_pos_from_triples  # local import ok

    note_budget = int(_W_MAX_EVENTS - 2)
    _seq_n = 0
    _notes_sum = 0
    _len_sum = 0
    _span_sum = 0
    _len_min = None
    _len_max = None
    _notes_min = None
    _notes_max = None
    _span_min = None
    _span_max = None


    sp, piece_path_s = task
    piece_path = Path(piece_path_s)

    vocab = _W_VOCAB
    input_dir = _W_INPUT_DIR
    max_events = _W_MAX_EVENTS

    if vocab is None or input_dir is None:
        return {
            "split": sp,
            "piece_path": piece_path_s,
            "error": "Worker not initialized (_W_VOCAB/_W_INPUT_DIR is None).",
            "batch": None,
            "meta": [],
            "txt": [],
            "counts": None,
            "export_stats": {
                "sequences": int(_seq_n),  # 这个 piece 产出的序列条数
                "len_events": {"count": int(_seq_n), "sum": int(_len_sum), "min": _len_min, "max": _len_max},
                "notes":      {"count": int(_seq_n), "sum": int(_notes_sum), "min": _notes_min, "max": _notes_max},
                "span_pos":   {"count": int(_seq_n), "sum": int(_span_sum), "min": _span_min, "max": _span_max},
            },
        }

    batch_rows: List[np.ndarray] = []
    meta_rows: List[dict] = []
    txt_lines: List[str] = []

    counts = None
    if _W_DO_COUNTS:
        counts = np.zeros((vocab.vocab_size,), dtype=np.int64)

    try:
        for triples, meta in iter_piece_triple_windows(
            piece_path=piece_path,
            input_dir=input_dir,
            seed=_W_SEED,
            xml_group=_W_XML_GROUP,
            max_windows_per_file=_W_MAX_WINDOWS_PER_FILE,
            max_windows_per_voice=_W_MAX_WINDOWS_PER_VOICE,
        ):
            events = encode_triples_to_events(triples, vocab, bos_start_pos=meta.bos_start_pos)
            padded, used_len = pad_or_truncate_events(events, max_events=max_events, vocab=vocab)

            batch_rows.append(padded)

            n_notes = int(used_len) - 2
            n_notes = max(0, min(n_notes, note_budget))
            span_pos = compute_span_pos_from_triples(triples, note_limit=n_notes)

            _seq_n += 1
            _notes_sum += int(n_notes)
            _len_sum += int(used_len)
            _span_sum += int(span_pos)
            _len_min = int(used_len) if _len_min is None else min(_len_min, int(used_len))
            _len_max = int(used_len) if _len_max is None else max(_len_max, int(used_len))
            _notes_min = int(n_notes) if _notes_min is None else min(_notes_min, int(n_notes))
            _notes_max = int(n_notes) if _notes_max is None else max(_notes_max, int(n_notes))
            _span_min = int(span_pos) if _span_min is None else min(_span_min, int(span_pos))
            _span_max = int(span_pos) if _span_max is None else max(_span_max, int(span_pos))

            meta_rows.append({
                "source_file": meta.source_file,
                "voice_id": meta.voice_id,
                "voice_idx": meta.voice_idx,
                "window_idx": meta.window_idx,
                "window_start_note_idx": meta.window_start_note_idx,
                "bos_start_pos": meta.bos_start_pos,
                "used_len_events": int(used_len),
            })

            if _W_WRITE_TXT:
                txt_lines.append(triples_to_token_line(triples, bos_start_pos=meta.bos_start_pos))

            if counts is not None:
                flat = padded[:used_len].reshape(-1)  # exclude pads
                counts += np.bincount(flat, minlength=vocab.vocab_size)

    except Exception as e:
        return {
            "split": sp,
            "piece_path": piece_path_s,
            "error": str(e),
            "batch": None,
            "meta": [],
            "txt": [],
            "counts": None,
            "export_stats": {
                "sequences": int(_seq_n),  # 这个 piece 产出的序列条数
                "len_events": {"count": int(_seq_n), "sum": int(_len_sum), "min": _len_min, "max": _len_max},
                "notes":      {"count": int(_seq_n), "sum": int(_notes_sum), "min": _notes_min, "max": _notes_max},
                "span_pos":   {"count": int(_seq_n), "sum": int(_span_sum), "min": _span_min, "max": _span_max},
            },
        }

    if not batch_rows:
        return {
            "split": sp,
            "piece_path": piece_path_s,
            "error": None,
            "batch": None,
            "meta": [],
            "txt": [],
            "counts": counts,
            "export_stats": {
                "sequences": int(_seq_n),  # 这个 piece 产出的序列条数
                "len_events": {"count": int(_seq_n), "sum": int(_len_sum), "min": _len_min, "max": _len_max},
                "notes":      {"count": int(_seq_n), "sum": int(_notes_sum), "min": _notes_min, "max": _notes_max},
                "span_pos":   {"count": int(_seq_n), "sum": int(_span_sum), "min": _span_min, "max": _span_max},
            },
        }

    batch = np.stack(batch_rows, axis=0).astype(np.int32, copy=False)  # (n, max_events, 3)

    return {
        "split": sp,
        "piece_path": piece_path_s,
        "error": None,
        "batch": batch,
        "meta": meta_rows,
        "txt": txt_lines,
        "counts": counts,
        "export_stats": {
            "sequences": int(_seq_n),  # 这个 piece 产出的序列条数
            "len_events": {"count": int(_seq_n), "sum": int(_len_sum), "min": _len_min, "max": _len_max},
            "notes":      {"count": int(_seq_n), "sum": int(_notes_sum), "min": _notes_min, "max": _notes_max},
            "span_pos":   {"count": int(_seq_n), "sum": int(_span_sum), "min": _span_min, "max": _span_max},
        },
    }


def build_pretrain_dataset(
    input_dir: Path,
    output_dir: Path,
    seed: int = 1234,
    xml_group: str = "staff_voice",
    ratios: Tuple[float, float, float] = DEFAULT_SPLIT,
    group_mode: str = "file",
    max_len_tokens: int = 0,
    write_txt_debug: bool = False,
    write_dict_txt: bool = False,
    debug: bool = True,
    num_workers: int = 0,
    chunksize: int = 8,
    maxtasksperchild: int = 200,
    ordered: bool = False,
    max_windows_per_file: int = 0,
    max_windows_per_voice: int = 0,
) -> Dict[str, int]:
    """
    Build SimpleMono pretrain corpus:
      - train.npy / valid.npy / test.npy   (int32, shape=(N, max_events, 3))
      - SimpleMono.pkl                     (token2id/id2token/event2word/word2event)
      - metadata.jsonl                     (sequence-level mapping)
      - split.json                         (piece-level split list)

    No intermediate train.txt/dict.txt by default.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir_empty(output_dir)

    vocab = SimpleMonoVocab.build()

    if max_len_tokens is None or max_len_tokens <= 0:
        max_len_tokens = default_max_len_tokens()
    if max_len_tokens % 3 != 0:
        raise ValueError(f"max_len_tokens must be divisible by 3, got {max_len_tokens}")
    max_events = max_len_tokens // 3

    if debug:
        print(f"[INFO] Reading score files from {input_dir}...")

    files = iter_score_files(input_dir)

    if debug:
        print(f"[INFO] Found {len(files)} score files.")
        print(f"[INFO] Splitting files by {group_mode}...")

    splits = split_files(files, input_dir=input_dir, seed=seed, ratios=ratios, group_mode=group_mode)

    # save split.json (piece-level)
    split_json = {
        k: [str(p.relative_to(input_dir).as_posix()) for p in v]
        for k, v in splits.items()
    }
    (output_dir / "split.json").write_text(json.dumps(split_json, ensure_ascii=False, indent=2), encoding="utf-8")

    if debug:
        print(f"[INFO] split.json done.")

    # writers
    writers = {
        sp: NpyAppendWriter(output_dir / f"{sp}.npy", dtype=np.int32, row_shape=(max_events, 3))
        for sp in ["train", "valid", "test"]
    }

    # optional debug txt
    txt_f = None
    if write_txt_debug:
        txt_f = {
            sp: (output_dir / f"{sp}.txt").open("w", encoding="utf-8")
            for sp in ["train", "valid", "test"]
        }

    meta_f = (output_dir / "metadata.jsonl").open("w", encoding="utf-8")

    do_counts = bool(write_dict_txt)
    counts_by_id = np.zeros((vocab.vocab_size,), dtype=np.int64) if do_counts else None
    row_idx = {"train": 0, "valid": 0, "test": 0}

    piece_total = sum(len(v) for v in splits.values())
    piece_done = 0

    bpm = DEFAULT_STATS_BPM
    pos_resolution = int(POS_RESOLUTION)

    acc_by_split = {sp: SeqExportStatsAccumulator() for sp in ["train", "valid", "test"]}

    file_counts = {
        sp: {
            "files_assigned": int(len(splits[sp])),
            "files_error": 0,
            "files_with_sequences": 0,
            "files_zero_sequences": 0,
        }
        for sp in ["train", "valid", "test"]
    }

    # ----------------------------
    # multiprocessing config
    # ----------------------------
    if num_workers is None or num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    if debug:
        print(f"[INFO] num_workers={num_workers}, chunksize={chunksize}, ordered={ordered}, maxtasksperchild={maxtasksperchild}")

    # 多进程时，建议把 BLAS 线程数压到 1，避免每个进程再开一堆线程导致“越并行越慢”
    # （spawn 子进程会继承环境变量）
    if num_workers > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # ----------------------------
    # main loop: compute in workers, write in main
    # ----------------------------
    if num_workers <= 1:
        # 单进程：保留你原来的逻辑（但记得 counts_by_id 要 conditional）
        for sp in ["train", "valid", "test"]:
            print(f"[INFO] Processing {sp} set...")
            for piece_path in tqdm(
                splits[sp],
                desc=f"Processing {sp} set",
                unit="piece",
                ascii=True,
                total=len(splits[sp]),
                smoothing=0.0,
            ):
                piece_has_seq = False
                try:
                    for triples, meta in iter_piece_triple_windows(
                        piece_path=piece_path,
                        input_dir=input_dir,
                        seed=seed,
                        xml_group=xml_group,
                        max_windows_per_file=max_windows_per_file,
                        max_windows_per_voice=max_windows_per_voice,
                    ):
                        events = encode_triples_to_events(triples, vocab, bos_start_pos=meta.bos_start_pos)
                        padded, used_len = pad_or_truncate_events(events, max_events=max_events, vocab=vocab)

                        note_budget = int(max_events - 2)
                        n_notes = int(used_len) - 2
                        n_notes = max(0, min(n_notes, note_budget))

                        span_pos = compute_span_pos_from_triples(triples, note_limit=n_notes)
                        acc_by_split[sp].add(len_events=int(used_len), n_notes=int(n_notes), span_pos=int(span_pos))
                        piece_has_seq = True

                        if counts_by_id is not None:
                            flat = padded[:used_len].reshape(-1)
                            counts_by_id += np.bincount(flat, minlength=vocab.vocab_size)

                        writers[sp].append(padded)

                        if txt_f is not None:
                            txt_f[sp].write(triples_to_token_line(triples, bos_start_pos=meta.bos_start_pos) + "\n")

                        meta_f.write(json.dumps({
                            "split": sp,
                            "row_idx": row_idx[sp],
                            "source_file": meta.source_file,
                            "voice_id": meta.voice_id,
                            "voice_idx": meta.voice_idx,
                            "window_idx": meta.window_idx,
                            "window_start_note_idx": meta.window_start_note_idx,
                            "bos_start_pos": meta.bos_start_pos,
                            "used_len_events": int(used_len),
                            "max_events": int(max_events),
                        }, ensure_ascii=False) + "\n")
                        row_idx[sp] += 1

                except Exception as e:
                    file_counts[sp]["files_error"] += 1
                    print(f"[ERROR] {piece_path}: {e}")
                    continue

                if piece_has_seq:
                    file_counts[sp]["files_with_sequences"] += 1
                else:
                    file_counts[sp]["files_zero_sequences"] += 1

    else:
        ctx = mp.get_context("spawn")  # 跨平台稳一点；Linux 也能用
        mtpc = None if (maxtasksperchild is None or maxtasksperchild <= 0) else int(maxtasksperchild)

        map_fn_name = "imap" if ordered else "imap_unordered"

        if debug:
            print(f"[INFO] Using multiprocessing ({map_fn_name}, spawn).")

        def _merge_piece_export_stats(acc: SeqExportStatsAccumulator, st: dict | None) -> int:
            """
            Merge worker-returned per-piece export_stats into a split accumulator.
            Returns the number of sequences merged (0 if none).
            """
            if not st:
                return 0

            n = int(st.get("sequences", 0))
            if n <= 0:
                return 0

            acc.n_seq += n

            # These dicts must have keys: count,sum,min,max
            acc.len_events.merge(RunningIntStats(**st["len_events"]))
            acc.n_notes.merge(RunningIntStats(**st["notes"]))
            acc.span_pos.merge(RunningIntStats(**st["span_pos"]))
            return n

        with ctx.Pool(
            processes=int(num_workers),
            initializer=_init_pretrain_worker,
            initargs=(
                str(input_dir), int(seed), str(xml_group), int(max_events),
                bool(write_txt_debug), bool(do_counts),
                int(max_windows_per_file), int(max_windows_per_voice),
            ),
            maxtasksperchild=mtpc,
        ) as pool:

            map_fn = pool.imap if ordered else pool.imap_unordered

            for sp in ["train", "valid", "test"]:
                print(f"[INFO] Processing {sp} set...")

                tasks = ((sp, os.fspath(p)) for p in splits[sp])

                it = map_fn(_process_piece_task, tasks, chunksize=int(chunksize))

                for res in tqdm(
                    it,
                    desc=f"Processing {sp} set",
                    unit="piece",
                    ascii=True,
                    total=len(splits[sp]),
                    smoothing=0.0,
                ):

                    err = res.get("error", None)
                    if err:
                        file_counts[sp]["files_error"] += 1
                        tqdm.write(f"[ERROR] {res.get('piece_path')}: {err}")
                        continue

                    # ---- (NEW) merge export_stats from worker ----
                    st = res.get("export_stats", None)
                    seq_n_stats = _merge_piece_export_stats(acc_by_split[sp], st)  # 0 if none

                    batch = res.get("batch", None)
                    if batch is None:
                        # this piece produced 0 sequences
                        file_counts[sp]["files_zero_sequences"] += 1
                        continue

                    file_counts[sp]["files_with_sequences"] += 1

                    # optional sanity check
                    n_seq_batch = int(batch.shape[0])
                    if seq_n_stats > 0 and n_seq_batch != seq_n_stats:
                        tqdm.write(
                            f"[WARN] export_stats mismatch: {res.get('piece_path')} "
                            f"export_stats.sequences={seq_n_stats} batch.shape[0]={n_seq_batch}"
                        )

                    meta_list = res.get("meta", [])
                    txt_list = res.get("txt", [])

                    # --- your existing code continues ---
                    n_seq = n_seq_batch
                    if len(meta_list) != n_seq:
                        tqdm.write(f"[WARN] meta_list len mismatch: {res.get('piece_path')} meta={len(meta_list)} batch={n_seq}")
                    if txt_f is not None and len(txt_list) != n_seq:
                        tqdm.write(f"[WARN] txt_list len mismatch: {res.get('piece_path')} txt={len(txt_list)} batch={n_seq}")

                    writers[sp].append(batch)

                    if counts_by_id is not None:
                        c = res.get("counts", None)
                        if c is not None:
                            counts_by_id += c

                    if txt_f is not None:
                        for line in txt_list:
                            txt_f[sp].write(line + "\n")

                    for m in meta_list:
                        meta_f.write(json.dumps({
                            "split": sp,
                            "row_idx": row_idx[sp],
                            "max_events": int(max_events),
                            **m,
                        }, ensure_ascii=False) + "\n")
                        row_idx[sp] += 1

    # close
    for w in writers.values():
        w.close()
    meta_f.close()
    if txt_f is not None:
        for f in txt_f.values():
            f.close()

    # overall
    overall_acc = SeqExportStatsAccumulator()
    for sp in ["train", "valid", "test"]:
        overall_acc.merge(acc_by_split[sp])

    export = {
        "schema": "simplemono_export_stats_v1",
        "dataset_kind": "pretrain",
        "bpm": float(bpm),
        "pos_resolution": int(pos_resolution),
        "max_events": int(max_events),
        "files": {sp: file_counts[sp] for sp in ["train", "valid", "test"]},
        "splits": {
            sp: {
                "counts": {
                    "files_assigned": file_counts[sp]["files_assigned"],
                    "files_with_sequences": file_counts[sp]["files_with_sequences"],
                    "sequences": acc_by_split[sp].n_seq,
                },
                "tracks": {
                    "x": acc_by_split[sp].to_dict(pos_resolution=pos_resolution, bpm=bpm),
                },
            }
            for sp in ["train", "valid", "test"]
        },
        "overall": {
            "counts": {
                "files_assigned": sum(file_counts[sp]["files_assigned"] for sp in ["train", "valid", "test"]),
                "files_with_sequences": sum(file_counts[sp]["files_with_sequences"] for sp in ["train", "valid", "test"]),
                "sequences": overall_acc.n_seq,
            },
            "tracks": {"x": overall_acc.to_dict(pos_resolution=pos_resolution, bpm=bpm)},
        },
    }

    write_json(output_dir / "export_stats.json", export)

    # write outputs
    vocab.save_simplemono_pkl(output_dir / "SimpleMono.pkl")

    if write_dict_txt:
        assert counts_by_id is not None
        vocab.write_dict_txt(output_dir / "dict.txt", counts_by_id=counts_by_id)

    print("[DONE]")
    print("Output:", output_dir)
    for sp in ["train", "valid", "test"]:
        print(f"  - {sp}.npy: {row_idx[sp]} sequences")
    print("  - SimpleMono.pkl")
    print("  - metadata.jsonl")
    print("  - split.json")
    if write_txt_debug:
        print("  - train/valid/test.txt (debug)")
    if write_dict_txt:
        print("  - dict.txt (optional)")

    return {sp: row_idx[sp] for sp in row_idx}

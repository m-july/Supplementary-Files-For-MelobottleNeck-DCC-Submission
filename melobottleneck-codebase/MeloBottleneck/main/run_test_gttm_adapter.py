# run_test_gttm_adapter.py
import numpy as np
import torch
from pathlib import Path
from main.vocab_utils import load_vocab_info
from main.evaluation import make_gttm_benchmark_dataloader, GTTMBackboneEvalConfig, evaluate_gttm_backbone
from main.evaluation.score_extraction import ScoreArrayAdapter

VOCAB_PKL = r".\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\SimpleMono.pkl"

GTTM_SPLIT_DIR = r".\preproc\output\gttm_bench_v1.2\test"

vocab = load_vocab_info(VOCAB_PKL)
root = Path(GTTM_SPLIT_DIR)

gold = np.load(root/"gold_score.npy")  # [N,L]
adapter = ScoreArrayAdapter(gold, pad_id=vocab.pad_id, special_n=vocab.special_n)

loader = make_gttm_benchmark_dataloader(root, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

cfg = GTTMBackboneEvalConfig(
    eval_rho=2/3,
    force_z_len_rho=None,      # ScoreArrayAdapter 不用 z_len
    compute_cut_curve=True,
    show_progress=True,
    amp=False,
)

m = evaluate_gttm_backbone(adapter, loader, device=torch.device("cpu"), cfg=cfg, prefix="oracle/")
print(m)

# usage:
# python -m main.run_test_gttm_adapter
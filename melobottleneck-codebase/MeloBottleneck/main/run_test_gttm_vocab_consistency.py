# run_test_gttm_vocab_consistency.py
import pickle, re
import numpy as np
from main.vocab_utils import load_vocab_info

VOCAB_PKL = r".\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\SimpleMono.pkl"

vocab = load_vocab_info(VOCAB_PKL)
with open(VOCAB_PKL, "rb") as f:
    obj = pickle.load(f)

token2id = obj["token2id"]
special_n = int(vocab.special_n)
dt_samples = int(vocab.deltatime_code_offset)

bad = []

# ---- duration: <1-c> ----
# 你也可以直接用 0..95（如果你确定 DUR_SAMPLES=96）
for c in range(0, 96):
    tok = f"<1-{c}>"
    if tok not in token2id:
        bad.append(("missing_duration_tok", tok))
        continue
    gid = token2id[tok]
    local = int(vocab.global2local_duration[gid])
    expect = special_n + c
    if local != expect:
        bad.append(("dur_mismatch", tok, gid, local, expect))

# ---- dt positive: <2-c> ----
for c in range(0, dt_samples):
    tok = f"<2-{c}>"
    if tok not in token2id:
        bad.append(("missing_dt_tok", tok))
        continue
    gid = token2id[tok]
    local = int(vocab.global2local_dt[gid])
    expect = special_n + c
    if local != expect:
        bad.append(("dt_pos_mismatch", tok, gid, local, expect))

# ---- dt negative: <2-!mag> encodes -mag ----
for mag in range(1, dt_samples + 1):
    tok = f"<2-!{mag}>"
    if tok not in token2id:
        bad.append(("missing_dt_tok", tok))
        continue
    gid = token2id[tok]
    local = int(vocab.global2local_dt[gid])
    expect = special_n + dt_samples + (mag - 1)
    if local != expect:
        bad.append(("dt_neg_mismatch", tok, gid, local, expect))

print("layout_check_bad_n =", len(bad))
print(bad[:30])

print("----------------------------")

for s in ["<pad>", "<s>", "</s>", "<mask>"]:
    gid = obj["token2id"][s]
    print(
        s,
        int(vocab.global2local_pitch[gid]),
        int(vocab.global2local_duration[gid]),
        int(vocab.global2local_dt[gid]),
    )


print("----------------------------")

pad, bos, eos, special_n = vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.special_n

def check(arr, name, n=100):
    bad = 0
    for i in range(min(n, arr.shape[0])):
        x = arr[i]
        # find first eos in pitch channel
        eos_pos = np.where(x[:,0] == eos)[0]
        if len(eos_pos) == 0:
            bad += 1; continue
        e = int(eos_pos[0])

        if not (x[0,0]==bos and x[0,1]==bos):
            bad += 1

        # EOS triple policy
        if not (x[e,0]==eos and x[e,1]==eos and x[e,2]==eos):
            print(name, "EOS triple differs at i=", i, "row=", x[e])
            bad += 1

        # after EOS should be pad in pitch
        if np.any(x[e+1:,0] != pad):
            print(name, "non-pad after eos at i=", i)
            bad += 1

    print(name, "bad =", bad)

TRAIN_NPY = r".\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\train.npy"
GTTM_X_NPY = r".\preproc\output\gttm_bench_v1.2\test\x.npy"

train = np.load(TRAIN_NPY, mmap_mode="r")   # [N,L,3] global or local? 取你训练时真正喂模型的版本
gttm  = np.load(GTTM_X_NPY, mmap_mode="r")  # [N,L,3] local

check(gttm, "gttm")

# usage:
# python -m main.run_test_gttm_vocab_consistency
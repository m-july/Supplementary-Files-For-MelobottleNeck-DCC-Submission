import numpy as np

pi = np.load("./preproc/output/tavern_silver_otb/test/pi.npy", mmap_mode="r")
len_orn = np.load("./preproc/output/tavern_silver_otb/test/len_x_orn.npy", mmap_mode="r")

span = []
center = []
skel_frac = []

for i in range(pi.shape[0]):
    L = int(len_orn[i])          # events length including BOS/EOS
    n = max(0, L - 2)            # number of note-events
    if n <= 1: 
        continue
    # note positions in pi are [1 .. 1+n-1]
    note_pi = pi[i, 1:1+n]
    mask = (note_pi >= 0)        # matched => skeleton
    idx = np.where(mask)[0]
    if idx.size == 0:
        continue
    span_ratio = (idx.max() - idx.min()) / (n - 1)
    ctr = (idx.mean() + 0.5) / n
    span.append(span_ratio)
    center.append(ctr)
    skel_frac.append(mask.mean())

print("span_ratio: min/mean/max", min(span), sum(span)/len(span), max(span))
print("center:     min/mean/max", min(center), sum(center)/len(center), max(center))
print("skel_frac:  min/mean/max", min(skel_frac), sum(skel_frac)/len(skel_frac), max(skel_frac))

# usage:
# python -m preproc.run_test_tavern_silver
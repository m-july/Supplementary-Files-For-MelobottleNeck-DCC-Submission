## Terminology Notes: Paper vs. Code

The codebase was developed before the final paper terminology was fixed, so a few
internal names differ from the terms used in the paper. The implementation and
the paper refer to the same model components unless explicitly stated otherwise.

### Main naming differences

- **Timeline Alignment Loss** in the paper is the same component as the
  implementation's **guided-attention loss**.
  - Paper notation: $\mathcal L_{\mathrm{T}}$, coefficient $\lambda_{\mathrm{T}}$
  - Code / earlier notes: $\mathcal L_{\mathrm{G}}$, `loss_G`, `lambda_G`
  - Meaning: an early-training regularizer that encourages the selected
    bottleneck positions to follow the global score timeline, preventing the
    extractor from collapsing to a narrow local region.

- **Ornament Exclusion Loss** in the paper is the same component as the
  implementation's **insertion loss**.
  - Paper notation: $\mathcal L_{\mathrm{E}}$, coefficient $\lambda_{\mathrm{E}}$
  - Code / earlier notes: $\mathcal L_{\mathrm{I}}$, `loss_I`, `lambda_I`
  - Meaning: a penalty on the selection mass assigned to newly inserted
    ornament events in the procedurally ornamented view.

- **Rhythmic closure** in the paper corresponds to the implementation's
  **forward-extend postprocess**.
  - Paper term: rhythmic closure / `Close(z)`
  - Code / earlier notes: forward-extend, forward-extended sequence
  - Meaning: the selected raw subsequence is not yet a standalone melody, so the
    deleted score time is absorbed into retained events to produce a
    rhythmically self-consistent skeleton.

- **Note event** in the paper often corresponds to what older code comments call
  a **token**.
  - Paper term: note event / event
  - Code / earlier notes: token
  - Meaning: one musical event represented by a `(pitch, duration, delta-time)`
    triple. In the paper, "event" is preferred to avoid confusion with
    individual attribute tokens such as pitch, duration, or delta-time IDs.

- **Note-selection score** in the paper corresponds to the implementation's
  **salience score**.
  - Paper term: note-selection logit / note-selection score / selection
    distribution
  - Code / earlier notes: salience logit / salience score / salience
    distribution
  - Meaning: the scalar score used by the compressor to rank note events before
    top-$K$ subsequence extraction. The paper avoids calling this "salience"
    because the method is framed as latent subsequence selection rather than
    note-wise salience classification.

### Stage names

- **Denoising Pretraining** in the paper corresponds to **Stage A** or
  **Seq2Seq Denoising MusicBART Pretraining** in the code and earlier notes.

- **Melody Prior Training** in the paper corresponds to **Stage B** or
  **Frozen Decoder-Only Music Prior** in the code and earlier notes.

- **Subsequence Bottleneck Training** in the paper corresponds to **Stage C** or
  **End-to-End Subsequence-bottleneck Training** in the code and earlier notes.

- **Pseudo-label Note Classification** in the paper corresponds to
  **O2B-Learner** in the code and earlier notes.

### Metrics and diagnostics

- **Diagnostic musical-bias metrics** in the paper correspond to the older
  **proxy lift** terminology.
  - Paper: Strong-beat Lift
  - Earlier notes: Lift Strong
  - Paper: Duration Lift
  - Earlier notes: Lift Duration

### Loss-name quick reference

| Paper term | Paper symbol | Code / earlier name | Typical code key |
| --- | --- | --- | --- |
| Reconstruction Loss | $\mathcal L_{\mathrm{R}}$ | reconstruction loss | `loss_R` |
| Melody Prior Loss | $\mathcal L_{\mathrm{P}}$ | frozen LM prior loss | `loss_P` |
| Length Regularization Loss | $\mathcal L_{\mathrm{L}}$ | length regularization loss | `loss_L` |
| Timeline Alignment Loss | $\mathcal L_{\mathrm{T}}$ | guided-attention loss | `loss_G` |
| Ornament-Invariant Consistency Loss | $\mathcal L_{\mathrm{C}}$ | consistency loss | `loss_C` |
| Ornament Exclusion Loss | $\mathcal L_{\mathrm{E}}$ | insertion loss | `loss_I` |

When reading the code together with the paper, prefer the paper terminology for
conceptual discussion, but keep the original code keys when referring to
configuration files, logs, checkpoints, or backward-compatible APIs.
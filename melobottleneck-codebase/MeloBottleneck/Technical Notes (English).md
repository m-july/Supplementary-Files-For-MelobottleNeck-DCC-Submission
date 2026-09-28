# Technical Notes (English)

> **Terminology note for paper readers.**  
> This document intentionally uses the implementation-side terminology found in the code and code comments. Some names differ from the final paper terminology. Before reading this document together with the paper, please refer to **"Terminology Notes: Paper vs. Code"**. For example, the paper's *Timeline Alignment Loss* $\mathcal L_{\mathrm{T}}$ corresponds to the implementation's guided-attention `loss_G`, and the paper's *Ornament Exclusion Loss* $\mathcal L_{\mathrm{E}}$ corresponds to the implementation's insertion `loss_I`. In this document, the code/technical-document terms are retained for easier cross-reference with the repository.

> Some parts of this document may be outdated. Please refer primarily to the data in actual code and in the paper.

> This note is translated by ChatGPT 5.5 Thinking from Chinese.

# SimpleMono: OctupleMIDI-Style Monophonic Three-Attribute Encoding

**Core idea.** SimpleMono inherits the multi-attribute factorized event representation of OctupleMIDI, but structurally represents only the note sequence of a single voice. It removes explicit absolute-time attributes such as bar and position, and instead connects adjacent notes using relative delta-time. Each note is encoded as a triple event `(pitch, duration, delta-time)`. At the beginning of the sequence, the third attribute of the BOS event provides a clipped temporal anchor.

### 1) Time Discretization: the `pos` Grid

Let the time-grid resolution be $R=12$, so that

$$
1\ \text{pos}=\frac{1}{R}\ \text{quarter note}.
$$

The input, such as MIDI, MusicXML, or KRN, is first quantized to integer `pos` values. For each voice, we also apply **empty-measure trimming**: measure intervals in which the entire voice is silent are removed, and all following notes are shifted leftward, i.e., temporally compressed. This reduces the modeling burden caused by long-distance silence.

### 2) Numerical Semantics and Clipping of Note Triples: Pitch / Duration / Delta-Time

For a quantized note sequence

$$
\{(p_i,s_i,e_i)\}_{i=1}^N,
$$

where each note is represented by `(pitch, onset, offset)`, we encode notes as triples:

- **Pitch**: $p_i\in[0,127]$, the MIDI pitch. Tokens have the form `<0-p_i>`, giving 128 pitch tokens from `<0-0>` to `<0-127>`.

- **Duration**: uniformly quantized in `pos` units:

    $$
    d_i=\max(1,e_i-s_i),\quad c^D_i=\min(d_i,N_D-1).
    $$

    Here $N_D=96$, so $c^D_i\in[0,95]$. Tokens have the form `<1-c^D_i>`, giving 96 duration tokens from `<1-0>` to `<1-95>`.

- **Delta-Time**: uniformly quantized, signed, and measured in `pos` units. It is defined as the difference between the next note's onset and the current note's offset:

    $$
    \Delta_i=
    \begin{cases}
    s_{i+1}-e_i,& i<N,\\
    0,& i=N,
    \end{cases}
    \quad
    c^\Delta_i=\mathrm{clip}(\Delta_i,-N_\Delta,N_\Delta-1).
    $$

    Here $N_\Delta=96$, so $c^\Delta_i\in[-96,95]$. Negative values naturally express note overlap, i.e., the next onset is earlier than the current offset. Tokens have the form `<2-c^Delta_i>`, with exclamation marks used to denote negative values. The delta-time vocabulary contains `<2-0>` to `<2-95>`, plus `<2-!1>` to `<2-!96>`, giving 192 tokens in total.

> Additional ordering rule: notes with the same onset time are first sorted by `(start, end, pitch)` and then deterministically shuffled within the same-onset group using a seed derived from `piece + voice`. This weakens the bias caused by a fixed intra-chord ordering. It also means that serializing notes with the same onset often produces negative $\Delta$ values.

- The total vocabulary size is 421.
    - Note events are represented as `(pitch, duration, delta-time)`, i.e.,

        $$
        E_i=(\langle0-p_i\rangle,\langle1-c^D_i\rangle,\langle2-c^\Delta_i\rangle).
        $$

    - There are 5 special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`, and `<mask>`. The BOS event reuses the delta-time quantizer to encode the clipped onset time of the first note in the window:

        $$
        E_0=(\langle s\rangle,\langle s\rangle,\langle2-\mathrm{clip}(s_1)\rangle).
        $$

- Maximum token length: 514.

### 3) Multi-Attribute Embedding Scheme

- We use a multi-attribute concatenate-and-project embedding scheme similar to MusicBERT. Each attribute has its own embedding table with $d_{\rm attr}=256$. The attribute embeddings are concatenated and then projected by a linear layer $W_{\uparrow}$ into the shared hidden space, with $d_{\rm model}=512$.
- The token heads used later in the models are weight-tied to the attribute embeddings. However, the reverse linear projection $W_{\downarrow}$ uses separate parameters.

---

# MeloBottleneck: Ornament-Invariant Subsequence Bottleneck for Self-Supervised Melody Skeleton Extraction

- We use a three-branch convergence scheme: Stage A performs pretraining, Stage B trains the LM Prior, and Stage C performs joint bottleneck training.

## Stage A: **Seq2Seq** Denoising MusicBART Pretraining

- Goal: **learn general musical representations and generation ability.** We pretrain a general music encoder-decoder backbone so that it has:
    - music token-level modeling ability;
    - the ability to recover from local masking, span masking, deletion, rotation, and related corruptions;
    - robustness to simple music-equivalent transformations such as transposition and time scaling.

- Architecture: MusicBART-style seq2seq denoising model.
    - Multi-attribute token embedding: 🔥 randomly initialized.
    - Backbone encoder: 🔥 randomly initialized.
    - Backbone decoder: 🔥 randomly initialized.
    - Multi-attribute LM head: tied embedding weights.

- Data processing path:
    - The training sample first goes through basic music augmentation $\mathcal A$, and then BART-style corruption $\mathcal C$:

        $$
        x \xrightarrow{\mathcal A} x'
        \xrightarrow{\mathcal C} \tilde x.
        $$

    - The model learns to reconstruct $x'$ from $\tilde x$.
    - Music Augmentation $\mathcal A$ includes:
        - **Pitch transpose**: sample a semitone shift from a Gaussian distribution;
        - **Time scaling**: sample duration scaling from `2.0x`, `1.0x`, and `0.5x`.
    - BART-Style Corruption $\mathcal C$ includes:
        - **Masking**: supports granularities `note@attribute`, `note@full-token`, `n-note@attribute`, and `n-note@full-token`. For `n-note@full-token`, the entire span is collapsed into a single `[MASK]`.
        - **Deletion**: supports granularities `note@full-token` and `note@full-token`.
        - **Document Rotation**.
    - Corruption curriculum:
        - Masking starts weak; deletion and rotation are enabled later.
        - The curriculum is smoothly updated by step rather than by epoch.

- Training objective: multi-attribute seq2seq cross-entropy.
    - We compute CE separately for pitch, duration, and delta-time, and take a weighted sum:

        $$
        \mathcal L_\text{A}
        =
        \lambda_p^{\text{A}}\,\mathrm{CE}_{p}
        +
        \lambda_d^{\text{A}}\,\mathrm{CE}_{d}
        +
        \lambda_\Delta^{\text{A}}\,\mathrm{CE}_{\Delta}.
        $$

    - Typical weights:

        $$
        (\lambda_p^{\text{A}},\lambda_d^{\text{A}},\lambda_\Delta^{\text{A}})=(1/3,1/3,1/3).
        $$

## Stage B: **Frozen Decoder-Only** Music Prior

- Goal: **learn a frozen autoregressive music prior.** The purpose is not to provide further initialization weights, but to train a decoder-only Music LM that will later be frozen and used as a language prior. The final model uses it to judge whether the extracted skeleton sequence is music-like.

- Architecture: autoregressive language model.
    - Multi-attribute token embedding: 🔥 initialized from Stage A.
    - Backbone decoder: 🔥 initialized from Stage A.
    - Multi-attribute LM head: tied embedding weights.

- Data processing path:
    - Only augmentation is applied:

        $$
        x \xrightarrow{\mathcal A_{\text{LM}}} x'.
        $$

    - The model trains the autoregressive language model on $x'$ as the corpus.
    - The augmentation mechanism $\mathcal A_{\text{LM}}$ is similar to $\mathcal A$, but duration shortening is rarer and duration lengthening is more frequent.

- Training objective: next-token cross-entropy.

    $$
    \mathcal L_{\text{B}}
    =
    \lambda_p^{\text{LM}}\,\mathrm{CE}_{p}
    +
    \lambda_d^{\text{LM}}\,\mathrm{CE}_{d}
    +
    \lambda_\Delta^{\text{LM}}\,\mathrm{CE}_{\Delta}.
    $$

    - Typical weights:

        $$
        (\lambda_p^{\text{LM}},\lambda_d^{\text{LM}},\lambda_\Delta^{\text{LM}})=(0.5,0.3,0.2).
        $$

## Stage C: End-to-End Subsequence-Bottleneck Training

- Goal: **learn the actual skeleton extractor end-to-end**, through sequence bottleneck reconstruction and ornament invariance.

- Architecture:
    - Compressor:
        - Multi-attribute token embedding: 🔥 initialized from Stage A.
        - Backbone encoder: 🔥 initialized from Stage A.
        - Skeleton token salience score head: 🔥 randomly initialized.
        - Rho predictor: 🔥 randomly initialized.
    - Forward-extend postprocess.
    - Reconstructor:
        - Multi-attribute token embedding: shared with the compressor.
        - Backbone encoder: shared with the compressor.
        - Backbone decoder: 🔥 initialized from Stage A.
        - Multi-attribute LM head: tied embedding weights.
    - LM prior: ❄️ from Stage B.

- Data augmentation:

    $$
    x \xrightarrow{\mathcal A} x'.
    $$

- Compressor architecture:

    $$
    x' \xrightarrow{\text{embed + encoder}} h
    \xrightarrow{\text{score head}} \ell.
    $$

    The compressor produces a salience score logit $\ell$ for each token. It first uses the mean-pooled encoded memory $h$ to predict the sequence compression ratio $\rho$, where $\rho_{\min}=1/3$ and $\rho_{\max}=1$. This gives the continuous intermediate-sequence length $T_{\text{cont}}$ and the hard length $T_{\text{hard}}$:

    $$
    \rho(x')
    =
    \rho_{\min}
    +
    (\rho_{\max}-\rho_{\min})\,
    \sigma\!\big(\mathrm{MLP}(\operatorname{Pool}(h))\big),
    $$

    $$
    T_{\text{cont}}
    =L_x\rho(x'),
    \quad
    T_{\text{hard}}
    =
    \left\lceil T_{\text{cont}}\right\rceil.
    $$

    Then, according to the compression ratio $\rho(x')$, the model takes the salience top-$T_{\text{hard}}$ tokens and obtains a compressed sequence that strictly preserves the original temporal order:

    $$
    z=\left(x_{i_1}',x_{i_2}',\dots,x_{i_K}',\text{EOS}\right),
    \quad
    i_1<i_2<\dots<i_K.
    $$

    In the implementation, to allow gradients to flow to the continuous length $T_{\text{cont}}$, a soft length gate $\gamma_t$ is applied to the later Reconstructor bottleneck embeddings:

    $$
    \gamma_t
    =
    \sigma\!\left(
    \frac{T_{\text{cont}}-(t+0.5)}{T_\rho}
    \right),
    \qquad
    T_\rho=0.5.
    $$

- Forward-extend postprocess: the current compressed sequence $z$ is still only a sampled subsequence of the original sequence $x'$. It is called the on-the-fly sequence $z$, and its duration / delta-time values are often no longer self-consistent. This postprocessing algorithm forwards the time occupied by deleted ornaments, passing tones, and syncopations into the previous skeleton note. As a result, the output postprocessed sequence $\bar z$ is a rhythmically self-consistent skeleton melody that can stand alone.

- Straight-through bottleneck embedding:
    - Forward pass: use the embedding of the forward-extended hard sequence $\bar z$.
    - Backward pass: use the compressor's soft-gathered $z$ embedding.
    - The straight-through form is:

        $$
        z^{\text{ST}}_t
        =
        \gamma_t (\operatorname{sg}\!\big(E(\bar z^{\text{hard}}_{t}) - E(z^{\text{soft}}_{t})\big)
        +
        E(z^{\text{soft}}_{t}))
        $$

        where $\operatorname{sg}(\cdot)$ denotes stop-gradient.

- Reconstructor: a Seq2Seq structure reconstructs the original sequence $x'$ from $z^{\text{ST}}$. In effect, it learns how to re-ornament the original sequence from the skeleton sequence.
    - Reconstructor compression conditioning: FiLM modulation is applied using the sample compression ratio, injecting the compression-ratio condition into the bottleneck embeddings.

- Core training objective:

    $$
    \mathcal J
    =
    \lambda_{\text{R}}\mathcal L_{\text{R}}
    +
    \lambda_{\text{P}}\mathcal L_{\text{P}}
    +
    \lambda_{\text{L}}\mathcal L_{\text{L}}
    +
    \lambda_{\text{G}}\mathcal L_{\text{G}}
    +
    \lambda_{\text{C}}\mathcal L_{\text{C}}
    +
    \lambda_{\text{I}}\mathcal L_{\text{I}}.
    $$

    - Reconstruction loss $\mathcal L_{\text{R}}$:
        - Goal: require the compressed skeleton sequence $\bar z$ to retain enough structural information about the original sequence $x'$.
        - Definition: duration-weighted multi-attribute CE for Reconstructor reconstruction of $x'$.

            $$
            \mathcal L_{\text{R}}
            =
            \frac{
            \sum_l d_l
            \left(
            \lambda_p^{\text{R}}\mathrm{CE}_{p,l}
            +
            \lambda_d^{\text{R}}\mathrm{CE}_{d,l}
            +
            \lambda_\Delta^{\text{R}}\mathrm{CE}_{\Delta,l}
            \right)
            }{
            \sum_l d_l
            }.
            $$

        - Typical weights:

            $$
            (\lambda_p^{\text{R}},\lambda_d^{\text{R}},\lambda_\Delta^{\text{R}})=(1/3,1/3,1/3).
            $$

        - Decoder regularization: decoder-input mask dropout with a curriculum warming up from `0%` masked, i.e., fully teacher-forced, to `80%` masked. This forces the Reconstructor decoder to rely more on the bottleneck-sequence information provided by the encoder, rather than only on autoregressive inertia.

    - Frozen LM prior loss $\mathcal L_{\text{P}}$:
        - Goal: require the intermediate sequence $\bar z$ to resemble a natural music sequence.
        - Definition: measure the degree to which $\bar z$ follows the frozen LM prior model trained in Stage B. Specifically, according to the compressor's soft selection distribution at each sampling step, we induce attribute distributions $r_t^p$, $r_t^d$, and $r_t^\Delta$ for each token $\bar z_t$, and compute their KL divergence against the next-token distributions of the frozen LM prior at the current step.

            $$
            \mathcal L_{\text{P}}
            =
            \sum_t
            \Big[
            \alpha_p\,\mathrm{KL}(r_t^p\,\|\,p_{\text{LM}}^p(\cdot\mid\bar z_{<t}))
            +
            \alpha_d\,\mathrm{KL}(r_t^d\,\|\,p_{\text{LM}}^d(\cdot\mid\bar z_{<t}))
            +
            \alpha_\Delta\,\mathrm{KL}(r_t^\Delta\,\|\,p_{\text{LM}}^\Delta(\cdot\mid\bar z_{<t}))
            \Big].
            $$

        - Typical weights:

            $$
            (\alpha_p,\alpha_d,\alpha_\Delta)=(0.7,0.2,0.1).
            $$

    - Length regularization loss $\mathcal L_{\text{L}}$:
        - Goal: require the model to actively handle a range of dynamic compression ratios.
        - Definition: match the predicted $\rho$ distribution within a batch to a truncated Gaussian target distribution. Let the effective compression ratios predicted within the batch be sorted as

            $$
            \rho^{\text{eff}}_{(1)}\le \rho^{\text{eff}}_{(2)}\le\cdots\le\rho^{\text{eff}}_{(B)}.
            $$

            Construct clipped normal target quantiles:

            $$
            u_i=\frac{i-0.5}{B},
            \qquad
            q_i=\operatorname{clip}\big(\mu_{\text{L}}+\sigma_{\text{L}}\Phi^{-1}(u_i),\rho_{\min},\rho_{\max}\big).
            $$

            Then the length regularization is:

            $$
            \mathcal L_{\text{L}}
            =
            \sum_{i=1}^B
            (\rho^{\text{eff}}_{(i)}-q_i)^2.
            $$

        - Typical values: $\mu_{\text{L}}=0.67$, $\sigma_{\text{L}}=0.22$.

    - Guided attention loss $\mathcal L_{\text{G}}$:
        - Goal: to avoid skeleton selection collapsing to a local region during early training, the current implementation also adds a guided attention loss based on normalized score time.
        - Definition: normalize the onsets of the $\bar z$ sequence as $u_l$, normalize the linear progress of score time as $v_t$, and measure the quadratic loss between these two normalized timelines:

            $$
            \mathcal L_{\text{G}}
            =
            \frac{1}{\sum_t m_t}
            \sum_t m_t\sum_l p_t(l)\cdot\frac{(u_l-v_t)^2}{2\sigma^2_{\text{G}}},
            $$

            $$
            u_l=\frac{o_l}{o_{\text{EOS}}},
            \quad
            v_t=\frac{t}{z_{\text{len}}-1}.
            $$

        - Typical value: $\sigma_{\text{G}}=0.075$.
        - Curriculum: this loss is only applied in the early stage; later, $\lambda_G$ decays to 0.

- Ornament invariance:
    - Goal: make skeleton extraction invariant to whether the sequence is ornamented, while also penalizing the selection of ornament notes.
    - Data flow:
        - Weak view: the $x'$ obtained above.
        - Ornamented strong view: $x'\xrightarrow{\mathcal O}(x^{(\text{orn})},y)$, which gives $x^{(\text{orn})}$.
        - Music Ornamenter $\mathcal O$: an algorithmic system that generates an ornamented version $x^{(\mathrm{orn})}$ online from the original sequence $x'$. It also outputs a skeleton-label sequence $y$ in parallel, indicating whether each ornamented token is a skeleton note ($y_k=1$) or an inserted ornament note ($y_k=0$).

            For most sequences, ornamentation preserves the total duration. For a smaller portion of sequences, ornamentation increases the total duration, turning the piece into a free-rhythm style sequence. For the validation/test O2B benchmark, we use an Out-of-Distribution distribution $\mathcal O_{\text{OOD}}$, which enables some mechanisms unseen during training and uses out-of-training-range parameters for some mechanisms.

    - Teacher-student mechanism:
        - The teacher extracts salience logits on the weak view $x'$ using the compressor in eval mode, and then applies softmax to obtain the distribution $s(x')$.
        - The student processes the ornamented strong view $x^{(\text{orn})}$ while being forced to use the same compression-ratio budget, extracts salience logits, and applies softmax to obtain $s(x^{(\text{orn})})$.

    - Loss components:
        - Consistency loss $\mathcal L_{\text{C}}$:
            - Goal: make the token salience distributions from the weak view and the ornamented strong view consistent.
            - Definition: KL divergence between the token salience distributions produced by the teacher and the student.

                $$
                \mathcal L_{\text{C}}=\mathrm{KL}(s(x')\|s(x^{(\text{orn})})).
                $$

        - Insertion loss $\mathcal L_{\text{I}}$:
            - Goal: explicitly discourage ornament notes in the ornamented strong view from entering the skeleton.
            - Definition: the loss value is the sum of the student's softmax salience mass over inserted ornament notes.

                $$
                \mathcal L_{\text{I}}=\sum_{k:y_k=0}s_k(x^{(\text{orn})}).
                $$

## Appendix: Music Ornamenter Behaviour

- Music Ornamenter mechanisms for $\mathcal O$ and $\mathcal O_{\text{OOD}}$ include:
    - **pre-grace**: split a long note into a short note followed by a long note, and mark the long note as the skeleton note.
    - **post-grace**: split a long note into a long note followed by a short note, and mark the long note as the skeleton note.
    - **between insert**: insert 1 to 4 passing notes between two adjacent original notes, and mark the two original notes as skeleton notes.
    - **trill**: split a long note into 3 or 5 segments, alternating between the main pitch and a neighboring pitch obtained by pitch jitter; mark the first note as the skeleton note.
    - **turn (OOD-only)**: split a long note into a 3-segment pattern `[main, upper neighbor, main]` or a 4-segment pattern `[main, upper neighbor, main, lower neighbor]`; mark the first note as the skeleton note.
    - **rearticulation**: split a long note into 2 to 4 repeated articulations; mark the first note as the skeleton note.
    - **pair-repeat (OOD-only)**: split a pair of notes into 2 to 4 rounds of alternating repetition; mark the first and second notes as skeleton notes.

## Appendix: Parameters

- Encoder-decoder backbone:
    - `max_seq_len = 514`
    - `d_attr = 256`
    - `d_model = 512`
    - encoder / decoder layers = `4 / 4`
    - attention heads = `8`
    - FFN dim = `2048`

- Music Augmentation $\mathcal A$ and $\mathcal A_{\text{LM}}$:
    - $\mathcal A$:
        - Pitch transpose: `σ=6`, range `[-12,12]`
        - Time scaling: `p(2x)=0.30`, `p(0.5x)=0.30`
    - $\mathcal A_{\text{LM}}$:
        - Pitch transpose: `σ=6`, range `[-12,12]`
        - Time scaling: `p(2x)=0.50`, `p(0.5x)=0.20`

- BART-Style Corruption $\mathcal C$ after curriculum:
    - masking noise density: `0.333`
    - deletion probability: `0.2`
    - masking and deletion uniformly sample from their corresponding granularity types
    - masking and deletion `n-note` span lambda: `3.0`
    - rotation probability: `0.5`

- Parameters of stages:

    | Stage | A | B | C | O2B-Learner |
    | --- | --- | --- | --- | --- |
    | #epoch | 100 | 80 | 1 | 5 |
    | batch size | 160 | 200 | 28 | 256 |
    | learning rate | 5e-4 | 4e-4 | 3e-4 & 4e-4* | 5e-4 |
    | dropout | 0.15 | 0.15 | 0.075 | 0.10 |

    \*Stage C learning-rate warmup: LR linearly increases from 5% to 100% over 30% of the steps, then follows cosine decay to 5%; `lr_backbone=3e-4`, `lr_pointer=4e-4`.

---

# Datasets / Benchmarks

### Main Dataset: A Mixed Collection of World Folk Songs

| Name | Region | Selected Range |
| --- | --- | --- |
| Anthology of Chinese Folk Songs | Chinese | All |
| Jiugong Dacheng | Chinese | Train Split |
| BFDB: A dataset of British Folk melodies in ABC Format | British | All |
| Essen's Folksong | World-wide | All |
| Henrik Norbeck's ABC Tunes | Irish & Swedish | All |
| IrishMAN | Irish | Train Split #1 ~ #7999 |
| MTC-FS-INST-2.0 | Dutch | All |

- Total number of pieces: 58,154. Each piece is converted into one or more sequences using sliding windows. File-level set split: Train:Valid:Test = 18:1:1.

    | seed | split | #sequence | #note | duration@80BPM (s) | duration@80BPM |
    | --- | --- | --- | --- | --- | --- |
    | 10101 | train | 47070 | 4819504 | 2621826.0 |  |
    | 10101 | test | 2644 | 275262 | 151657.375 |  |
    | 20202 | train | 47110 | 4823833 | 2623428.375 |  |
    | 20202 | test | 2607 | 264910 | 144174.9375 |  |
    | 30303 | train | 47161 | 4832326 | 2630900.3125 |  |
    | 30303 | test | 2584 | 261697 | 139792.8125 |  |
    | 40404 | train | 47112 | 4817670 | 2621921.875 |  |
    | 40404 | test | 2632 | 278619 | 150697.375 |  |
    | 50505 | train | 47090 | 4831218 | 2628249.0 |  |
    | 50505 | test | 2643 | 268048 | 145882.5625 |  |
    | avg | train | 47108.6 | 4824910.2 | 2625265.1125 | 30d9h14m |
    | avg | test | 2622.0 | 269707.2 | 146441.0125 | 1d16h41m |

- The train set is used for training in all stages.

### Synthetic Benchmark: Main Ornament-to-Backbone (Main-O2B)

- Used for validation and testing.
- Data sources:
    - Valid Set: uses the validation split of the main dataset.
    - Test Set: uses the test split of the main dataset.
- Benchmark construction:
    - For each sequence $x$, use OOD Music Ornament to obtain the ornamented sequence $x_{\rm orn}$, with the compression ratio constrained to $[1/3,1]$.
    - Construct evaluation pairs $x_{\rm orn}\to x$. During evaluation, the model uses the fixed compression ratio

        $$
        \rho=\mathrm{len}(x)/\mathrm{len}(x_{\rm orn}).
        $$

- Number of sequences: 2,593.

### Zero-Shot Cross-Domain Benchmark: TAVERN Variation-to-Theme (TAVERN-V2T)

- Used only for testing.
- Data source: variation-theme piece pairs from **TAVERN: A New Data Set for Symbolic Music Analysis**. The dataset consists of Mozart and Beethoven classical piano scores, which are cross-domain relative to our main corpus. We only use the right-hand main melody part.
- Benchmark construction:
    - For each variant $x$ and theme $y$ in the dataset, we apply Needleman-Wunsch alignment with Match, Insertion, and Deletion operations. This gives the subsequence $x'$ in the variation that is optimally aligned as corresponding to the theme. Data pairs with low matching scores are discarded; 9% of the pairs in the original corpus are removed.
    - Construct evaluation pairs $x\to x'$. During evaluation, the model uses the fixed compression ratio

        $$
        \rho=\mathrm{len}(x')/\mathrm{len}(x).
        $$

- Number of sequences: 591.

### Zero-Shot In-Domain Benchmark: Jiugong Dacheng Ornamented-to-Gongche (Jiugong-O2G)

- Used only for testing.
- Data source: **Jiugong Dacheng Nanbei Ci Gongpu** is a Qing-dynasty collection of opera and court-music scores. It records melody skeletons using Gongche notation. Wang Zhenglai's *Newly Edited Translation and Annotation of Jiugong Dacheng Nanbei Ci Gongpu* translates it into modern numbered musical notation while adding a series of ornament notes based on practical performance experience with Gongche notation.
- Benchmark construction:
    - For each pair of Gongche skeleton melody $x$ and numbered-notation translation $y$, we manually input $y$ as MIDI and annotate the subsequence $y'$ aligned to $x$.
    - Construct evaluation pairs $y\to y'$. During evaluation, the model uses the fixed compression ratio

        $$
        \rho=\mathrm{len}(y')/\mathrm{len}(y).
        $$

- Appendix note on the Jiugong split: the Jiugong portion in the Main Train Set includes all volumes except Volumes 2, 51, and 71. The Jiugong retrieval set uses Volumes 2, 51, and 71. Jiugong-O2G selects `*` pieces from Volumes 2, 51, and 71 and manually inputs them.

# Baselines and Ablations

### Naive Baselines

- **Random downsampling**: sample according to the compression ratio.
    - Soft score: uniformly distribute mass over the selected tokens, including BOS and excluding EOS, so that the total mass is 1.

- **Uniform downsampling**: sample uniformly on normalized score time according to the compression ratio.
    - Soft score: uniformly distribute mass over the selected tokens, including BOS and excluding EOS, so that the total mass is 1.

- **Top-K duration**: select the longest notes according to the compression ratio.
    - Soft score: uniformly distribute mass weighted by duration.

### Heuristic Baseline: AMR-No-Harmony

- Automatic Melody Reduction via Shortest Path Finding (AMR) is a heuristic algorithm. It treats a melody as a weighted directed graph and converts the Music Reduction problem into a shortest-path problem from the first note to the last note.
- The original algorithm depends on simultaneously provided harmony data. To adapt it to our melody-only benchmarks, we modify the algorithm and obtain an adapted version, AMR-No-Harmony:
    - remove harmony dependency by removing related edge weights;
    - add length control, so that results can be generated for any specified compression ratio;
    - allow free endpoints, fixing the original algorithm's constraint that the first and last notes must be selected.

### Pseudo-Label Classifier: O2B-Learner

- Goal: use an encoder to learn a binary classification task, **skeleton vs. ornament**, from O2B pseudo-labels generated by Music Ornamenter.

- Architecture:
    - Multi-attribute token embedding: 🔥 randomly initialized.
    - Backbone encoder: 🔥 randomly initialized.
    - Skeleton token salience score head: 🔥 randomly initialized.

- Data processing path:
    - First apply Augmentation $\mathcal A$, then Ornamenter $\mathcal O$:

        $$
        x \xrightarrow{\mathcal A} x'
        \xrightarrow{\mathcal O}(x^{(\mathrm{orn})},y).
        $$

    - Model computation:

        $$
        x^{(\mathrm{orn})}
        \xrightarrow{\text{embed + encoder}}h
        \xrightarrow{\text{score head}}\ell.
        $$

        The score head is a two-layer MLP with $d_{\text{middle}}=256$. It extracts one salience score logit for each note token in the sequence, representing each note's salience, i.e., skeletal importance.

- Training objective: per-note-token binary classification loss. The note mask is defined as $m_k$:

    $$
    \mathcal L_{\text{O2B-L}}
    =
    \frac{1}{\sum_k m_k}
    \sum_k
    m_k\,\omega(y_k)\,
    \mathrm{BCEWithLogits}(\ell_k,y_k).
    $$

    - The negative class, i.e., ornament notes, is dynamically upweighted:

        $$
        \omega(0)
        =
        \operatorname{clip}\!\left(\frac{N_{\text{pos}}}{N_{\text{neg}}},1,10\right),
        \qquad
        \omega(1)=1.
        $$

### Ablations

- Stage ablation: 1 group, removing BART-denoise initialization.
- Loss ablations: 6 groups, obtained by respectively setting $\lambda_{\text{R}}$, $\lambda_{\text{P}}$, $\lambda_{\text{L}}$, $\lambda_{\text{G}}$, $\lambda_{\text{C}}$, and $\lambda_{\text{I}}$ to zero.

# Evaluation Metrics

### Benchmark Evaluation Metrics

> This family of metrics measures the quality of skeleton-note extraction.

- **Hard-set F1 Score (Hard F1)**: measures the end-to-end quality of skeleton-note extraction. It is the token-level F1 between the model's final discrete skeleton-note set and the ground-truth skeleton-note set. Higher is better.

- **Soft Average Precision (Soft AP)**: measures the model's ranking quality for skeleton notes. It is the binary-classification Average Precision (AP) between the per-token salience scores output by the model and the ground-truth selection labels. Higher is better.

- **Normalized F1 AUC over Cut Ratios (Cut F1 AUC-N)**: measures the overall F1 quality of the model under different compression ratios. For a series of cut compression ratios, top-$k$ is computed with $k$ varying by cut ratio. The corresponding F1 values are computed, the F1-cut curve is integrated using the trapezoidal rule, and the result is normalized by the cut-interval length. Higher is better.

- **Insertion Mass**: measures the model's ability to avoid selecting ornament notes. The model's token scores over a sequence are normalized into a mass distribution, and the probability mass on ground-truth ornament notes is summed. Lower is better.

### Music Prior Proxy Lifts

> This family of metrics measures whether skeleton-note extraction conforms to a set of heuristic musical priors.

- Computation scheme, placed in the appendix: for any feature $f_l\ge0$ defined only on note tokens, compute the ratio between the salience-weighted expectation and the uniform-baseline expectation as the Lift score. A value larger than 1 indicates that the score has a positive preference for that feature.

    $$
    \mathrm{Lift}(f)
    =
    \frac{\mathbb E_{\tilde{s}}[f]}{\mathbb E_U[f]},
    \quad
    \mathbb E_{\tilde{s}}[f]
    =
    \sum_{l\in\mathcal N}\tilde{s}_l f_l,
    \quad
    \mathbb E_U[f]
    =
    \frac{1}{|\mathcal N|}\sum_{l\in\mathcal N}f_l.
    $$

- **Lift Strong**: measures the skeleton-note extraction model's preference for notes on strong beats, where every two beats define a strong beat.
- **Lift Duration**: measures the skeleton-note extraction model's preference for long-duration notes.

### Auxiliary Metrics

> These are enabled only for the last $N$ batches and indicate reconstruction quality during training.

- **Weighted geometric mean perplexity**:
    - For the average attribute CEs over the last $N$ batches, $\bar L_p$, $\bar L_d$, and $\bar L_\Delta$, define:

        $$
        \text{PPL}_p=e^{\bar L_p},
        \quad
        \text{PPL}_d=e^{\bar L_d},
        \quad
        \text{PPL}_\Delta=e^{\bar L_\Delta}.
        $$

    - Combine them using normalized attribute weights:

        $$
        \text{PPL}_{\text{recon}}
        =
        \exp\!\left(
        \frac{\lambda_p\bar L_p+\lambda_d\bar L_d+\lambda_\Delta\bar L_\Delta}{\lambda_p+\lambda_d+\lambda_\Delta}
        \right).
        $$

- **Bottleneck Dependence Ratio**:
    - Under teacher forcing, compute:
        - $\mathcal L_{\text{rec}}^{\text{TF}}(z)$: reconstruction loss under the normal bottleneck condition;
        - $\mathcal L_{\text{rec}}^{\text{TF}}(\varnothing)$: reconstruction loss after removing the bottleneck condition. In implementation, the bottleneck input is replaced with dummy memory.
    - Then compute:

        $$
        \mathrm{TF\text{-}BDR}
        =
        \frac{\mathcal L_{\text{rec}}^{\text{TF}}(\varnothing)}
        {\mathcal L_{\text{rec}}^{\text{TF}}(z)}.
        $$

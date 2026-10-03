# MeloBottleneck Experiment Codebase

This is the experiment codebase for the manuscript “MeloBottleneck: Compact Symbolic Melody Representations for Robust Retrieval”. For an overview of the supplied files and the audio demo, see the [repository README](../../README.md).

## Repository Structure

| Location | Contents |
| --- | --- |
| [ckpt/](ckpt/) | Included evaluation records. Model checkpoint weights are distributed separately. |
| [data_jiugong_o2g/](data_jiugong_o2g/) | MIDI source files for the Jiugong ornamented-to-gongche (O2G) benchmark. |
| [demo_midi_source/](demo_midi_source/) | Original melodies and extracted skeletons used for the qualitative demo. |
| [main/](main/) | Training, inference, evaluation, and retrieval scripts. |
| [preproc/](preproc/) | SimpleMono preprocessing and decoding scripts. |
| [preproc/output/](preproc/output/) | Included metadata, statistics, and vocabulary files; prepared NumPy data are supplied in `output.rar`. |
| `retrieval_exp/` | Retrieval intermediates, populated by extracting `retrieval_exp.rar` or running the retrieval workflow. |
| [requirements.txt](requirements.txt) | Pinned base dependencies, excluding PyTorch and optional score-format readers. |
| [results.xlsx](results.xlsx) | Per-seed results for MeloBottleneck, baselines, and ablations. |
| [results-summary.xlsx](results-summary.xlsx) | Aggregate results across five seeds. |

See [More Notes](#more-notes) for the technical and terminology documents.

## Experiment Results

[results.xlsx](results.xlsx) reports the individual seed results, and [results-summary.xlsx](results-summary.xlsx) reports the mean and standard deviation across seeds. The included [ckpt/](ckpt/) tree contains evaluation records for the main model, baselines, and ablations. These files can be inspected without downloading model weights or running training.

## Data and Model Checkpoints

### Supplied Data Archives

The data archives are already included under [other-materials/](../../other-materials/):

| Archive | Contents | Extraction target, relative to this codebase directory |
| --- | --- | --- |
| [output.rar](../../other-materials/output.rar) | Prepared corpus and benchmark arrays, metadata, vocabulary files, and decoded MIDI files; approximately 2.31 GiB unpacked. | `preproc/`; retain the top-level `output/` folder. |
| [retrieval_exp.rar](../../other-materials/retrieval_exp.rar) | Prepared queries, query metadata, and document/query skeleton arrays; approximately 180 MiB unpacked. | `./`; retain the top-level `retrieval_exp/` folder. |

The unpacked sizes refer to all file contents in each supplied archive. Merge the extracted folders with the corresponding repository folders. From this codebase directory, Windows `tar` can extract both RAR archives:

```powershell
tar -xf ..\..\other-materials\output.rar -C .\preproc
tar -xf ..\..\other-materials\retrieval_exp.rar -C .
```

An archive application with RAR support can be used with the same extraction targets. After extraction, the relevant data layout is:

```text
preproc/output/
  jiugong_test_for_retrieval/
  real_jiugongdacheng_otb_bench/
  seed10101/
  seed20202/
  seed30303/
  seed40404/
  seed50505/
  tavern_silver_otb/
retrieval_exp/
  jiugong_query_ornamented_corrupted/
  jiugong_skeleton/
```

The archive also contains demo data and the legacy `gttm_bench_v1.2/` benchmark. GTTM evaluation is disabled in the default training configuration. To prepare a custom corpus, see [Appendix B](#appendix-b-simplemono-data-preprocessing-and-decoding).

### Model Checkpoints

Model weights (`.pt` files) are not included. To request them, contact [m.july@qq.com](mailto:m.july@qq.com) and specify the model variant and seed. Checkpoint files should be placed under `ckpt/` with the appropriate experiment directory structure.

The included evaluation records use the following roots:

```text
ckpt/2604-final/seed10101/...                 # main model and ablations
ckpt/keep-baseline-2604-final/seed10101/...   # O2B-Learner variants
```

Equivalent directories exist for seeds 20202, 30303, 40404, and 50505. The current training entry point writes new main-model runs to `ckpt/main-2604-final/` by default; see [Training Configuration](#training-configuration) before selecting a checkpoint path.

Training from scratch requires the prepared corpus data. Skeleton inference requires a compatible model checkpoint. The original-melody and random-ranking retrieval baselines, as well as retrieval over the supplied skeleton arrays, can run without checkpoint weights.

## Environment Setup

The reference experiment environment uses Python 3.10 and PyTorch 2.9.0 with the CUDA 13.0 build. The commands below use Windows PowerShell and start from the repository root:

```powershell
cd .\melobottleneck-codebase\MeloBottleneck
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Choose a PyTorch build compatible with your hardware and driver using the [official installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/). The documented training configuration uses `device="cuda"`; inference commands expose `--device`.

### Optional Score-Format Readers

MIDI preprocessing uses the base dependencies. Install the additional reader only for the source format you need:

```powershell
python -m pip install partitura  # MusicXML
python -m pip install music21   # Humdrum/Kern (.krn)
```

These optional readers are not version-pinned in the reference requirements. Legacy GTTM/MuDeP scripts also require external baseline code and its dependencies; those scripts are outside the main training and retrieval workflow documented here.

## Run Experiments

All commands below run from `melobottleneck-codebase/MeloBottleneck/`. Extract the required data archives first. Paths are relative to this directory.

### Training Configuration

The training entry points configure experiments in their Python source. Edit the settings near the `__main__` block before launching a run; the training commands below do not accept seed or variant selection flags.

| Setting | MeloBottleneck: [train_skeleton_end2end.py](main/train_skeleton_end2end.py) | O2B-Learner: [train_skeleton_keep_baseline.py](main/train_skeleton_keep_baseline.py) |
| --- | --- | --- |
| Seeds | `SEEDS`: 10101, 20202, 30303, 40404, 50505. | The same five seeds. |
| Experiment variants | `ablation_experiments`: the full model (`def`) and eight ablations. | `VARIANTS`: `w_bart_init` and `wo_bart_init`. |
| Input data | `DATA_ROOT` and `CORPUS_DIRNAME` select each seed's prepared corpus. | The same settings select each seed's corpus. |
| Output root | `OUTPUT_ROOT = r".\ckpt\main-2604-final"`. | `OUTPUT_ROOT = r".\ckpt\keep-baseline-2604-final"`. |
| Pretraining | Stages A and B are trained or reused once per seed under `_AB_shared/`. | The initialized variant loads Stage A weights from `STAGEA_ROOT`, whose default is `.\ckpt\2604-final`. |
| W&B logging | `use_wandb=True` in `_base_cfg_template`. | `use_wandb=True` in `base_cfg`. |

For a single-seed full-model run, replace the existing seed and variant definitions in `train_skeleton_end2end.py` with:

```python
SEEDS = [10101]
ablation_experiments = [("def", {}, {})]
```

For a single-seed initialized O2B-Learner run, use the following definitions in `train_skeleton_keep_baseline.py`:

```python
SEEDS = [10101]
VARIANTS = [("w_bart_init", False)]
```

The initialized baseline requires `STAGEA_ROOT/seed10101/_AB_shared/stageA_pretrain/backbone_pretrained_last.pt`. If Stage A was trained using the current MeloBottleneck output defaults, set `STAGEA_ROOT = r".\ckpt\main-2604-final"` in the baseline entry point. The `wo_bart_init` variant does not require Stage A weights.

To disable W&B, set `use_wandb=False` in the entry point's configuration object; changing only the dataclass default does not override this explicit setting. If logging is enabled, configure W&B authentication and `WANDB_PROJECT` before training. Adjust the stage-specific `batch_size` settings to fit the available GPU memory; [Appendix A](#hardware-and-efficiency) records the reference hardware and batch sizes.

### Run MeloBottleneck

After configuring the seeds, variants, data paths, and logging:

```powershell
python -m main.train_skeleton_end2end
```

The current default runner shares Stage A/B checkpoints across the Stage C variants of each seed. A full-model checkpoint from a new run is written to:

```text
ckpt/main-2604-final/seed10101/def/stageC_skeleton/skeleton_last.pt
```

The runner reuses existing Stage A/B weights and skips Stage C runs with an existing final checkpoint. Set `FORCE_RETRAIN_AB=True` to retrain Stages A/B, and `FORCE_RERUN_C=True` to rerun Stage C; these switches operate independently.

For inference, replace the example paths with the checkpoint and SimpleMono arrays you want to use:

```powershell
python -m main.infer_train_skeleton_end2end --ckpt ".\path\to\skeleton_last.pt" --input_npy ".\path\to\melodies.npy" --output_npy ".\path\to\skeletons.npy" --device cuda --batch_size 128 --rho_mode predict
```

See [Appendix B](#appendix-b-simplemono-data-preprocessing-and-decoding) to decode SimpleMono arrays to MIDI.

### Run O2B-Learner

After configuring the baseline variants and, for the initialized variant, `STAGEA_ROOT`:

```powershell
python -m main.train_skeleton_keep_baseline
```

### Run Retrieval Experiment

The commands below use 10,000 queries, fragment lengths of 8–64 notes, trigram BM25 (`n=3`), and `topM=50`. Document skeletons use a fixed retention setting of 0.8, and query skeletons use predicted retention. The result blocks are the recorded experiment outputs.

For a step-by-step guide, including preprocessing a new retrieval corpus, see [run_bm25.md](main/retrieval/run_bm25.md). If using the supplied `retrieval_exp.rar` arrays, you can run retrieval directly and skip query generation and skeleton inference. Generating new document/query skeletons requires a checkpoint; the example checkpoint path below refers to the existing experiment layout and should be replaced when using a newly trained model.

- Make ornamented corrupted fragment queries
    
    ```powershell
    python -m main.retrieval.make_fragment_queries --corpus_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --out_dir .\retrieval_exp\jiugong_query_ornamented_corrupted --n_queries 10000 --frag_min_notes 8 --frag_max_notes 64 --ornament_apply --ornament_json .\preproc\ornament_ood.json --corrupt_apply --corrupt_json .\main\retrieval\corrupt.json --seed 1234
    ```
    
- Full sequence retrieval (ornamented corrupted fragment - full seq)
    
    ```powershell
    python -m main.retrieval.run_bm25 --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --n 3 --topM 50
    ```
    
    - Result
        
        ```text
        ========== Results ==========
        Recall@1: 0.1170  (1170/10000)
        Recall@5: 0.2013  (2013/10000)
        Recall@10: 0.2442  (2442/10000)
        Recall@20: 0.2867  (2867/10000)
        MRR: 0.1592
        [Query] time: 3.66601s | per-query: 0.0003666s
        empty queries: 0
        ================================
        ```
        
- Skeleton inference (document and query)
    
    ```powershell
    python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\2604-final\seed10101\def\stageC_skeleton\skeleton_last.pt --input_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --output_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --rho_mode fixed --rho 0.8 --device cuda --batch_size 128
    ```
    
    ```powershell
    python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\2604-final\seed10101\def\stageC_skeleton\skeleton_last.pt --input_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --output_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --rho_mode predict --device cuda --batch_size 128
    ```
    
- Skeleton retrieval (skeletal ornamented corrupted fragment - skeletal full seq)
    
    ```powershell
    python -m main.retrieval.run_bm25 --docs_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --queries_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --n 3 --topM 50
    ```
    
    - Result
        
        ```text
        ========== Results ==========
        Recall@1: 0.2084  (2084/10000)
        Recall@5: 0.3145  (3145/10000)
        Recall@10: 0.3579  (3579/10000)
        Recall@20: 0.4018  (4018/10000)
        MRR: 0.2584
        [Query] time: 2.73521s | per-query: 0.0002735s
        empty queries: 0
        ================================
        ```
        
- Baseline retrieval metrics
    
    ```powershell
    python -m main.retrieval.run_random_rank_baseline --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --topM 50
    ```
    
    - Result
        
        ```text
        ========== Random Ranking Baseline ==========
        N_docs: 219
        N_queries: 10000
        valid_targets: 10000  (ratio=1.0000)
        topM: 50  (effective topM=min(topM,N_docs)=50)
        Ks: [1, 5, 10, 20]
        
        --- Analytic (exact expectation; std is std-of-mean over queries approx) ---
        Recall@1: 0.004566   (approx 95% CI [0.003245, 0.005888])
        Recall@5: 0.022831   (approx 95% CI [0.019904, 0.025759])
        Recall@10: 0.045662   (approx 95% CI [0.041571, 0.049754])
        Recall@20: 0.091324   (approx 95% CI [0.085678, 0.096970])
        MRR: 0.020544   (approx 95% CI [0.018905, 0.022184])
        ===========================================
        ```
        

## More Notes

Refer to these notes for more information:

- [Terminology Notes (English)](Terminology%20Notes.md): mappings between the terminology in the paper and the implementation.
- Technical Notes ([Chinese](Technical%20Notes%20%28Chinese%29.md) / [English](Technical%20Notes%20%28English%29.md)): details of SimpleMono and MeloBottleneck.

## Appendix A: Datasets, Hardware, and Efficiency

### Main Dataset Components

| Name | Region | Selected Range |
| --- | --- | --- |
| Anthology of Chinese Folk Songs | Chinese | All |
| Jiugong Dacheng | Chinese | Train Split |
| BFDB: A dataset of British Folk melodies in ABC Format | British | All |
| Essen's Folksong | World-wide | All |
| Henrik Norbeck's ABC Tunes | Irish & Swedish | All |
| IrishMAN | Irish | Train Split #1 ~ #7999 |
| MTC-FS-INST-2.0 | Dutch | All |

### Hardware and Efficiency

- Reference training settings:

    | Stage | A | B | C | O2B-Learner |
    | --- | --- | --- | --- | --- |
    | Epochs | 100 | 80 | 1 | 5 |
    | Batch size | 160 | 200 | 28 | 256 |
- Reference hardware:
    - GPU: RTX 5090 D (32GB VRAM)
    - CPU: AMD Ryzen 9 9950X 16-Core Processor
    - RAM: 64GB
- Training times, averaged across five seeds:
    - Stages A + B: 3h8m
    - O2B-Learner (initialized from Stage A): 10m48s
    - MeloBottleneck (initialized from Stage A, with the frozen prior from Stage B): 28m52s

## Appendix B: SimpleMono Data Preprocessing and Decoding

- Prepare a training corpus:
    - The preprocessing pipeline recursively reads files under the specified input directory. It supports MIDI, MusicXML, and Humdrum/Kern; MusicXML and Kern require the [optional readers](#optional-score-format-readers).
    
    ```powershell
    python -m preproc.preproc_pretrain --input_dir "[your_corpus_dir]" --output_dir ".\preproc\output\seed70707\[your_corpus_name]" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file --num_workers 8 --seed 70707
    ```
    
- Generate procedurally ornamented validation and test data:
    
    ```powershell
    python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed70707\[your_corpus_name]\valid.npy" --vocab_pkl ".\preproc\output\seed70707\[your_corpus_name]\SimpleMono.pkl" --output_dir ".\preproc\output\seed70707\[your_corpus_name]\otb_bench" --split_name valid_ood --rho_min 0.3333333333 --seed 70707 --show_hist --ornament_json ".\preproc\ornament_ood.json"
    ```
    
    ```powershell
    python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed70707\[your_corpus_name]\test.npy" --vocab_pkl ".\preproc\output\seed70707\[your_corpus_name]\SimpleMono.pkl" --output_dir ".\preproc\output\seed70707\[your_corpus_name]\otb_bench" --split_name test_ood --rho_min 0.3333333333 --seed 70707 --show_hist --ornament_json ".\preproc\ornament_ood.json"
    ```
    
- These custom-corpus examples use seed 70707 and a placeholder corpus name. To use the resulting data for training, update `SEEDS` and `CORPUS_DIRNAME` in the training entry point accordingly.
- For other preprocessing tasks, see the [preproc/](preproc/) scripts.
- To decode SimpleMono arrays to MIDI, replace the placeholder paths and choose `train`, `valid`, or `test` for `--metadata_split`:
    
    ```powershell
    python -m preproc.decode_npy_to_midi --npy_path ".\path\to\simplemono_data.npy" --metadata ".\path\to\metadata.jsonl" --out_dir ".\path\to\midi_output" --metadata_split test
    ```

# MeloBottleneck Experiment Codebase

This is the experiment codebase for manuscript “MeloBottleneck: Compact Symbolic Melody Representations for Robust Retrieval”.

# Repo Structure

- `./ckpt/`: checkpoints and evaluation results
- `./data_jiugong_o2g/`: MIDI source files of O2G (Jiugong ornamented-to-gongche) benchmark
- `./demo_midi_source/`: MIDI examples of Ours vs. baseline (O2B-Learner) skeleton extraction (source files for the web page DEMO)
- `./main/`: main scripts for MeloBottleneck
- `./preproc/`: main scripts for SimpleMono and SimpleMono data
    - `./preproc/output/`: folder for SimpleMono data (training corpus and benchmarks)
- `./retrieval_exp`: path for intermediate files of retrieval experiments
- `./requirements.txt`: a mininum Python package requirement list
- Experiment results
    - `./results.xlsx`: raw experiment results of ours model, baselines, and ablations
    - `./results-summary.xlsx`: results averaged over 5 seeds
- Technical notes
    - `./Technical Notes (Chinese).md`
    - `./Technical Notes (English).md`
    - `./Terminology Notes.md`

# Experiment Results

- Here are the experiment records corresponding to the results in the paper:
    - Overall results are at `./results.xlsx` (reporting results of each seed) and `./results-summary.xlsx` (reporting mean±std over all seeds).
    - You can find full experiment records at `./ckpt` .

# External Data and Checkpoints

> Note: Due to the double-blind review requirements and the CMT maximum file size, we are unable to provide the full external data and checkpoints at this stage. They will be made available through alternative hosting platforms in the published codebase.
> 
- SimpleMono format corpus data (1.79 GB uncompressed)
    - If you want to prepare your own custom data, see Appendix.
    - You should place them at: `./preproc/output` (currently there are metadata files only), that is:
        
        ```python
        ./preproc/output/gttm_bench_v1.2/... # obsolete
        ./preproc/output/jiugong_test_for_retrieval/...
        ./preproc/output/real_jiugongdacheng_otb_bench/...
        ./preproc/output/seed10101/...
        ./preproc/output/seed20202/...
        ./preproc/output/seed30303/...
        ./preproc/output/seed40404/...
        ./preproc/output/seed50505/...
        ./preproc/output/tavern_silver_otb/...
        ```
        
- Full experiment checkpoints and results (47.2 GB uncompressed)
    - You should place them at: `./ckpt` (currently there are experiment records only), that is:
        
        ```python
        ./ckpt/2604-final/seed10101/...
        ./ckpt/2604-final/seed20202/...
        ./ckpt/2604-final/seed30303/...
        ./ckpt/2604-final/seed40404/...
        ./ckpt/2604-final/seed50505/...
        ./ckpt/keep-baseline-2604-final/seed10101/...
        ./ckpt/keep-baseline-2604-final/seed20202/...
        ./ckpt/keep-baseline-2604-final/seed30303/...
        ./ckpt/keep-baseline-2604-final/seed40404/...
        ./ckpt/keep-baseline-2604-final/seed50505/...
        ```
        
- Retrieval experiment intermediates (180 MB uncompressed)
    - You should place them at: `./retrieval_exp` (currently it is empty), that is:
        
        ```python
        ./retrieval_exp/jiugong_query_ornamented_corrupted/...
        ./retrieval_exp/jiugong_skeleton/...
        ```
        

# Environment Setup

- We recommend using Python 3.10 and PyTorch 2.9.0+cu130. The experiments were conducted with CUDA 13.0.
- `./requirements.txt` is a minimum feasible setting and it doesn’t include PyTorch family (torch, torchaudio, torchvision).

# Run Experiments

> Before running these experiments, SimpleMono format corpus data (`./preproc/output`) is required. See “External Data and Checkpoints”.
> 

### Run MeloBottleneck

- To run our MeloBottleneck model (full model and ablations), execute this at root folder:
    
    ```python
    python -m main.train_skeleton_end2end
    ```
    
- After getting checkpoint, you can execute this for inference:
    
    ```python
    python -m main.infer_train_skeleton_end2end --ckpt [skeleton_ckpt.pt] --input_npy [simplemono_melodies.npy] --output_npy [simplemono_skeletons.npy] --device cuda --batch_size 128 --rho_mode predict
    ```
    
    - If you want to decode SimpleMono to MIDI, see “SimpleMono Data Preprocessing / Decoding” Section.

### Run O2B-Learner

- To run our O2B-Learner baseline (w/-Init and w/o-Init ablations), execute this at root folder:
    
    ```python
    python -m main.train_skeleton_keep_baseline
    ```
    

### Run Retrieval Experiment

- Make ornamented corrupted fragment queries
    
    ```python
    python -m main.retrieval.make_fragment_queries --corpus_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --out_dir .\retrieval_exp\jiugong_query_ornamented_corrupted --n_queries 10000 --frag_min_notes 8 --frag_max_notes 64 --ornament_apply --ornament_json .\preproc\ornament_ood.json --corrupt_apply --corrupt_json .\main\retrieval\corrupt.json --seed 1234
    ```
    
- Full sequence retrieval (ornamented corrupted fragment - full seq)
    
    ```python
    python -m main.retrieval.run_bm25 --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --n 3 --topM 50
    ```
    
    - Result
        
        ```python
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
    
    ```python
    python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\2604-final\seed10101\def\stageC_skeleton\skeleton_last.pt --input_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --output_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --rho_mode fixed --rho 0.8 --device cuda --batch_size 128
    ```
    
    ```python
    python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\2604-final\seed10101\def\stageC_skeleton\skeleton_last.pt --input_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --output_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --rho_mode predict --device cuda --batch_size 128
    ```
    
- Skeleton retrieval (skeletal ornamented corrupted fragment - skeletal full seq)
    
    ```python
    python -m main.retrieval.run_bm25 --docs_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --queries_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --n 3 --topM 50
    ```
    
    - Result
        
        ```python
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
    
    ```python
    python -m main.retrieval.run_random_rank_baseline --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --topM 50
    ```
    
    - Result
        
        ```python
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
        

# More Notes

Refer to these notes for more information:

- **Terminology Notes (English)**: explaining terminological differences between the paper and the code/documents.
- **Technical Notes (Chinese / English)**: technical document on SimpleMono and MeloBottleneck.

# Appendix A

### Appendix A1: Main Dataset Components

| Name | Region | Selected Range |
| --- | --- | --- |
| Anthology of Chinese Folk Songs | Chinese | All |
| Jiugong Dacheng | Chinese | Train Split |
| BFDB: A dataset of British Folk melodies in ABC Format | British | All |
| Essen's Folksong | World-wide | All |
| Henrik Norbeck's ABC Tunes | Irish & Swedish | All |
| IrishMAN | Irish | Train Split #1 ~ #7999 |
| MTC-FS-INST-2.0 | Dutch | All |

### Appendix A2: Hardware specifications and Efficiency statistics

- Local training setting
    
    
    | Stage | A | B | C | O2B-Learner |
    | --- | --- | --- | --- | --- |
    | #epoch | 100 | 80 | 1 | 5 |
    | batch size | 160 | 200 | 28 | 256 |
- Local hardware specification:
    - GPU: RTX 5090 D (32GB VRAM)
    - CPU: AMD Ryzen 9 9950X 16-Core Processor
    - RAM: 64GB
- Training time on stages, averaged over 5 seeds:
    - Stage 1 + 2: 3h8m
    - O2B-Learner (init from Stage 1): 10m48s
    - MeloBottleneck (init from Stage 1 and uses frozen prior from Stage 2): 28m52s

# Appendix B

### Appendix B1: SimpleMono Data Preprocessing / Decoding

- For training corpus data
    - Note: Supports MIDI, MusicXML and Kern Format. It recursively finds all files in all subfolders under the specified path.
    
    ```python
    python -m preproc.preproc_pretrain --input_dir "[your_corpus_dir]" --output_dir ".\preproc\output\seed70707\[your_corpus_name]" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file --num_workers 8 --seed 70707
    ```
    
- Furthermore, for procedually ornamented validation/test data
    
    ```python
    python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed70707\[your_corpus_name]\valid.npy" --vocab_pkl ".\preproc\output\seed70707\[your_corpus_name]\SimpleMono.pkl" --output_dir ".\preproc\output\seed70707\[your_corpus_name]\otb_bench" --split_name valid_ood --rho_min 0.3333333333 --seed 70707 --show_hist --ornament_json ".\preproc\ornament_ood.json"
    ```
    
    ```python
    python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed70707\[your_corpus_name]\test.npy" --vocab_pkl ".\preproc\output\seed70707\[your_corpus_name]\SimpleMono.pkl" --output_dir ".\preproc\output\seed70707\[your_corpus_name]\otb_bench" --split_name test_ood --rho_min 0.3333333333 --seed 70707 --show_hist --ornament_json ".\preproc\ornament_ood.json"
    ```
    
- For other data preprocessings, please refer to `./preproc` scripts.
- For decoding (SimpleMono → MIDI), run:
    
    ```python
    python -m preproc.decode_npy_to_midi --npy_path [path\simplemono_data.npy] --metadata [path\metadata.jsonl] --out_dir [path_output] --metadata_split [train/valid/test]
    ```
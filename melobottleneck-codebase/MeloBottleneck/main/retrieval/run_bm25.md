- 准备 SimpleMono 数据
    
    ```python
    python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\jiugong-test" --output_dir ".\preproc\output\jiugong_test_for_retrieval" --test_ratio 1.0 --valid_ratio 0 --train_ratio 0 --max_len_tokens 0 --group_mode file --max_windows_per_file 1
    ```
    
- 生成 ornamented corrupted fragment queries
    
    ```python
    python -m main.retrieval.make_fragment_queries --corpus_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --out_dir .\retrieval_exp\jiugong_query_ornamented_corrupted --n_queries 200 --frag_min_notes 6 --frag_max_notes 24 --ornament_apply --ornament_json .\preproc\ornament_ood.json --corrupt_apply --corrupt_json .\main\retrieval\corrupt.json --seed 1234
    ```
    
- Full Sequence 检索（ornamented corrupted fragment - full seq）
    
    ```python
    python -m main.retrieval.run_bm25 --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --n 3 --topM 100
    ```
    
- Skeleton Inference (Original and Query)
    
    ```python
    python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\stageC_last.pt --input_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --output_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --rho_mode predict --device cuda --batch_size 128
    ```
    
    ```python
    python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\stageC_last.pt --input_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --output_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --rho_mode predict --device cuda --batch_size 128
    ```
    
- Skeleton 检索 （skeletal ornamented corrupted fragment - skeletal full seq）
    
    ```python
    python -m main.retrieval.run_bm25 --docs_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --queries_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\anthology_for_retrieval\SimpleMono.pkl --n 3 --topM 100
    ```
    
- Baseline
    
    ```python
    python -m main.retrieval.run_random_rank_baseline --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --topM 100
    ```
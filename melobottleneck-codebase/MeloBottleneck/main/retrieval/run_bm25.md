# 检索实验操作说明

本说明与[代码库 README 的检索流程](../../README.md#run-retrieval-experiment)使用相同设置：10,000 条查询、8–64 个音符的片段、三元组 BM25（`n=3`）、`topM=50`；文档骨架使用固定保留率 0.8，查询骨架使用预测保留率。README 中保留了对应的历史实验输出。

## 运行目录与所需文件

先按[环境与数据说明](../../README.md#environment-setup)安装依赖，并解压仓库提供的数据。下面所有命令均在 `melobottleneck-codebase/MeloBottleneck/` 目录执行，而不是本说明所在的 `main/retrieval/` 目录。从仓库根目录进入：

```powershell
cd .\melobottleneck-codebase\MeloBottleneck
```

[output.rar](../../../../other-materials/output.rar) 中已包含 Jiugong 检索语料。[retrieval_exp.rar](../../../../other-materials/retrieval_exp.rar) 中已包含查询、查询元数据及文档／查询骨架。使用这些现成数组时，可跳过第 1、2、4 步，直接执行第 3、5、6 步，无需模型权重。

重新生成骨架需要一个兼容的模型 checkpoint。第 4 步的示例路径对应已记录实验的目录；若使用当前训练入口生成的模型，应替换为实际的权重路径，例如 `.\ckpt\main-2604-final\seed10101\def\stageC_skeleton\skeleton_last.pt`。权重获取方式见[模型说明](../../README.md#model-checkpoints)。

## 1. 准备新的 SimpleMono 检索语料（可选）

使用已提供的 Jiugong 数据时无需执行此步。若要处理自己的 MIDI 数据，将 `--input_dir` 替换为实际目录；后续命令中的语料和词表路径也应与生成的数据对应。

```powershell
python -m preproc.preproc_pretrain --input_dir ".\path\to\jiugong-test" --output_dir ".\preproc\output\jiugong_test_for_retrieval" --test_ratio 1.0 --valid_ratio 0 --train_ratio 0 --max_len_tokens 0 --group_mode file --max_windows_per_file 1
```

## 2. 生成带装饰与扰动的片段查询

此步会生成或替换 `queries.npy` 和 `queries_meta.jsonl`。生成新查询后，第 4 步中的查询骨架也需要重新推理，才能与查询元数据对应。

```powershell
python -m main.retrieval.make_fragment_queries --corpus_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --out_dir .\retrieval_exp\jiugong_query_ornamented_corrupted --n_queries 10000 --frag_min_notes 8 --frag_max_notes 64 --ornament_apply --ornament_json .\preproc\ornament_ood.json --corrupt_apply --corrupt_json .\main\retrieval\corrupt.json --seed 1234
```

## 3. 原始旋律检索

将带装饰和扰动的片段查询与原始完整旋律匹配。

```powershell
python -m main.retrieval.run_bm25 --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --n 3 --topM 50
```

## 4. 推理文档与查询的骨架

文档使用固定保留率：

```powershell
python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\2604-final\seed10101\def\stageC_skeleton\skeleton_last.pt --input_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --output_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --rho_mode fixed --rho 0.8 --device cuda --batch_size 128
```

查询使用预测保留率：

```powershell
python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\2604-final\seed10101\def\stageC_skeleton\skeleton_last.pt --input_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --output_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --rho_mode predict --device cuda --batch_size 128
```

## 5. 骨架检索

文档、查询及 `SimpleMono.pkl` 使用同一套语料的编码规则。此处采用 Jiugong 的词表文件。

```powershell
python -m main.retrieval.run_bm25 --docs_npy .\retrieval_exp\jiugong_skeleton\docs_skel.npy --queries_npy .\retrieval_exp\jiugong_skeleton\queries_skel.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --simplemono_pkl .\preproc\output\jiugong_test_for_retrieval\SimpleMono.pkl --n 3 --topM 50
```

## 6. 随机排序基线

```powershell
python -m main.retrieval.run_random_rank_baseline --docs_npy .\preproc\output\jiugong_test_for_retrieval\test.npy --queries_meta_jsonl .\retrieval_exp\jiugong_query_ornamented_corrupted\queries_meta.jsonl --queries_npy .\retrieval_exp\jiugong_query_ornamented_corrupted\queries.npy --topM 50
```

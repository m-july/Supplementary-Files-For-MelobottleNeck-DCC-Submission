# Supplementary Materials for MeloBottleneck

This repository provides the experiment code, technical documentation, evaluation records, prepared data, and qualitative demo for MeloBottleneck.

## Repository Contents

| Location | Contents |
| --- | --- |
| [melobottleneck-codebase/MeloBottleneck/](melobottleneck-codebase/MeloBottleneck/) | Source code, [requirements](melobottleneck-codebase/MeloBottleneck/requirements.txt), technical notes, data metadata and vocabulary files, and evaluation records. |
| [other-materials/output.rar](other-materials/output.rar) | Prepared SimpleMono corpus and benchmark data, including NumPy arrays, metadata, vocabulary files, and decoded MIDI files. Approximately 2.31 GiB unpacked. |
| [other-materials/retrieval_exp.rar](other-materials/retrieval_exp.rar) | Prepared retrieval queries, query metadata, and inferred document/query skeletons. Approximately 180 MiB unpacked. |
| [melobottleneck-audio-and-html-demo.zip](melobottleneck-audio-and-html-demo.zip) | A self-contained static webpage with audio and piano-roll examples. |

The unpacked sizes above refer to the total file contents of the supplied archives. Model checkpoint weights (`.pt` files) are not included. Evaluation records are stored under [ckpt/](melobottleneck-codebase/MeloBottleneck/ckpt/); the two RAR archives contain prepared data and retrieval intermediates.

## Getting Started

See the [codebase README](melobottleneck-codebase/MeloBottleneck/README.md) for environment setup, data preparation, training configuration, inference, and retrieval commands.

All experiment commands run from `melobottleneck-codebase/MeloBottleneck/`. When unpacking the supplied data:

- Extract `output.rar` into `melobottleneck-codebase/MeloBottleneck/preproc/`, preserving the archive's top-level `output/` folder.
- Extract `retrieval_exp.rar` into `melobottleneck-codebase/MeloBottleneck/`, preserving the archive's top-level `retrieval_exp/` folder.

Merge the archive folders with the corresponding repository folders. The codebase README includes command-line extraction examples and explains which workflows require model weights.

## Audio and Piano-Roll Demo

The [online demo](https://m-july.github.io/papers/melobottleneck/) presents four examples from both held-out and training samples. It compares MeloBottleneck with O2B-Learner through piano-roll visualizations and interactive audio playback, with extracted skeletons overlaid on the original melodies.

For local playback, extract `melobottleneck-audio-and-html-demo.zip` and open `index.html` inside the extracted folder. See the README inside the archive for local preview instructions.

## Model Checkpoints

Model weights are distributed separately because of their size. To request checkpoints, contact [m.july@qq.com](mailto:m.july@qq.com) and specify the model variant and seed you need. A suitable transfer method, such as Baidu Netdisk or Google Drive, can then be arranged.

The included evaluation records and [result summaries](melobottleneck-codebase/MeloBottleneck/results-summary.xlsx) can be inspected without checkpoint weights. Skeleton inference requires a compatible checkpoint; training from scratch requires the prepared corpus data and the configuration described in the codebase README.

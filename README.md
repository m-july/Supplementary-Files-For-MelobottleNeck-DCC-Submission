# Supplementary Materials for MeloBottleneck (DCC Submission Version)

This supplementary package mainly contains two archives.

## 1. melobottleneck-codebase

This archive contains the MeloBottleneck codebase, including:

- Complete source code and requirements.txt;
- Technical notes in Markdown format;
- Metadata and statistic reports for the training corpus and benchmark data;
- Evaluation result records for all experiments, including our method, baselines, and ablations.

## 2. melobottleneck-audio-and-html-demo.zip

This archive contains a static demo webpage for paired melody-skeleton playback. The online version is on [https://m-july.github.io/papers/melobottleneck/](https://m-july.github.io/papers/melobottleneck/)

The demo includes four examples, covering both held-out and in-training samples. It compares MeloBottleneck with the baseline method O2B-Learner. Each example provides piano-roll visualization and interactive audio playback, where the extracted skeleton is overlaid on the original melody. The demo is intended to help reviewers compare the skeleton extraction quality of different models.

To use the demo, please extract the archive and open index.html in a local web browser.

## Other Materials

All records of training, testing, and retrieval experiments are provided in the `./other-materials` folder.

The checkpoint files for all experiments are currently not involved, since they are too big (about 43.0 GB). You can contact `m.july@qq.com` and we will try to figure out how to send them to you (for example, Baidu Netdisk or Google Drive).
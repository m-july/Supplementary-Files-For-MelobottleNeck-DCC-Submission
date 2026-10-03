# Technical Notes (Chinese)

> **Terminology note for paper readers.**  
> This document intentionally uses the implementation-side terminology found in the code and code comments. Some names differ from the final paper terminology. Before reading this document together with the paper, please refer to [Terminology Notes: Paper vs. Code](Terminology%20Notes.md). For example, the paper's *Timeline Alignment Loss* $\mathcal L_{\mathrm{T}}$ corresponds to the implementation's guided-attention `loss_G`, and the paper's *Ornament Exclusion Loss* $\mathcal L_{\mathrm{E}}$ corresponds to the implementation's insertion `loss_I`. In this document, the code/technical-document terms are retained for easier cross-reference with the repository.

> Some parts of this document may be outdated. Please refer primarily to the data in actual code and in the paper.

# SimpleMono：OctupleMIDI-style 单声部三属性编码

**总体思想.** SimpleMono 继承 OctupleMIDI 的“多属性因子化（multi-attribute）”事件表示，但在结构上仅表示单个 voice 的音符序列；同时去除显式绝对时间（如 bar/position），改用相对时间增量（delta-time）来连接相邻音符。每个音符被编码为一个三元组事件 (pitch, duration, delta-time)，并在序列开头用 BOS 的第三属性提供一个（截断的）时间锚点。

### 1) 时间离散化（pos 网格）

设时间网格分辨率为 $R=12$，则 $1 \ \text{pos} = \frac{1}{R}\ \text{quarter note}$。输入（MIDI/MusicXML/KRN 等）首先被量化到整数 pos；并对每个 voice 进行 empty-measure trimming：删除该 voice 中“整小节均为休止”的小节区间，并将后续音符整体左移（时间压缩），从而减少长距离静默对序列建模的干扰。

### 2) 音符三元组（Pitch/Duration/Delta-Time）的数值语义与截断

对量化后的音符 (pitch, onset, offset) 序列 $\{(p_i,s_i,e_i)\}_{i=1}^N$，编码为 triples：

- **Pitch**：$p_i\in[0,127]$（MIDI pitch）。token 形如 `<0-p_i>`，一共是 `<0-0>`…`<0-127>` 共 128 个。
- **Duration**（均匀量化，pos 单位）：
    
    $$
    d_i=\max(1, e_i-s_i),\quad c^D_i=\min(d_i, N_D-1)
    $$
    
    其中 $N_D=96$，故 $c^D_i\in[0,95]$。token 形如 `<1-c^D_i>`，一共是 `<1-0>`…`<1-95>` 共 96个。
    
- **Delta-Time**（均匀量化，带符号，pos 单位）定义为「下一音 onset 相对当前音 offset 的差」：
    
    $$
    \Delta_i=
    \begin{cases}
    s_{i+1}-e_i,& i<N\\
    0,& i=N
    \end{cases},
    \quad
    c^\Delta_i=\mathrm{clip}(\Delta_i,-N_\Delta, N_\Delta-1)
    $$
    
- 其中 $N_\Delta=96$，故 $c^\Delta_i\in[-96,95]$。当 $c^\Delta_i<0$ 时自然表达音符重叠（next onset 早于 current offset）。token 形式形如 `<2-c^Delta_i>`（用感叹号表负号），一共是 `<2-0>`…`<2-95>` 加上 `<2-!1>`…`<2-!96>` 共 192 个。

> 额外的顺序处理：同一起音时刻的音符会先按 (start,end,pitch) 为 key 排序后，在组内做**确定性随机打乱**（按 piece+voice 派生种子），以减弱「和弦内部固定排序」偏置。这也使得同一 onset 的序列化通常产生负 $\Delta$。
> 
- 总词表大小为 421。
    - 音符事件使用 (pitch, duration, delta-time) 表示，即 $E_i = (\langle0-p_i\rangle,\langle1-c^D_i\rangle,\langle2-c^\Delta_i\rangle)$。
    - 共有 `<pad> <unk> <s> </s> <mask>` 5 个 special token，其中 BOS 事件将复用 delta-time 的量化器编码窗口第一个音符的 onset 时间，即 $E_0 = (\langle s\rangle,\langle s\rangle,\langle 2-\mathrm{clip}(s_1)\rangle)$。
- max token length: 514.

### 3) 多属性嵌入方案

- 使用类似于 MusicBERT 中的多属性拼接投影，对各 attribute 独立建表得到嵌入（$d_{\rm attr}=256$），拼接后通过线性投影 $W_{\uparrow}$ 到统一隐层空间（$d_{\rm model} = 512$）。
- 后续模型中出现的 token head 都会被 weight-tied to attribute embeddings，不过反向线性投影 $W_{\downarrow}$ 将使用独立的参数。

---

# MeloBottleneck: Ornament-Invariant Subsequence Bottleneck for Self-Supervised Melody Skeleton Extraction

- 采用「三路汇合」方案：Stage A 为预训练，Stage B 训练 LM Prior，Stage C 进行汇合训练。

## Stage A: **Seq2Seq** Denoising MusicBART Pretraining

- 目标：**学通用音乐表征与生成能力。**预训练一个通用的音乐 encoder-decoder backbone，让它具备：
    - 音乐 token 级建模能力
    - 对局部缺失、span 缺失、删除、旋转等 corruption 的恢复能力
    - 对简单音乐等价变换如移调、时间缩放的鲁棒性
- 架构：MusicBART-style seq2seq denoising model
    - Multi-attribute token embedding (🔥 from random init)
    - Backbone encoder (🔥 from random init)
    - Backbone decoder (🔥 from random init)
    - Multi-attribute LM head (tied embedding weight)
- 数据处理路径
    - 数据流程：训练样本先经过基础音乐增强 $\mathcal A$，再经过 BART-style corruption $\mathcal C$：
        
        $$
        x \xrightarrow{\mathcal A} x'
        \xrightarrow{\mathcal C} \tilde x
        $$
        
    - 模型学习：尝试从 $\tilde x$ 重建 $x'$。
    - Music Augmentation $\mathcal A$ 机制包括：
        - Pitch transpose：高斯采样半音偏移；
        - Time scaling：在 2.0 倍、1.0 倍、0.5 倍中采样时值缩放。
    - BART-Style Corruption $\mathcal C$ 机制：
        - Masking：支持粒度包括 `note@attribute`、 `note@full-token`、 `n-note@attribute`、 `n-note@full-token`；其中对 `n-note@full-token` 处理时，整个 span 被压成一个 [MASK]。
        - Deletion：支持粒度包括 `note@full-token`、 `note@full-token`。
        - Document Rotation。
    - Corruption curriculum
        - 课程包括：masking 从较弱开始；deletion 和 rotation 则更晚开启。
        - 课程按 step 平滑更新，而非按 epoch。
- 训练目标：Multi-attribute seq2seq cross-entropy
    - 对 pitch/duration/delta-time 三个属性分别做 CE，加权求和
        
        $$
        \mathcal L_\text{A}
        =
        \lambda_p^{\text{A}} \,\mathrm{CE}_{p}
        +
        \lambda_d^{\text{A}} \,\mathrm{CE}_{d}
        +
        \lambda_\Delta^{\text{A}}\,\mathrm{CE}_{\Delta}.
        $$
        
        - 典型值为 $(\lambda_p^{\text{A}}, \lambda_d^{\text{A}}, \lambda_\Delta^{\text{A}}) =(1/3, 1/3, 1/3)$。

## Stage B: **Frozen Decoder-Only** Music Prior

- 目标：**学一个冻结的自回归音乐先验。**不是为了继续提供初始化权重，而是训练一个后续冻结使用、用于提供语言先验的 decoder-only Music LM，供最终模型判断提取出来的骨干序列是否「像音乐」。
- 架构：Auto-regressive language model
    - Multi-attribute token embedding (🔥 from Stage A init)
    - Backbone decoder (🔥 from Stage A init)
    - Multi-attribute LM head (tied embedding weight)
- 数据处理路径
    - 数据流程：只有 Augmentation：
        
        $$
        x \xrightarrow{\mathcal A_{\text{LM}}} x'
        $$
        
    - 模型学习：将 $x'$ 作为语料训练自回归语言模型。
    - Augmentation 机制 $\mathcal A_{\text{LM}}$：类似于 $\mathcal A$，但压短时值更罕见，拉长时值更频繁。
- 训练目标：Next-token cross-entropy
    
    $$
    \mathcal L_{\text{B}}
    =
    \lambda_p^{\text{LM}} \,\mathrm{CE}_{p}
    +
    \lambda_d^{\text{LM}} \,\mathrm{CE}_{d}
    +
    \lambda_\Delta^{\text{LM}} \,\mathrm{CE}_{\Delta}
    $$
    
    - 典型值为 $(\lambda_p^{\text{LM}}, \lambda_d^{\text{LM}}, \lambda_\Delta^{\text{LM}}) =(0.5, 0.3, 0.2)$。

## Stage C: End-to-End Subsequence-bottleneck Training

- 目标：**端到端学出真正的 skeleton extractor。**通过 sequence bottleneck reconstruction 和 ornament invariance。
- 架构：
    - Compressor
        - Multi-attribute token embedding (🔥 from Stage A init)
        - Backbone encoder (🔥 from Stage A init)
        - Skeleton token salience score head (🔥 from random init)
        - Rho predictor (🔥 from random init)
    - Forward-extend postprocess
    - Reconstructor
        - Multi-attribute token embedding (shared with compressor)
        - Backbone encoder (shared with compressor)
        - Backbone decoder (🔥 from Stage A init)
        - Multi-attribute LM head (tied embedding weight)
    - LM prior (❄️ from Stage B)
- 数据增广：Augmentation $x \xrightarrow{\mathcal A} x'$。
- Compressor 架构：$x' \xrightarrow{\text{embed + encoder}} h
\xrightarrow{\text{score head}} \ell$，对每个 token 产生一个 salience score logit $\ell$。先使用 mean-pooled encoded memory $h$ 预测序列压缩率 $\rho$（$\rho_{\min}=1/3, \ \rho_{\max}=1$）并获得中间序列的连续长度 $T_{\text{cont}}$ 与硬长度 $T_{\text{hard}}$：
    
    $$
    \rho(x')
    =
    \rho_{\min}
    +
    (\rho_{\max}-\rho_{\min})\,
    \sigma\!\big(\mathrm{MLP}(\operatorname{Pool}(h))\big), \\
    
    T_{\text{cont}}
    = L_x \rho(x'),
    \quad
    T_{\text{hard}}
    =
    \left\lceil T_{\text{cont}}\right\rceil
    $$
    
    最终根据压缩率 $\rho(x')$ 作 salience Top-$T_{\text{hard}}$ 得到出若干个 token，得到严格保持原时序的压缩序列：
    
    $$
    z = \left (x_{i_1}', x_{i_2}', \dots, x_{i_K}', \text{EOS} \right), \quad i_1 < i_2 < \dots < i_K
    $$
    
    实现中，为了让连续长度 $T_{\text{cont}}$ 获得梯度，对后面 Reconstructor 的 bottleneck embedding 施加了软长度门 $\gamma_t$。
    
    $$
    \gamma_t
    =
    \sigma\!\left(
    \frac{T_{\text{cont}} - (t+0.5)}{T_\rho}
    \right),
    \qquad T_\rho = 0.5
    $$
    
- Forward-extend postprocess：目前得到的压缩序列 $z$ 还只是原序列 $x'$ 的抽样子序列，被称为 on-the-fly 序列 $z$，其 duration / delta-time 往往不再自洽。这个后处理算法把被删掉的装饰/经过音/切分音所占的时值前向并入前一个骨干音，使得输出的后处理序列 $\bar z$ 是一个节奏上自洽、可以单独成立的 skeleton melody。
- Straight-through bottleneck embedding：
    - 前向：使用 forward-extended hard $\bar z$ embedding；
    - 反向：使用 compressor 的 soft-gathered $z$ embedding。
    - Straight-through 形式如下：
        
        $$
        z^{\text{ST}}_t
        =
        \gamma_t (\operatorname{sg}\!\big(E(\bar z^{\text{hard}}_{t}) - E(z^{\text{soft}}_{t})\big)
        +
        E(z^{\text{soft}}_{t}))
        $$
        
        其中 $\operatorname{sg}(\cdot)$ 表示 stop-gradient。
        
- Reconstructor：通过 Seq2Seq 结构从 $z^{\text{ST}}$ 重建原序列 $x'$，相当于尝试学习如何加花从骨干序列还原原序列。
    - Reconstructor compression conditioning：对样本压缩率做 FiLM 调制，实现将压缩率条件注入到 bottleneck embeddings 中。
- 核心训练目标：
    
    $$
    \mathcal J
    =
    \lambda_{\text{R}} \mathcal L_{\text{R}}
    +
    \lambda_{\text{P}} \mathcal L_{\text{P}}
    +
    \lambda_{\text{L}} \mathcal L_{\text{L}}
    +
    \lambda_{\text{G}} \mathcal L_{\text{G}}
    +
    \lambda_{\text{C}} \mathcal L_{\text{C}}
    +
    \lambda_{\text{I}} \mathcal L_{\text{I}}
    $$
    
    - Reconstruction loss $\mathcal L_{\text{R}}$
        - 目标：要求压缩所得骨干序列 $\bar z$ 能保留关于原始序列 $x'$ 的足够结构信息。
        - 定义：衡量 Reconstructor 重建 $x'$ 的 duration-weighted multi-attribute CE。
            
            $$
            \mathcal L_{\text{R}}
            =
            \frac{
            \sum_l d_l
            \left(
            \lambda_p^{\text{R}} \mathrm{CE}_{p,l}
            +
            \lambda_d^{\text{R}} \mathrm{CE}_{d,l}
            +
            \lambda_\Delta^{\text{R}} \mathrm{CE}_{\Delta,l}
            \right)
            }{
            \sum_l d_l
            }
            $$
            
            - 典型值为 $(\lambda_p^{\text{R}}, \lambda_d^{\text{R}}, \lambda_\Delta^{\text{R}}) =(1/3, 1/3, 1/3)$。
        - Decoder regularization: Decoder-input mask dropout with a course warming up from 0% masked (fully teacher forcing) to 80% masked。这迫使 Reconstructor decoder 更多依赖 encoder 所传来的 bottleneck sequence 相关的信息，而不是只靠自回归惯性。
    - Frozen LM prior loss $\mathcal L_{\text{P}}$
        - 目标：要求中间序列 $\bar z$ 像一条自然的音乐序列。
        - 定义：中间序列 $\bar z$ 遵循 Stage B 训练得到的 frozen LM prior 模型的程度。具体而言，根据 compressor 在每个采样 step 上的 soft selection 分布，为每个 token $\bar z_t$ 诱导出对应的属性分布 $r_t^p, r_t^d, r_t^\Delta$ 后，对其与当前步在 frozen LM prior 中的 next-token 分布之间作 KL 散度。
            
            $$
            \mathcal L_{\text{P}}
            =
            \sum_t
            \Big[
            \alpha_p \,\mathrm{KL}(r_t^p \,\|\, p_{\text{LM}}^p(\cdot \mid \bar z_{<t}))
            +
            \alpha_d \,\mathrm{KL}(r_t^d \,\|\, p_{\text{LM}}^d(\cdot \mid \bar z_{<t}))
            +
            \alpha_\Delta \,\mathrm{KL}(r_t^\Delta \,\|\, p_{\text{LM}}^\Delta(\cdot \mid \bar z_{<t}))
            \Big]
            $$
            
            - 典型值为 $(\alpha_p,\alpha_d,\alpha_\Delta)=(0.7,0.2,0.1)$。
    - Length regularization loss $\mathcal L_{\text{L}}$
        - 目标：要求模型能够主动应对多种动态压缩率。
        - 定义：让 batch 内预测出的 $\rho$ 分布去匹配一个截断高斯目标分布。令 batch 内预测得到的有效压缩率排序为 $\rho^{\text{eff}}_{(1)} \le \rho^{\text{eff}}_{(2)} \le \cdots \le \rho^{\text{eff}}_{(B)}$，构造 clipped normal target quantiles
            
            $$
            u_i = \frac{i-0.5}{B},
            \qquad
            q_i = \operatorname{clip}\big(\mu_{\text{L}} + \sigma_{\text{L}} \Phi^{-1}(u_i),\, \rho_{\min},\, \rho_{\max}\big)
            $$
            
            然后长度正则为
            
            $$
            \mathcal L_{\text{L}}
            = \sum_{i=1}^B 
            (\rho^{\text{eff}}_{(i)} - q_i)^2
            $$
            
            - 典型值为 $\mu_{\text{L}} = 0.67$、$\sigma_{\text{L}} = 0.22$。
    - Guided attention loss $\mathcal L_{\text{G}}$
        - 目标：为了避免训练早期的 skeleton 选择塌缩到局部区域，当前实现还加入一个基于归一化乐谱时间的 guided attention loss。
        - 定义：将 $\bar z$ 序列的 onset 归一化为 $u_l$，乐谱时间线性进度归一化为 $v_t$，衡量这两个归一化时间之间的 quadratic loss
            
            $$
            \mathcal L_{\text{G}}
            =
            \frac{1}{\sum_t m_t}
            \sum_t m_t \sum_l p_t(l) \cdot \frac{(u_l-v_t)^2}{2\sigma^2_{\text{G}}}, \\
            
            u_l = \frac{o_l}{o_{\text{EOS}}}, \quad v_t = \frac{t}{z_{\text{len}} - 1}
            $$
            
            - 典型值为 $\sigma_{\text{G}} = 0.075$。
        - curriculum：这个 loss 只施加于 early-stage，后期 $\lambda_G$ 将衰减到 0。
- Ornament invariance
    - 目标：让骨干序列的抽取对序列是否加花具有不变性，同时惩罚对装饰音的选择。
    - 数据流：
        - 弱视图：上面得到的 $x'$
        - 加花强视图：$x' \xrightarrow{\mathcal O} (x^{({\text{orn}})}, y)$ 得到的 $x^{({\text{orn}})}$。
        - Music Ornamenter $\mathcal O$ 机制：通过一套算法，对原序列 $x'$ 在线生成添加装饰音后的 $x^{(\mathrm{orn})}$，并行产出骨干音标签序列 $y$，以说明各个 ornamented token 是骨干音（$y_k=1$）还是插入的装饰音（$y_k=0$）。
            
            大部分序列上的加花将保持总时值不变，少部分序列上的加花将增长总时值，使曲目变为散板节奏。对于 valid/test O2B benchmark 上将使用 Out-of-Distribution (OOD) 分布 $\mathcal O_{\text{OOD}}$，这将启用训练中未曾见过的部分机制，以及对部分机制使用训练范围以外的参数。
            
    - Teacher-student 机制
        - Teacher 在弱视图 $x'$ 上通过 eval-mode compressor 提取 salience logit，再 softmax 得到分布 $s(x')$。
        - Student 在加花强视图 $x^{({\text{orn}})}$ 上，强制使用同一压缩率预算提取 salience logit，再 softmax 得到分布 $s(x^{({\text{orn}})})$。
    - Loss 构成：
        - Consistency loss $\mathcal L_{\text{C}}$
            - 目标：让弱视图和加花强视图得到的 token salience 分布一致。
            - 定义：teacher 和 student 得到的 token salience 分布的 KL 散度。
                
                $$
                \mathcal L_{\text{C}} = {\rm KL} (s(x') \| s(x^{({\text{orn}})}))
                $$
                
        - Insertion loss $\mathcal L_{\text{I}}$
            - 目标：明确加花强视图中的装饰音不该进入骨干。
            - 定义：loss 的大小就是 student 在插入装饰音上的 softmax salience 加和。
                
                $$
                \mathcal L_{\text{I}} = \sum_{k:y_k=0} s_k(x^{({\text{orn}})})
                $$
                

## Appendix: Music Ornamenter Behaviour

- Music Ornamenter ($\mathcal O$ & $\mathcal O_{\text{OOD}}$) 机制包括：
    - pre-grace：将长音分裂为前短后长两个音，标记长音为骨干音。
    - post-grace：将长音分裂为前长后短两个音，标记长音为骨干音。
    - between insert：在相邻两个原始音之间插入 1 ~ 4 个经过音，标记两个原始音为骨干音。
    - trill：把长音分裂为 3 段或 5 段，奇偶交替主音和 pitch-jitter 所得邻音，标记首音是骨干音。
    - turn (OOD-only)：把长音分裂为 3 段（`[主, 上邻, 主]`）或 4 段（`[主, 上邻, 主, 下邻]`）模式排列，标记首音是骨干音。
    - rearticulation：将长音拆成 2~4 次重复发音，标记首音是骨干音。
    - pair-repeat (OOD-only)：把一对音拆成 2~4 轮交替重复，标记第一、二个音是骨干音。

## Appendix: Parameters

- Encoder-decoder backbone
    - `max_seq_len = 514`
    - `d_attr = 256`
    - `d_model = 512`
    - encoder / decoder layers = `4 / 4`
    - attention heads = `8`
    - FFN dim = `2048`
- Music Augmentation $\mathcal A$ & $\mathcal A_{\text{LM}}$
    - $\mathcal A$
        - Pitch transpose：`σ=6`, range `[-12,12]`
        - Time scaling：`p(2x)=0.30`, `p(0.5x)=0.30`
    - $\mathcal A_{\text{LM}}$
        - Pitch transpose：`σ=6`, range `[-12,12]`
        - Time scaling：`p(2x)=0.50`, `p(0.5x)=0.20`
- BART-Style Corruption $\mathcal C$ (after curriculum)
    - masking noise density: `0.333`
    - deletion prob: `0.2`
    - masking & deletion 将从相应的粒度类型均匀采样
    - masking & deletion n-note span lambda: `3.0`
    - rotation prob: `0.5`
- Parameter of Stages
    
    | Stage | A | B | C | O2B-Learner |
    | --- | --- | --- | --- | --- |
    | #epoch | 100 | 80 | 1 | 5 |
    | batch size | 160 | 200 | 28 | 256 |
    | learning rate | 5e-4 | 4e-4 | 3e-4 & 4e-4* | 5e-4 |
    | dropout | 0.15 | 0.15 | 0.075 | 0.10 |

    \*Stage C learning-rate warmup: LR linearly increases from 5% to 100% over 30% of the steps, then follows cosine decay to 5%; `lr_backbone=3e-4`, `lr_pointer=4e-4`.
    

---

# 数据集 / Benchmark

### 主数据集：世界民间歌曲大杂烩

| 名称 | 地域 | 节选范围 |
| --- | --- | --- |
| Anthology of Chinese Folk Songs | Chinese | All |
| Jiugong Dacheng | Chinese | Train Split |
| BFDB: A dataset of British Folk melodies in ABC Format | British | All |
| Essen’s Folksong | World-wide | All |
| Henrik Norbeck's ABC Tunes | Irish & Swedish | All |
| IrishMAN | Irish | Train Split #1 ~ #7999 |
| MTC-FS-INST-2.0 | Dutch | All |
- 总曲目数目：58,154 曲目，对曲目滑窗得到一个或多个序列。Set Split on File Level: Train:Valid:Test = 18:1:1。
    
    
    | seed | split | 序列数 | 音符数 | 时长@80BPM (s) | 时长@80BPM |
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
- Train Set 将被用于所有阶段的训练

### Synthetic Benchmark: Main Ornament-to-Backbone (Main-O2B)

- 用于 Validation 和 Test。
- 数据来源：
    - Valid Set: 使用主数据集的 Valid Split
    - Test Set: 使用主数据集的 Test Split
- Benchmark 构建：
    - 对于序列 $x$，使用 OOD Music Ornament 得到加花后序列 $x_{\rm orn}$（约束压缩率在 $[1/3, 1]$）。
    - 构成 evaluation 对 $x_{\rm orn} \to x$（评估时，模型固定压缩率 $\rho = {\rm len} (x)/ {\rm len} (x_{\rm orn})$）。
- 序列数目：2,593。

### Zero-shot Cross-domain Benchmark: TAVERN Variation-to-Theme (TAVERN-V2T)

- 仅用于 Test。
- 数据来源：「TAVERN: A New Data Set for Symbolic Music Analysis」中的 Variation-Theme 乐曲对。数据集为 Mozart and Beethoven 古典钢琴谱 (Cross-domain)，我们只使用右手的主旋律部分。
- Benchmark 构建：
    - 对于每一对数据集中的 variant $x$ 和 theme $y$，对它们作 Needleman-Wunsch 对齐（允许 Match、Insertion、Deletion 三种操作），从而得到 Variation 中被最优对齐判定为「对应 Theme」的子序列 $x'$。匹配分数低的数据对会被丢弃 (原始语料中 9% 的数据对被丢弃)。
    - 构成 evaluation 对 $x \to x'$（评估时，模型固定压缩率 $\rho = {\rm len} (x')/ {\rm len} (x)$）。
- 序列数目：591。

### Zero-shot In-domain Benchmark: Jiugong Dacheng Ornamented-to-Gongche (Jiugong-O2G)

- 仅用于 Test。
- 数据来源：「九宫大成南北词宫谱」是一部清代戏曲和宫廷音乐乐谱集，以工尺谱记谱法的形式记录了旋律骨干。王正来《新定九宫大成南北词宫谱译注》将其翻译为现代简谱记谱法的同时，根据工尺谱实际表演的经验添加了一系列装饰音。
- Benchmark 构建：
    - 对于每一对工尺骨干旋律 $x$ 和简谱翻译版本 $y$，将 $y$ 手工输入为 MIDI 形式并标注对齐 $x$ 的子序列 $y'$。
    - 构成 evaluation 对 $y \to y'$（评估时，模型固定压缩率 $\rho = {\rm len} (y')/ {\rm len} (y)$）。
- 附录说明 Jiugong 的切分方案：Main Train Set 中的 Jiugong 是除去了 Volume 2, 51 and 71 的其余卷目所有曲目；Retrieval 中的 Jiugong 是 Volume 2, 51 and 71；Jiugong-O2G 是从 Volume 2, 51 and 71 中节选了 * 首并人工录入。

# Baselines and Ablations

### Naive Baselines

- Random downsampling（按压缩率随机等距采样）
    - soft score：在被选中的 token（含 BOS、不含 EOS）上均匀分摊，和为 1。
- Uniform downsampling（按压缩率在归一化乐谱时间上等距采样）
    - soft score：在被选中的 token（含 BOS、不含 EOS）上均匀分摊，和为 1。
- Top-K duration（按压缩率取最长的那些音）
    - soft score：按 duration 加权的均匀分摊。

### Heuristic Baseline: AMR-No-Harmony

- Automatic Melody Reduction via Shortest Path Finding (AMR) 是一个启发式算法，其把旋律看成一个带权有向图，把 Music Reduction 问题转化为「从首音到尾音的最短路径问题」。
- 原版算法依赖同时输入的和声数据。为了适配我们仅提供旋律的 Benchmark，我们修改算法得到一个适配版 AMR-No-Harmony：
    - 去除和声依赖（去掉相关边权）
    - 增加长度控制（能对任意指定压缩率生成结果）
    - 自由首尾（修正了原版强制选中首音、尾音的问题）

### Pseudo-label Classifier: O2B-Learner

- 目标：使用 encoder 从 Music Ornamenter 生成的 O2B 伪标签学「骨干 vs 装饰」二分类。
- 架构：
    - Multi-attribute token embedding (🔥 from random init)
    - Backbone encoder (🔥 from random init)
    - Skeleton token salience score head (🔥 from random init)
- 数据处理路径
    - 数据流程：先经过 Augmentation $\mathcal A$，再经过 Ornamenter $\mathcal O$。
        
        $$
        x \xrightarrow{\mathcal A} x'
        \xrightarrow{\mathcal O} (x^{(\mathrm{orn})}, y)
        $$
        
    - 模型工作：$x^{(\mathrm{orn})} \xrightarrow{\text{embed + encoder}} h
    \xrightarrow{\text{score head}} \ell$。这里的 score head 通过一个两层 MLP（$d_{\text {middle}}=256$）为序列中每个 note token 提取一个 salience score logit，表示每个 note 在骨干意义上的 salience (skeletal importance)。
- 训练目标：逐 note-token（定义 note mask 为 $m_k$）二分类损失
    
    $$
    \mathcal L_{\text{O2B-L}}
    =
    \frac{1}{\sum_k m_k}
    \sum_k
    m_k \,\omega(y_k)\,
    \mathrm{BCEWithLogits}(\ell_k, y_k)
    $$
    
    - 其中会对 negative 类（装饰音类）做动态增权：
        
        $$
        \omega(0)
        =
        \operatorname{clip}\!\left(\frac{N_{\text{pos}}}{N_{\text{neg}}},\, 1,\, 10\right),
        \qquad
        \omega(1)=1
        $$
        

### Ablations

- 阶段消融（共 1 组）：去掉 BART-denoise init 实现 1 组消融。
- Loss 消融（共 6 组）：分别将 $\lambda_{\text{R}}, \lambda_{\text{P}}, \lambda_{\text{L}}, \lambda_{\text{G}}, \lambda_{\text{C}}, \lambda_{\text{I}}$ 置零实现 6 组消融。

# 评估指标

### Benchmark Evaluation Metrics

> 这一族指标用于衡量骨干音提取的质量。
> 
- Hard-set F1 Score (Hard F1)：用于衡量端到端骨干音提取的质量。该分数为模型最终输出的离散骨干音集合与 ground-truth 骨干音集合之间的 token-level F1。越大越好。
- Soft Average Precision (Soft AP)：用于衡量模型对骨干音的排序质量。该分数为模型输出的逐 token salience score 与 ground-truth selection 之间的二分类 Average Precision (AP)。越大越好。
- Normalized F1 AUC over Cut Ratios (Cut F1 AUC-N)：用于衡量模型在不同压缩率下的整体 F1 质量。对一系列 cut 压缩率（top‑k，k 随 cut ratio 变化）计算对应的 F1，并对 F1-cut 曲线做梯形积分得到 AUC，随后按 cut 区间长度归一化得到该分数。越大越好。
- Insertion Mass：用于衡量模型避免选择装饰音的能力。将模型对序列输出的所有 token scores 归一化为质量分布，累加其在 ground-truth 装饰音上的概率质量得到该分数。越小越好。

### Music Prior Proxy Lifts

> 这一族指标用于衡量骨干音提取是否符合一组启发式的音乐先验。
> 
- 计算方案（置于附录）：对任意 feature $f_l\ge 0$（只在 note tokens 上定义），计算「salience 加权期望」与「均匀基线期望」之比作为 Lift 分数，大于 1 则认为分数对该 feature 具有正向偏好。
    
    $$
    \mathrm{Lift}(f) = \frac{\mathbb{E}_{\tilde{s}}[f]}{\mathbb{E}_{U}[f]}, \quad \mathbb{E}_{\tilde{s}}[f] = \sum_{l\in\mathcal{N}} \tilde{s}_l f_l, \quad \mathbb{E}_{U}[f] = \frac{1}{|\mathcal{N}|}\sum_{l\in\mathcal{N}} f_l
    $$
    
- Lift Strong：衡量骨干音提取模型对强拍（每两拍为一个强拍）上的音的偏好性。
- Lift Duration：衡量骨干音提取模型对长时值的音的偏好性。

### 辅助指标

> 只对最后 N 个 batch 启用，表明训练重建的质量。
> 
- weighted geometric mean perplexity
    - 对最后 N 个 batch 的平均 attr CE $\bar{L}_p,\ \bar{L}_d,\ \bar{L}_\Delta$ 定义
        
        $$
        \text{PPL}_p = e^{\bar{L}_p},\quad
        \text{PPL}_d = e^{\bar{L}_d},\quad
        \text{PPL}_\Delta = e^{\bar{L}_\Delta}
        $$
        
    - 用 归一化后的 attr 权重组合得到
        
        $$
        \text{PPL}_{\text{recon}}=\exp\!\left(\frac{\lambda_p \bar{L}_p + \lambda_d \bar{L}_d + \lambda_\Delta \bar{L}_\Delta}{\lambda_p + \lambda_d + \lambda_\Delta}\right)
        $$
        
- Bottleneck Dependence Ratio
    - 在 teacher-forced 情况下，计算：
        - $\mathcal L_{\text{rec}}^{\text{TF}}(z)$：在 正常 bottleneck 条件下的 reconstruction loss；
        - $\mathcal L_{\text{rec}}^{\text{TF}}(\varnothing)$：在 去掉 bottleneck 条件后得到的 reconstruction loss。实现上将 bottleneck 输入替换为 dummy memory。
    - 得到：
        
        $$
        \mathrm{TF\text{-}BDR} = \frac{\mathcal L_{\text{rec}}^{\text{TF}}(\varnothing)}
        {\mathcal L_{\text{rec}}^{\text{TF}}(z)}
        $$

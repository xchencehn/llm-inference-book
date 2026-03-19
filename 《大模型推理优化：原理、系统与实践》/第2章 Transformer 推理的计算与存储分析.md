# 第2章 Transformer 推理的计算与存储分析

推理优化的第一步，是精确理解推理过程中"算了什么"和"存了什么"。本章将从 Transformer 的基本架构出发，逐层分析推理的计算流程，建立 FLOPs 和显存占用的定量估算框架，并借助 Roofline 模型揭示 Prefill 与 Decode 两阶段截然不同的性能瓶颈本质。最后，本章将专门讨论 MoE（Mixture of Experts）架构在推理中引入的特殊性，为后续章节中 KTransformers 的异构推理方案和 vLLM/SGLang 的专家并行策略提供分析基础。

---

## 2.1 Transformer 架构回顾：Attention、FFN、LayerNorm

自 Vaswani 等人于 2017 年提出 Transformer 架构以来，几乎所有主流大语言模型——从 GPT 系列、LLaMA 系列到 DeepSeek 系列——都建立在这一架构之上。尽管具体实现细节随模型迭代不断演进，Transformer Decoder 的核心组件保持了高度一致：自注意力机制（Self-Attention）、前馈网络（Feed-Forward Network, FFN）以及归一化层（Normalization Layer）。本节回顾这些基础组件的结构与数学定义，为后续的计算和存储分析奠定符号基础。

**自注意力机制（Self-Attention）。** 自注意力是 Transformer 架构的核心计算单元。给定输入序列的隐藏表示 $\mathbf{X} \in \mathbb{R}^{s \times d}$（其中 $s$ 为序列长度，$d$ 为隐藏维度），自注意力机制首先通过三组线性投影将输入映射为查询（Query）、键（Key）和值（Value）三组矩阵：

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

其中 $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$，$\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$。在标准的多头注意力（Multi-Head Attention, MHA）中，$d_k = d_v = d / n_h$，其中 $n_h$ 为注意力头数。每个头独立计算注意力：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

各头的输出拼接后，经过一个输出投影矩阵 $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ 映射回原始维度。整个多头注意力的计算可以概括为四个核心步骤：QKV 投影、注意力分数计算（$\mathbf{Q}\mathbf{K}^\top$）、加权聚合（乘以 $\mathbf{V}$）、以及输出投影。

在实际的大语言模型中，自注意力机制还包含因果掩码（causal mask），确保每个位置只能关注到它自身及之前的 Token，这是自回归生成的基本约束。此外，位置编码（如 RoPE，Rotary Position Embedding）也通常在注意力计算过程中融入 $\mathbf{Q}$ 和 $\mathbf{K}$，但其计算量相对于矩阵乘法来说较小。

需要指出的是，现代大模型广泛采用了多头注意力的变体。Multi-Query Attention（MQA）让所有头共享同一组 $\mathbf{K}$ 和 $\mathbf{V}$，Grouped-Query Attention（GQA）则将头分为若干组、组内共享 $\mathbf{K}$ 和 $\mathbf{V}$，这两种变体主要目的是减少 KV Cache 的显存占用（详见第4章和第5章）。DeepSeek 系列模型引入的 Multi-head Latent Attention（MLA）则通过低秩压缩进一步减少 KV 的存储需求。这些变体不改变注意力机制的核心计算逻辑，但会显著影响推理过程中的存储特性，本章的分析将以标准 MHA 为起点，在需要时指出变体带来的差异。

**前馈网络（FFN）。** Transformer 每一层中，注意力子层之后紧跟一个前馈网络。经典 Transformer 的 FFN 由两个线性变换和一个激活函数组成：

$$\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$，$\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$，$d_{\text{ff}}$ 是 FFN 的中间维度，通常取 $d_{\text{ff}} = 4d$。$\sigma$ 为激活函数，早期模型使用 ReLU，现代模型更常使用 GeLU 或 SiLU（也称为 Swish）。

当代大语言模型（如 LLaMA、DeepSeek 等）广泛采用 SwiGLU（Swish-Gated Linear Unit）变体，将 FFN 改为门控形式：

$$\text{FFN}_{\text{SwiGLU}}(\mathbf{x}) = (\text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}}))\mathbf{W}_{\text{down}}$$

其中 $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$，$\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$，$\odot$ 表示逐元素乘法。相比经典 FFN 的两个矩阵，SwiGLU 引入了三个矩阵（gate、up、down），参数量增加了约 50%。为保持总参数量可比，实践中通常将 $d_{\text{ff}}$ 相应调小（例如 LLaMA 系列取 $d_{\text{ff}} = \frac{8}{3}d$ 再上取整到某个便于计算的值）。

从推理计算的角度来看，FFN 是"参数密集"的组件：它不涉及序列维度上的 Token 间交互（每个 Token 独立通过 FFN），但包含大规模的矩阵乘法。在标准 Transformer 中，FFN 的参数量通常占据模型总参数量的约三分之二。

**归一化层（Normalization Layer）。** 归一化层在 Transformer 中起到稳定训练和推理数值的作用。经典 Transformer 使用 Post-Norm（在子层之后归一化），现代大模型几乎一致采用 Pre-Norm（在子层之前归一化），即 Layer Normalization 置于注意力和 FFN 之前：

$$\mathbf{x}’ = \text{LayerNorm}(\mathbf{x}), \quad \mathbf{x}_{\text{out}} = \mathbf{x} + \text{SubLayer}(\mathbf{x}')$$

其中 LayerNorm 的计算为：

$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

许多现代模型进一步简化为 RMSNorm（Root Mean Square Layer Normalization），省去了均值中心化步骤：

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

RMSNorm 的参数仅有缩放向量 $\boldsymbol{\gamma} \in \mathbb{R}^d$，没有偏置项 $\boldsymbol{\beta}$。从推理的角度看，归一化层的计算量和参数量相对于注意力和 FFN 来说都非常小（每层仅有 $O(d)$ 级别的参数和计算），在性能分析中往往可以忽略，但在工程实现中，归一化操作与残差加法的融合（fused kernel）是常见的算子优化点（见第16章）。

**残差连接（Residual Connection）。** 每个子层（注意力、FFN）都配有残差连接，将子层的输入直接加到子层的输出上。残差连接本身仅是逐元素加法，计算量可忽略不计，但它决定了推理过程中激活值的数据流向，对显存分析有间接影响。

**综合来看**，一个标准的 Transformer Decoder 层包含以下计算序列：

$$\mathbf{h} \leftarrow \mathbf{h} + \text{Attention}(\text{Norm}(\mathbf{h}))$$  
$$\mathbf{h} \leftarrow \mathbf{h} + \text{FFN}(\text{Norm}(\mathbf{h}))$$

整个模型由 $L$ 个这样的层堆叠而成，输入端有词嵌入层（Embedding Layer），输出端有语言模型头（LM Head，通常与词嵌入共享权重或单独一个线性层）。后续的分析将反复回到这一基本结构。

---

## 2.2 推理过程的逐层计算流

在第1章中我们已经建立了推理两阶段（Prefill 与 Decode）的宏观理解，本节将深入到模型内部，逐算子追踪一次完整推理的计算流程，为后续的 FLOPs 估算和显存分析做准备。

**Prefill 阶段的逐层计算流。** 假设用户输入的提示（Prompt）经过 Tokenizer 编码后得到 $s_p$ 个 Token。Prefill 阶段需要将这 $s_p$ 个 Token 一次性送入模型，完成所有层的前向计算，并生成第一个输出 Token。

对于第 $l$ 层（$l = 1, 2, \ldots, L$），输入为 $\mathbf{H}^{(l)} \in \mathbb{R}^{s_p \times d}$（所有 Token 的隐藏状态），计算流程如下：

第一步，Pre-Norm：$\mathbf{H}’ = \text{RMSNorm}(\mathbf{H}^{(l)})$。这是一个按行（每个 Token 独立）的归一化操作，访存密集但计算量小。

第二步，QKV 投影：$\mathbf{Q} = \mathbf{H}‘\mathbf{W}_Q$，$\mathbf{K} = \mathbf{H}’\mathbf{W}_K$，$\mathbf{V} = \mathbf{H}'\mathbf{W}_V$。这是三次矩阵乘法（在实现中通常将 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ 合并为一个大矩阵 $\mathbf{W}_{QKV}$ 做一次 GEMM），形状为 $[s_p, d] \times [d, 3d] = [s_p, 3d]$（以标准 MHA 为例）。

第三步，RoPE 位置编码：对 $\mathbf{Q}$ 和 $\mathbf{K}$ 施加旋转位置编码。这是逐元素操作，计算量为 $O(s_p \cdot d)$，相对于矩阵乘法可忽略。

第四步，存储 KV Cache：将当前层计算得到的 $\mathbf{K}$ 和 $\mathbf{V}$ 存储到 KV Cache 中，以供后续 Decode 阶段使用。这一步的关键开销不是计算，而是显存写入。

第五步，注意力分数计算：$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$。这是一个 $[s_p, d_k] \times [d_k, s_p] = [s_p, s_p]$ 的矩阵乘法（每个头独立计算），生成 $s_p \times s_p$ 的注意力分数矩阵。此处施加因果掩码，将上三角部分置为 $-\infty$。

第六步，Softmax：对 $\mathbf{S}$ 的每一行做 Softmax 归一化，得到注意力权重 $\mathbf{A}$。

第七步，加权聚合：$\mathbf{O}_{\text{attn}} = \mathbf{A}\mathbf{V}$，形状为 $[s_p, s_p] \times [s_p, d_v] = [s_p, d_v]$。

第八步，输出投影：$\mathbf{O} = \text{Concat}(\mathbf{O}_{\text{attn}}^{(1)}, \ldots, \mathbf{O}_{\text{attn}}^{(n_h)})\mathbf{W}_O$，将多头输出拼接后投影，矩阵乘法形状为 $[s_p, d] \times [d, d]$。

第九步，残差连接：$\mathbf{H}^{(l)}_{\text{mid}} = \mathbf{H}^{(l)} + \mathbf{O}$。

第十步，FFN Pre-Norm：$\mathbf{H}‘’ = \text{RMSNorm}(\mathbf{H}^{(l)}_{\text{mid}})$。

第十一步，FFN 计算（以 SwiGLU 为例）：$\mathbf{G} = \text{SiLU}(\mathbf{H}‘’\mathbf{W}_{\text{gate}})$，$\mathbf{U} = \mathbf{H}‘’\mathbf{W}_{\text{up}}$，$\mathbf{F} = (\mathbf{G} \odot \mathbf{U})\mathbf{W}_{\text{down}}$。包含三次矩阵乘法和一次逐元素乘法。

第十二步，残差连接：$\mathbf{H}^{(l+1)} = \mathbf{H}^{(l)}_{\text{mid}} + \mathbf{F}$。

以上流程在所有 $L$ 层中依次执行后，最终层的输出 $\mathbf{H}^{(L+1)}$ 经过一次 RMSNorm 和 LM Head 线性层（$[d, V]$，其中 $V$ 为词表大小），得到最后一个位置的 logits，再通过采样策略（如 top-p, top-k, temperature scaling）选出第一个输出 Token。

值得注意的是，在 Prefill 阶段，所有操作涉及的矩阵维度中，序列长度 $s_p$ 同时出现在矩阵乘法的多个维度上（batch 维或序列维），因此矩阵乘法的操作数可以保持较大规模，使得 GPU 的 Tensor Core 能够高效利用。这是 Prefill 阶段呈现计算密集特性的微观原因。

**Decode 阶段的逐层计算流。** 在 Decode 阶段，模型每一步仅处理一个新 Token（或者在批处理场景下，处理 batch 中每个请求的一个新 Token）。假设当前已经生成了前 $t-1$ 个 Token，现在要生成第 $t$ 个 Token。

对于第 $l$ 层，输入仅为当前 Token 的隐藏状态 $\mathbf{h}^{(l)} \in \mathbb{R}^{1 \times d}$（一个行向量），计算流程在结构上与 Prefill 完全相同，但矩阵维度发生了根本性变化：

QKV 投影变为 $[1, d] \times [d, 3d] = [1, 3d]$，即矩阵-向量乘法（GEMV），而非矩阵-矩阵乘法（GEMM）。注意力分数计算变为 $[1, d_k] \times [d_k, t] = [1, t]$（Query 只有一个 Token，但 Key 来自长度为 $t$ 的 KV Cache），加权聚合也类似。FFN 的三次矩阵乘法同样变为 GEMV。

这一维度变化的核心后果是：Decode 阶段的每一个矩阵运算的"计算量"相对于"需要读取的权重数据量"来说非常小。以 QKV 投影为例，计算量为 $O(d^2)$（或更精确地说，$6d^2$ 次浮点运算，对应三个 $d \times d$ 的矩阵与一个长度为 $d$ 的向量相乘），但需要从显存中读取 $3d^2$ 个参数。这意味着每读取一个参数仅做约 2 次浮点运算，远低于 GPU 的计算-访存比（以 A100 为例约为 312 TFLOPS / 2 TB/s ≈ 156 FLOPs/Byte），使得 Decode 阶段严重受限于显存带宽而非计算能力。这正是 Decode 阶段"访存密集型"特性的微观根源，我们将在 2.5 节通过 Roofline 模型更严格地论证这一点。

**KV Cache 的作用可以从逐层计算流中直观理解。** 在 Prefill 阶段，我们一次性计算并存储了所有 $s_p$ 个 Token 在每一层的 Key 和 Value。到 Decode 第 $t$ 步时，注意力计算需要用到从第 1 个 Token 到第 $t$ 个 Token 的全部 Key 和 Value，其中前 $t-1$ 个来自 KV Cache（无需重新计算），只有第 $t$ 个 Token 的 Key 和 Value 需要新计算并追加到 Cache 中。如果没有 KV Cache，每生成一个新 Token 就需要重新计算前面所有 Token 的 Key 和 Value，这将使 Decode 阶段的计算量从 $O(t)$ 变为 $O(t^2)$（在序列长度维度上），而在生成长序列时这是不可接受的。KV Cache 本质上是一种"用空间换时间"的策略，但它引入了显存管理的巨大挑战（详见第5章）。

---

## 2.3 计算复杂度分析：FLOPs 估算方法

准确估算推理过程的浮点运算量（FLOPs），是理解推理性能瓶颈、指导优化决策的基础。本节建立系统的 FLOPs 估算方法。

**矩阵乘法的 FLOPs 计算规则。** 两个矩阵 $\mathbf{A} \in \mathbb{R}^{m \times k}$ 和 $\mathbf{B} \in \mathbb{R}^{k \times n}$ 的乘法 $\mathbf{C} = \mathbf{A}\mathbf{B}$，结果矩阵 $\mathbf{C}$ 有 $m \times n$ 个元素，每个元素需要 $k$ 次乘法和 $k-1$ 次加法，共约 $2mnk$ 次浮点运算（FLOPs）。这一规则是下述所有估算的基础。

**单层 Transformer（标准 MHA + SwiGLU FFN）的 Prefill FLOPs。** 设输入序列长度为 $s$，隐藏维度为 $d$，注意力头数为 $n_h$，每头维度 $d_k = d/n_h$，SwiGLU FFN 中间维度为 $d_{\text{ff}}$。

注意力部分的矩阵乘法 FLOPs 包含以下几项。QKV 投影：输入 $[s, d]$ 乘以权重 $[d, 3d]$（标准 MHA 中 Q、K、V 的投影维度均为 $d$），FLOPs 为 $2 \times s \times d \times 3d = 6sd^2$。注意力分数 $\mathbf{Q}\mathbf{K}^\top$：对所有头，等价于 $[s, d] \times [d, s]$，FLOPs 为 $2s^2d$。加权聚合 $\mathbf{A}\mathbf{V}$：同样为 $2s^2d$。输出投影：$[s, d] \times [d, d]$，FLOPs 为 $2sd^2$。注意力部分合计 $8sd^2 + 4s^2d$。

FFN 部分（SwiGLU）的矩阵乘法 FLOPs 包括：Gate 投影 $[s, d] \times [d, d_{\text{ff}}]$ 和 Up 投影 $[s, d] \times [d, d_{\text{ff}}]$，共 $4sd \cdot d_{\text{ff}}$；Down 投影 $[s, d_{\text{ff}}] \times [d_{\text{ff}}, d]$，为 $2sd \cdot d_{\text{ff}}$。FFN 部分合计 $6sd \cdot d_{\text{ff}}$。

因此，单层 Transformer 的 Prefill FLOPs 为：

$$F_{\text{layer,prefill}} = 8sd^2 + 4s^2d + 6sd \cdot d_{\text{ff}}$$

全模型 $L$ 层的总 FLOPs 为（忽略 Embedding 层和 LM Head 的贡献，它们通常占比较小）：

$$F_{\text{prefill}} \approx L \cdot (8sd^2 + 4s^2d + 6sd \cdot d_{\text{ff}})$$

这个公式揭示了两个重要特性。其一，当序列长度 $s$ 较短时（$s \ll d$），$4s^2d$ 项远小于 $8sd^2 + 6sd \cdot d_{\text{ff}}$，总 FLOPs 近似与 $s$ 线性相关，等价于 $s$ 次前向传播的线性叠加。其二，当序列长度 $s$ 很长（$s \gg d$）时，$4s^2d$ 项成为主导，注意力的二次复杂度开始显现。以 LLaMA-2 70B 为例（$d = 8192$, $d_{\text{ff}} = 28672$, $L = 80$），当 $s > 8192$ 时注意力的二次项开始超过线性项，这正是长上下文推理面临的核心计算挑战。

值得一提的是，对于常见的参数配置（$d_{\text{ff}} \approx \frac{8}{3}d$），线性项 $8sd^2 + 6sd \cdot d_{\text{ff}} \approx 8sd^2 + 16sd^2 = 24sd^2$，因此一个常用的 Prefill 阶段快速估算公式（忽略注意力二次项）为：

$$F_{\text{prefill}} \approx 2 \cdot s \cdot P_{\text{model}}$$

其中 $P_{\text{model}}$ 为模型的总参数量。直觉上，对于每个 Token 的前向传播，每个模型参数大约参与 2 次浮点运算（一次乘法、一次加法）。这一"$2 \times \text{tokens} \times \text{params}$"的近似公式虽然粗糙，但在实践中广泛使用，因为它在大多数实际序列长度下（$s < d$）误差不大。

**Decode 阶段的单步 FLOPs。** 在 Decode 阶段，每一步仅处理一个新 Token（$s = 1$），但注意力计算需要关注所有已有的 $t$ 个 Token（包括 Prompt 和之前生成的 Token）。

QKV 投影：$[1, d] \times [d, 3d]$，FLOPs 为 $6d^2$。注意力分数：$[1, d_k] \times [d_k, t] = [1, t]$（每个头），所有头合计 $2td$。加权聚合：$2td$。输出投影：$2d^2$。注意力部分合计 $8d^2 + 4td$。FFN 部分：$6d \cdot d_{\text{ff}}$。单层 Decode 单步 FLOPs 为：

$$F_{\text{layer,decode}} = 8d^2 + 4td + 6d \cdot d_{\text{ff}}$$

全模型单步 Decode FLOPs 为：

$$ F_{ \text{decode step}} \approx L \cdot (8d^2 + 4td + 6d \cdot d_{\text{ff}})$$

对于生成 $s_g$ 个 Token 的完整 Decode 过程，总 FLOPs 为对 $t$ 从 $s_p + 1$ 到 $s_p + s_g$ 求和。注意力二次项的贡献为 $4dL\sum_{t}t \approx 2dLs_g(2s_p + s_g)$，但在大多数实际场景中（$t < d$），线性项仍然主导，可以近似为 $F_{\text{decode total}} \approx 2 \cdot s_g \cdot P_{\text{model}}$。

**关键洞察：Prefill 和 Decode 的 FLOPs 量级差异。** Prefill 处理 $s_p$ 个 Token 的 FLOPs 约为 $2s_pP$，而 Decode 生成一个 Token 的 FLOPs 约为 $2P$（忽略注意力项）。也就是说，Decode 单步的计算量约为 Prefill 的 $1/s_p$。然而如前所述，Decode 的瓶颈不在计算量，而在于这些计算所需的数据访问模式——大量的模型权重读取和 KV Cache 读取。这种"计算少、搬运多"的特性，使得 Decode 阶段的实际耗时远高于其 FLOPs 所暗示的水平。

**使用 GQA 时的 FLOPs 修正。** 当使用 Grouped-Query Attention 时，KV 投影的参数量减少。设 KV 的组数为 $n_{\text{kv}}$（$n_{\text{kv}} < n_h$），则 KV 投影的参数变为 $\mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times 2n_{\text{kv}}d_k}$。QKV 投影的 FLOPs 修正为 $2sd(d + 2n_{\text{kv}}d_k) = 2sd(d + 2d \cdot n_{\text{kv}}/n_h)$。在 $n_{\text{kv}} \ll n_h$ 的情况下，这一修正可使 QKV 投影的 FLOPs 接近减半。注意力分数和加权聚合的 FLOPs 不变（因为 Q 仍然有 $n_h$ 个头，只是 K 和 V 被广播到对应的头组）。实践中，GQA 带来的 FLOPs 节省相对于其在 KV Cache 显存上的节省来说并不突出，后者才是 GQA 的核心价值。

---

## 2.4 显存占用分析：模型参数、KV Cache、激活值

推理过程中 GPU 显存的占用主要由三个部分构成：模型参数、KV Cache 和中间激活值。不同于训练过程还需要存储优化器状态和梯度，推理的显存需求虽然更低，但随着模型规模和并发请求数的增长，显存仍然是制约推理吞吐量的核心瓶颈。

### 2.4.1 模型参数的显存公式

模型参数的显存占用是最直观的：它取决于模型的参数总量和每个参数的存储精度。

对于一个标准的 $L$ 层 Transformer Decoder（MHA + SwiGLU FFN），逐层参数量如下。每层注意力部分包含 QKV 投影和输出投影，标准 MHA 下为 $4d^2$（$\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ 各 $d \times d$，$\mathbf{W}_O$ 为 $d \times d$）。使用 GQA 时修正为 $d^2 + 2n_{\text{kv}}d_k \cdot d + d^2 = 2d^2 + 2n_{\text{kv}}d_k d$。每层 FFN 部分（SwiGLU）包含 gate、up、down 三个矩阵，参数量为 $3d \cdot d_{\text{ff}}$。每层归一化层有两个 RMSNorm（注意力前和 FFN 前），参数量为 $2d$。

因此，单层总参数量约为：

$$P_{\text{layer}} = 4d^2 + 3d \cdot d_{\text{ff}} + 2d$$

对于 $L$ 层模型，加上词嵌入层（$V \times d$，$V$ 为词表大小）和可能的输出层，总参数量为：

$$P_{\text{total}} = L \cdot (4d^2 + 3d \cdot d_{\text{ff}} + 2d) + V \cdot d + P_{\text{head}}$$

其中 $P_{\text{head}}$ 为 LM Head 的参数（如果与 Embedding 共享权重则为 0）。

显存占用为参数量乘以每个参数的字节数。在 FP16/BF16 精度下，每个参数占 2 字节；在 FP32 下占 4 字节；在 INT8 量化下占 1 字节；在 INT4 量化下占 0.5 字节。因此：

$$M_{\text{params}} = P_{\text{total}} \times b_{\text{param}}$$

其中 $b_{\text{param}}$ 为每参数字节数。

以几个典型模型为例进行具体计算。LLaMA-2 7B 参数量约 70 亿，FP16 下显存占用约 $7 \times 10^9 \times 2 = 14$ GB；INT4 量化后约 3.5 GB。LLaMA-2 70B 参数量约 700 亿，FP16 下约 140 GB，超出单张 A100 80GB 的显存，需要至少两卡张量并行或使用量化；INT4 量化后约 35 GB，可以放入单张 80GB 卡。DeepSeek-V3 的总参数量约 671B，但由于是 MoE 架构，每次推理仅激活约 37B 参数用于计算，不过所有参数都需要加载到内存中（GPU 显存、CPU 内存或二者的组合）。FP16 下所有参数约 1.34 TB 显存——这正是 KTransformers 将大部分专家参数卸载到 CPU 内存的核心动机。

**量化对显存的影响是直接而显著的。** 从 FP16 到 INT4 量化，参数显存减少 4 倍。这解释了为什么量化技术在推理优化中如此关键（详见第6章）：它直接释放了显存空间，使得同一张 GPU 可以部署更大的模型，或者为更大的 Batch Size 和更多的 KV Cache 腾出空间。

### 2.4.2 KV Cache 的显存增长模型

KV Cache 是推理场景特有的显存消耗来源，也是推理系统设计中最关键的资源管理对象。与模型参数（推理期间固定不变）不同，KV Cache 的大小随序列长度和并发请求数动态变化。

对于标准 MHA 架构，每一层需要存储 Key 和 Value 两个矩阵，每个的维度为 $[s, d]$（$s$ 为当前序列长度），因此每层 KV Cache 的大小为 $2 \times s \times d$（元素个数）。$L$ 层模型的单条请求 KV Cache 总量为：

$$\text{KV}_{\text{elements}} = 2 \times L \times s \times d$$

以字节计：

$$M_{\text{KV}} = 2Lsd \times b_{\text{kv}}$$

其中 $b_{\text{kv}}$ 为 KV Cache 每元素字节数（FP16 下为 2，INT8 KV Cache 为 1）。

当使用 GQA 时，KV 的头数从 $n_h$ 减少到 $n_{\text{kv}}$，KV Cache 大小修正为：

$$M_{\text{KV,GQA}} = 2L \times s \times n_{\text{kv}} \times d_k \times b_{\text{kv}} = 2Ls \cdot d \cdot \frac{n_{\text{kv}}}{n_h} \cdot b_{\text{kv}}$$

GQA 带来的 KV Cache 压缩比为 $n_{\text{kv}}/n_h$。以 LLaMA-2 70B 为例（$n_h = 64$, $n_{\text{kv}} = 8$），GQA 将 KV Cache 压缩到标准 MHA 的 $1/8$。

将公式具体化以建立直觉。考虑 LLaMA-2 13B（$L = 40$, $d = 5120$, $n_h = 40$, GQA $n_{\text{kv}} = 40$ 即标准 MHA），FP16 KV Cache，单条请求序列长度 $s = 4096$：

$$M_{\text{KV}} = 2 \times 40 \times 4096 \times 5120 \times 2 \text{ bytes} \approx 3.28 \text{ GB}$$

单条请求就消耗约 3.28 GB 的 KV Cache 显存。如果要支持 16 个并发请求，KV Cache 总消耗约 52.5 GB——加上模型参数（约 26 GB），几乎占满一张 A100 80GB。这个简单计算就解释了为什么 KV Cache 管理是推理引擎设计的核心问题：在高并发、长上下文场景下，KV Cache 往往比模型参数本身占用更多的显存。

**KV Cache 显存的增长模型具有如下特征。** 其一，与序列长度线性增长：每生成一个新 Token，每层的 KV Cache 增加 $2d \cdot b_{\text{kv}}$ 字节。其二，与并发请求数（Batch Size）线性增长：每多一个并发请求，就多一份完整的 KV Cache。其三，与层数线性增长。三者的乘积效应使得 KV Cache 在规模化推理中迅速成为显存的主导消耗。

正是因为 KV Cache 的这种增长特性，vLLM 的 PagedAttention（通过分页机制减少碎片化浪费）和 SGLang 的 RadixAttention（通过前缀缓存复用 KV Cache）才成为如此关键的技术创新——它们本质上都是在最大化 KV Cache 显存的利用效率（详见第5章）。

### 2.4.3 激活值的峰值显存估算

推理过程中的中间激活值（intermediate activations）是第三类显存消耗。与训练不同——训练需要保存所有层的中间激活值用于反向传播——推理只需要保留当前正在计算的层的激活值，前一层的激活值在计算完毕后即可释放。因此，推理中激活值的峰值显存由单层计算中的峰值决定，而不像训练那样与层数成正比。

分析单层 Transformer 推理中的激活值峰值，需要追踪计算流中每个步骤产生的中间张量。对于 Prefill 阶段（序列长度 $s$，批大小 $B$，有效 Token 数为 $N = B \times s$），主要的中间激活包括：

注意力子层中，QKV 投影的输出 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$，大小为 $3 \times N \times d \times b_{\text{act}}$。注意力分数矩阵 $\mathbf{S} \in \mathbb{R}^{N \times s}$（每个头），所有头合计 $n_h \times N \times s \times b_{\text{act}}$。这是峰值的关键贡献者——当 $s$ 很大时，$n_h \times s^2$ 可能非常大。不过在使用 FlashAttention 的情况下，注意力分数矩阵无需完整存储（FlashAttention 的核心优势之一），峰值显存大幅降低。

FFN 子层中，SwiGLU 的中间激活（gate 和 up 投影的输出）大小为 $2 \times N \times d_{\text{ff}} \times b_{\text{act}}$。这往往是 FFN 部分的峰值。

在使用 FlashAttention 的现代推理引擎中，注意力计算的峰值显存被有效控制，FFN 的中间激活通常成为单层峰值的主要来源。对于单层，FFN 中间激活的峰值约为：

$$M_{\text{act,peak}} \approx 2 \times N \times d_{\text{ff}} \times b_{\text{act}}$$

以 LLaMA-2 70B（$d_{\text{ff}} = 28672$）、FP16 激活、$N = 4096$（一个请求的 Prefill）为例：

$$M_{\text{act,peak}} \approx 2 \times 4096 \times 28672 \times 2 \approx 450 \text{ MB}$$

与模型参数（约 140 GB）和 KV Cache（数十 GB）相比，单层激活值的峰值显存在大多数配置下是相对较小的。这一结论的前提是使用了 FlashAttention；如果不使用 FlashAttention，注意力分数矩阵的 $O(n_h \cdot s^2)$ 显存在长序列下可能非常大。

在 Decode 阶段，由于每步仅处理一个 Token（$N = B$，$B$ 为 Batch Size），中间激活值非常小，通常可以忽略不计。

**三部分显存的总和与动态变化。** 综合以上分析，推理过程中的总显存占用可以概括为：

$$M_{\text{total}} = M_{\text{params}} + M_{\text{KV}}(B, s) + M_{\text{act}}(B, s)$$

其中 $M_{\text{params}}$ 是固定的，$M_{\text{KV}}$ 随 Batch Size 和序列长度线性增长（是主要的动态开销），$M_{\text{act}}$ 在 Prefill 时有一定开销、在 Decode 时可忽略。

推理引擎的显存管理核心就是在 $M_{\text{params}}$ 固定的前提下，将剩余显存尽可能高效地分配给 KV Cache，以支持更大的 Batch Size（提高吞吐量）或更长的序列（支持长上下文）。vLLM 的 `gpu_memory_utilization` 参数正是控制这一分配比例的关键配置（详见第9章）。

---

## 2.5 Roofline 模型与推理瓶颈定位

Roofline 模型是分析计算任务在特定硬件上的性能瓶颈的经典框架。通过将推理中的各个操作映射到 Roofline 模型上，我们可以精确判断每个操作是受限于计算能力还是访存带宽，从而有针对性地选择优化策略。

### 2.5.1 计算密集 vs. 访存密集的判定

任何一个计算任务都可以用两个指标来刻画：它需要多少次浮点运算（FLOPs）以及它需要读写多少字节的数据（Bytes）。这两者的比值被称为算术强度（Arithmetic Intensity）：

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}$$

单位为 FLOPs/Byte。直觉上，算术强度反映了"每搬运一个字节的数据，要做多少次计算"。

硬件方面，GPU 有两个核心性能参数：峰值计算吞吐量 $\pi$（单位 FLOPS，即每秒浮点运算次数）和峰值内存带宽 $\beta$（单位 Bytes/s）。它们的比值 $\pi / \beta$ 被称为硬件的"脊点"（Ridge Point），单位也是 FLOPs/Byte，代表硬件计算能力和访存能力恰好平衡的算术强度。

Roofline 模型的核心判定法则如下。如果一个算子的算术强度 $\text{AI} < \pi / \beta$，那么它是访存密集型的（Memory-Bound），其性能由内存带宽决定，实际吞吐为 $\text{Perf} = \text{AI} \times \beta$ FLOPS。如果 $\text{AI} > \pi / \beta$，则它是计算密集型的（Compute-Bound），性能受限于计算单元的峰值吞吐，$\text{Perf} = \pi$ FLOPS。

以 NVIDIA A100 80GB SXM 为例，其 FP16 Tensor Core 峰值算力为 $\pi = 312$ TFLOPS，HBM 带宽为 $\beta = 2.0$ TB/s，脊点为 $312 / 2.0 = 156$ FLOPs/Byte。对于 H100 SXM，FP16 Tensor Core 峰值约 $\pi = 990$ TFLOPS（稠密），HBM3 带宽约 $\beta = 3.35$ TB/s，脊点约 $295$ FLOPs/Byte。

这意味着在 A100 上，一个算子需要每搬运一字节数据至少做 156 次浮点运算，才能充分利用计算资源。如果低于这个阈值，无论算力多强大，性能都取决于数据能多快从内存传到计算单元。

### 2.5.2 Arithmetic Intensity 分析

现在我们计算推理中关键操作的算术强度。

**线性层（矩阵乘法）：Prefill vs. Decode。** 考虑一个线性层 $\mathbf{Y} = \mathbf{X}\mathbf{W}$，其中 $\mathbf{W} \in \mathbb{R}^{d \times d}$，$\mathbf{X} \in \mathbb{R}^{N \times d}$。

FLOPs 为 $2Nd^2$。数据搬运量包括读取权重 $\mathbf{W}$（$d^2 \times b_w$ 字节，$b_w$ 为权重精度字节数）、读取输入 $\mathbf{X}$（$Nd \times b_x$ 字节）、写出输出 $\mathbf{Y}$（$Nd \times b_y$ 字节）。在推理引擎的典型实现中，权重是最大的数据搬运项（$d$ 通常远大于 $N$，尤其在 Decode 阶段），因此可近似为：

$$\text{AI}_{\text{linear}} \approx \frac{2Nd^2}{d^2 b_w + Nd(b_x + b_y)} \approx \frac{2N}{b_w} \quad (\text{当 } d \gg N)$$

在 Decode 阶段，$N = B$（Batch Size），FP16 权重下 $b_w = 2$，则 $\text{AI} \approx B$。这意味着：在 A100 上（脊点 156），Decode 阶段的线性层要达到计算密集，需要 $B > 156$。而实际推理场景中，特别是低延迟要求下，Batch Size 往往远小于这个值——单用户场景 $B = 1$ 时，$\text{AI} = 1$，距离脊点差了两个数量级。

在 Prefill 阶段，$N = s_p$（序列长度），$\text{AI} \approx s_p / b_w$。对于 $s_p = 2048$、FP16 权重，$\text{AI} \approx 1024$，远超 A100 的脊点 156，线性层处于计算密集区域。

这一分析清晰地揭示了 Prefill 与 Decode 的根本性差异：同样的矩阵乘法操作，仅因为输入的 Token 数不同，就从计算密集变成了访存密集。

**注意力计算的算术强度。** 注意力分数计算 $\mathbf{Q}\mathbf{K}^\top$（每个头）可以看作 $[N, d_k] \times [d_k, t]$ 的矩阵乘法。在 Decode 阶段（$N = 1$），FLOPs 为 $2d_k t$，需要读取 $d_k + d_k t$ 个元素（一个 Query 向量和整个 Key Cache），$\text{AI} \approx 2d_k t / (d_k t \cdot b) \approx 2/b$，非常低，属于典型的访存密集操作。在 Prefill 阶段（$N = s_p$），使用 FlashAttention 的情况下，分块计算使得有效的算术强度接近 $\text{AI} \approx s_p / b$，与线性层类似。

**逐元素操作（RMSNorm、激活函数、残差加法）。** 这些操作的 FLOPs 为 $O(Nd)$，数据搬运也是 $O(Nd)$ 字节，因此 $\text{AI} \approx O(1)$——它们始终是访存密集的。在推理中，这些操作通常通过算子融合（fused kernel）来减少额外的显存读写（见第16章）。

### 2.5.3 Prefill 与 Decode 在 Roofline 模型中的位置

将上述分析汇总到 Roofline 图上，可以得到以下清晰的图景：

Prefill 阶段的主要操作（线性层的 GEMM、FlashAttention）位于 Roofline 图的右上方，算术强度高于脊点，处于计算密集区域。这意味着 Prefill 的性能主要由 GPU 的峰值算力决定。提升 Prefill 速度的方法包括：使用更高算力的 GPU（如 H100 相比 A100）、使用 FP8 精度（将 Tensor Core 吞吐翻倍）、增加并行度（张量并行分摊计算量）等。

Decode 阶段的几乎所有操作都位于 Roofline 图的左下方，算术强度远低于脊点，处于访存密集区域（尤其在小 Batch Size 下）。这意味着 Decode 的性能主要由 GPU 的内存带宽决定。提升 Decode 速度的方法有两个方向：一是增加内存带宽（如 HBM3 相比 HBM2e），二是减少需要搬运的数据量（权重量化减少参数读取量、KV Cache 量化减少 Cache 读取量、GQA 减少 KV 头数）。

Decode 阶段还有一个重要的优化方向：增大 Batch Size。如前所述，Decode 线性层的算术强度近似为 $B$，随着 Batch Size 增大，同一份权重数据被多个请求共享，有效算术强度提高。当 $B$ 足够大时（$B > \pi/\beta$），即使是 Decode 也能进入计算密集区域，此时 GPU 的算力才得到充分利用。这正是 Continuous Batching（连续批处理）技术（第8章）的核心价值：通过动态地将更多请求打入同一批次，提升 Decode 阶段的批量大小，从而提高 GPU 利用率和整体吞吐量。

以 A100 上部署 LLaMA-2 7B（FP16）为例进行定量估计。模型参数约 14 GB，每步 Decode 需要读取全部参数，在 2 TB/s 带宽下，纯参数读取耗时约 $14/2000 = 7$ ms。如果 $B = 1$，则仅生成 1 个 Token 就需要约 7 ms（不含 KV Cache 读取开销），即约 143 tokens/s。如果 $B = 32$，则 32 个请求共享一次参数读取，7 ms 内生成 32 个 Token，每个请求的有效速度仍约 143 tokens/s，但系统吞吐提升到 $32 \times 143 \approx 4576$ tokens/s。进一步增大 $B$ 直到计算成为瓶颈（$B \approx 156$ 时），吞吐可以持续线性增长。但 Batch Size 的增大受限于 KV Cache 的显存容量——这又回到了 KV Cache 管理的核心问题。

这一分析框架的实践意义在于：当观察到推理延迟不符合预期时，通过计算相关操作的算术强度并与硬件脊点对比，可以立即判断瓶颈所在。如果操作是访存密集的，再怎么优化计算效率也无济于事，应该从减少数据搬运量或提高带宽入手。如果操作是计算密集的，则应关注算子实现的计算效率是否接近峰值。

---

## 2.6 MoE（Mixture of Experts）架构的推理特殊性

Mixture of Experts 架构近年来成为超大规模语言模型的主流选择，其核心思想是将 FFN 层替换为多个"专家"（Expert）网络，每个 Token 只激活其中少数专家进行计算。这使得模型可以在保持巨大参数量（和对应的表示能力）的同时，将每次前向传播的实际计算量控制在较低水平。然而，MoE 架构为推理引入了一系列独特挑战，也催生了专门的优化方案——KTransformers 的 CPU 专家卸载和 vLLM/SGLang 的专家并行策略都直接回应这些挑战。

### 2.6.1 Gate 网络与专家路由

MoE 层的结构可以表述如下。设共有 $E$ 个专家（每个专家是一个独立的 FFN），Gate 网络（也称为 Router）是一个小型线性层，将输入 Token 的隐藏状态映射为 $E$ 维的专家选择概率：

$$\mathbf{g} = \text{Softmax}(\text{TopK}(\mathbf{x}\mathbf{W}_g, k))$$

其中 $\mathbf{W}_g \in \mathbb{R}^{d \times E}$ 是 Gate 网络的权重，TopK 操作选出得分最高的 $k$ 个专家（其余设为 $-\infty$ 使 Softmax 后为 0）。每个 Token 的输出是被选中的 $k$ 个专家输出的加权和：

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}} g_i \cdot \text{Expert}_i(\mathbf{x})$$

典型的配置包括：GShard 和 Switch Transformer 使用 $k = 1$（每个 Token 只激活一个专家），Mixtral 8x7B 使用 $E = 8, k = 2$，而 DeepSeek-V3 使用 $E = 256, k = 8$（加上一个共享专家，详见 2.6.3 节）。

Gate 网络本身的计算量极小（$2dE$ FLOPs），但它的输出决定了后续计算的分派（routing），因此对系统的调度和通信模式有深远影响。

从推理性能的角度，专家路由引入的核心复杂性在于其动态性和不均匀性。每个 Token 可能被路由到不同的专家组合，这意味着每个专家在同一批次中处理的 Token 数量是不同的（甚至可能为零）。这种不均匀性在批处理和并行计算中造成负载不平衡（load imbalance）：某些专家可能被分配到大量 Token 而成为瓶颈，其他专家则空闲等待。尽管训练中通常引入辅助损失（auxiliary loss）来鼓励均匀路由，推理时的实际路由分布仍然可能高度不均匀。

### 2.6.2 激活稀疏性带来的计算与通信挑战

MoE 的核心优势——激活稀疏性——在推理中既带来了计算上的好处，也引入了独特的系统挑战。

**计算方面。** 对于一个有 $E$ 个专家、每次激活 $k$ 个的 MoE 层，单个 Token 的 FFN 计算量仅为 Dense 模型（如果将所有专家的参数视为一个大 FFN）的 $k/E$。以 DeepSeek-V3 为例，256 个路由专家中激活 8 个，FFN 计算量降低到约 $8/256 = 3.125%$（再加上共享专家的计算）。这使得 DeepSeek-V3 虽然拥有 671B 参数，但每个 Token 的激活参数量仅约 37B，FLOPs 与一个约 37B 的 Dense 模型相当。

然而，参数存储方面，所有 $E$ 个专家的参数都需要在内存中可访问。以 DeepSeek-V3 为例，256 个路由专家加上共享专家的参数量约占总参数的大部分。在 FP16 下，仅专家参数就需要超过 1 TB 的存储空间。这一"大存储、小计算"的特性使得 MoE 模型的推理呈现出极端的访存密集特征：大量参数需要存储但只有少数会被"激活"执行计算。

**Decode 阶段的极端访存压力。** 在 Decode 阶段，每个 Token 需要读取其被路由到的 $k$ 个专家的全部参数。以 DeepSeek-V3 的单个路由专家为例（SwiGLU FFN，$d = 7168$，$d_{\text{ff}} = 2048$，参数量约 $3 \times 7168 \times 2048 \approx 44$ M），8 个专家约 352 M 参数，FP16 下约 704 MB。加上共享专家和注意力层参数，Decode 单步需要读取数 GB 的参数。如果所有参数存储在 GPU HBM 中，带宽尚可支撑；但如果参数需要从 CPU 内存读取（如 KTransformers 的异构方案），PCIe 带宽（通常 32-64 GB/s）就成为严重瓶颈。

**分布式推理中的通信开销。** 当 MoE 模型通过专家并行（Expert Parallelism）部署在多个 GPU 上时，不同专家分布在不同设备上。由于每个 Token 需要的专家可能分布在不同设备上，推理需要 All-to-All 通信：将 Token 发送到持有对应专家的设备，计算完成后再将结果返回。这种 All-to-All 通信的延迟和带宽需求是 MoE 分布式推理的核心瓶颈之一（详见第13章的专家并行讨论）。

**批处理效率的挑战。** 在 Dense 模型中，Batch 中的所有 Token 执行相同的计算路径，GEMM 操作可以直接受益于更大的 Batch Size。在 MoE 模型中，同一 Batch 中的 Token 被路由到不同的专家，每个专家实际处理的 Token 数量是 Batch Size 的一个子集。如果 $B$ 个 Token 均匀分配给 $E$ 个专家（每次 Top-$k$），每个专家平均处理 $Bk/E$ 个 Token。当 $E$ 很大时（如 DeepSeek-V3 的 256），即使 $B$ 不小，每个专家处理的 Token 数也可能很少，导致每个专家的 GEMM 规模很小，难以充分利用 GPU 的计算单元。这一问题在 Decode 阶段尤为严重。

### 2.6.3 DeepSeek-V3/R1 的 MoE 架构案例分析

DeepSeek-V3 是目前最具代表性的超大规模 MoE 语言模型之一，其架构设计集中体现了 MoE 推理的各种挑战和应对思路。本小节以 DeepSeek-V3 为例，具体分析 MoE 架构对推理的影响。

**架构概览。** DeepSeek-V3 拥有约 671B 总参数，61 层 Transformer。其中前 3 层使用 Dense FFN，后 58 层使用 MoE FFN。每个 MoE 层包含 1 个共享专家（Shared Expert）和 256 个路由专家（Routed Experts），每个 Token 激活 8 个路由专家加上共享专家。注意力层使用 MLA（Multi-head Latent Attention），通过 KV 的低秩压缩大幅减少 KV Cache。

**参数分布分析。** 我们可以估算参数在各组件间的分布。注意力层（含 MLA 的投影矩阵）的参数量约占较小比例。Dense FFN 层（前 3 层）的参数量较少。MoE 层中，每层有 $1 + 256 = 257$ 个专家，58 层共 $58 \times 257 = 14906$ 个专家。专家参数占据了总参数的绝大部分（超过 90%）。

**推理计算量分析。** 尽管总参数量为 671B，但每个 Token 的激活参数仅约 37B。具体而言，每个 MoE 层中，一个 Token 激活 1 个共享专家 + 8 个路由专家 = 9 个专家的 FFN。相比将所有 256 个路由专家替换为一个等大的 Dense FFN（假设那样的 Dense FFN 参数量为 $256 \times P_{\text{expert}}$），MoE 的计算量仅为 $9/257$ 左右。

**推理显存与部署挑战。** 在 FP16 下，671B 参数需要约 1.34 TB 显存。即使使用 8 张 H100 80GB（总计 640 GB 显存）也无法全部容纳 FP16 参数。常见的部署方案包括：使用更多的 GPU（如 16 张或 32 张 H100）进行专家并行和张量并行的组合；使用 FP8 量化将参数减半至约 670 GB，勉强放入 8 张 H100；使用 INT4 量化进一步压缩至约 335 GB，但可能影响精度。KTransformers 提供了另一条路径：将约 90% 的专家参数卸载到 CPU 内存（DDR5，容量通常数百 GB），仅将注意力层参数和共享专家参数保留在 GPU 显存中。这使得单张消费级 GPU（如 RTX 4090 24GB）加上足够的 CPU 内存（如 382 GB）即可运行 DeepSeek-V3 671B 模型，尽管吞吐量受 CPU 计算能力和 PCIe 带宽限制而远低于纯 GPU 方案。

**MLA 对 KV Cache 的压缩效果。** DeepSeek-V3 的 MLA 将每层的 KV Cache 从标准 MHA 的 $2 \times n_h \times d_k$ 维压缩到一个较低维的隐变量空间（Latent 维度远小于 $n_h \times d_k$）。这大幅降低了 KV Cache 的显存需求。以标准 MHA 估算，61 层、$d = 7168$，单条请求 $s = 4096$ 的 FP16 KV Cache 约需 $2 \times 61 \times 4096 \times 7168 \times 2 \approx 7.2$ GB。而 MLA 将其压缩到远小于此的值，使得即使在长上下文场景下，KV Cache 也不再是 DeepSeek-V3 推理的首要显存瓶颈——参数存储才是。

**DeepSeek-V3 推理的综合性能画像。** 将以上分析综合，DeepSeek-V3 的推理呈现如下特征：Prefill 阶段，由于每个 Token 仅激活 37B 参数，计算量适中，但 Gate 网络的路由决策和 All-to-All 通信引入额外开销；在张量并行 + 专家并行的分布式部署下，Prefill 性能主要受计算和通信的平衡制约。Decode 阶段，每步仍需为 8 个路由专家读取参数（在 GPU 方案下从 HBM 读取，在 KTransformers 方案下部分从 CPU 内存读取），呈现访存密集特性；批处理效率因专家路由的分散化而低于同计算量的 Dense 模型。

这些特殊性使得 MoE 模型的推理优化不能简单套用 Dense 模型的经验。后续章节中，我们将看到 vLLM 和 SGLang 如何通过专家并行和优化的 All-to-All 通信来高效部署 MoE 模型（第13章），KTransformers 如何利用 CPU-GPU 异构架构和 AMX 指令集来在有限资源下实现 MoE 模型的可用推理（第11章），以及 KLLM 的 K-Means 量化如何通过更极致的参数压缩来缓解 MoE 模型的参数存储压力（第12章）。

---

## 本章小结

本章建立了 Transformer 推理的定量分析框架。我们从架构组件出发，逐层追踪了 Prefill 和 Decode 两阶段的计算流程，推导了 FLOPs 估算公式和显存占用模型，并通过 Roofline 分析揭示了推理性能的根本瓶颈：Prefill 阶段是计算密集的，性能由 GPU 算力决定；Decode 阶段是访存密集的，性能由内存带宽决定。这一分析框架将贯穿全书，成为评估各项优化技术效果的理论基础。

具体而言，读者应当从本章带走以下核心认知：推理的 FLOPs 可以通过 $2 \times \text{tokens} \times \text{params}$ 快速估算（线性项主导时）；KV Cache 显存与序列长度和并发数线性增长，是推理系统中最关键的动态资源；Roofline 模型中算术强度的概念是判断优化方向的指南针——访存密集时优化数据搬运，计算密集时优化算子效率；MoE 架构通过激活稀疏性降低了计算量，但引入了参数存储、路由不均匀和通信等新挑战。

下一章将转向硬件视角，详细分析 GPU 和 CPU 的计算架构、内存层次和互联拓扑，为理解各推理引擎的底层优化手段提供硬件知识基础。
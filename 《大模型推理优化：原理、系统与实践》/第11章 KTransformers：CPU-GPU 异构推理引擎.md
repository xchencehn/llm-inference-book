
# 第11章 KTransformers：CPU-GPU 异构推理引擎

在前几章中，我们系统地介绍了 vLLM 和 SGLang 这两个以 GPU 为中心的推理引擎。它们的设计假设是模型的全部参数能够被完整加载到 GPU 显存中，进而通过高效的调度与缓存策略最大化吞吐量。然而，随着模型规模的急剧膨胀，这一假设在许多实际场景中不再成立。以 DeepSeek-V3/R1 为例，其 671B 参数在 BF16 精度下需要超过 1.2TB 的存储空间，即使采用 INT4 量化也需要约 340GB，远远超出单块甚至多块消费级 GPU 的显存容量。对于那些出于数据安全、隐私保护或研究目的而希望在本地部署超大模型的用户而言，纯 GPU 方案的硬件成本几乎令人望而却步。

KTransformers 正是为解决这一困境而诞生的。它由清华大学 MADSys 实验室与 [Approaching.AI](http://approaching.ai/) 联合开发，是一个专门为 MoE（Mixture-of-Experts）模型设计的 CPU-GPU 异构推理引擎。其核心洞察在于：MoE 模型的稀疏激活特性使其天然适合异构计算——将计算密集但参数量小的注意力层和共享专家放在 GPU 上执行，而将参数量庞大但每次仅稀疏激活少数几个的路由专家卸载到 CPU 侧进行计算。通过这种精心设计的计算划分，KTransformers 能够在仅配备一块 24GB 显存消费级 GPU 加上数百 GB 系统内存的单机上，运行 671B 规模的大模型，并达到实用的推理速度。

KTransformers 的相关工作发表于操作系统领域顶级会议 SOSP 2025，论文题为 “KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models”。其在开源社区获得了广泛关注与采用，截至 2026 年初已在 GitHub 上获得超过两万颗星标，并被部署在众多需要本地推理的生产场景中。

---

## 11.1 设计动机：资源受限条件下的超大模型本地推理

理解 KTransformers 的设计动机，需要首先审视当前大模型推理部署所面临的根本性资源矛盾。

在大规模云端部署场景中，推理服务提供商通常配备多块高端数据中心级 GPU（如 NVIDIA A100 80GB 或 H100 80GB），通过张量并行、流水线并行等分布式策略将模型参数均匀分布到集群中。这种方案虽然能够提供高吞吐量和低延迟，但其硬件成本极其高昂——部署一个 671B 模型可能需要 8 块或更多 A100/H100 GPU，单台服务器的采购成本可达数十万美元。对于许多组织和个人而言，这一成本是不可接受的。

与此同时，本地部署推理的需求却在持续增长。这些需求来源于多个方面。首先是数据安全与隐私保护：金融、医疗、法律等行业的敏感数据不允许传输到外部 API 服务。其次是模型研究与调试：研究人员需要在本地环境中深入分析模型的内部行为，包括中间层的激活值、专家路由分布等细粒度信息，这些在远程 API 调用中无法获取。再者是网络延迟与可靠性：在某些边缘计算或网络受限的场景中，依赖远程推理服务并不可行。

面对这种供需矛盾，一个自然的想法是利用 CPU 侧丰富的内存资源。现代服务器 CPU 通常支持数百 GB 甚至数 TB 的 DRAM 容量，而消费级 DDR5 内存的价格远低于 GPU HBM 显存。更重要的是，现代 CPU 的计算能力也在快速提升——Intel 从第 4 代 Xeon 可扩展处理器（Sapphire Rapids）开始引入了 AMX（Advanced Matrix Extensions）指令集，专门用于加速矩阵乘法运算，理论峰值可达数十 TFLOPS。这为在 CPU 侧执行大规模矩阵计算提供了硬件基础。

MoE 架构的稀疏激活特性使得 CPU-GPU 异构推理成为一种特别有吸引力的方案。在 DeepSeek-V3 模型中，每层有 256 个路由专家，但每个 Token 仅激活其中 8 个（通过 Grouped Top-K 路由机制选择）。这意味着在低并发场景下（如单用户使用），每一步推理仅需要访问全部专家参数的约 3% 左右。这种极端的稀疏性大幅降低了每次推理所需的有效内存带宽，使得 CPU 侧的 DDR 带宽足以支撑实用的推理速度。

然而，此前的 CPU-GPU 异构推理方案存在显著的性能瓶颈。以 Fiddler 为代表的早期工作虽然证明了基于算术强度的计算划分思路的可行性，但在实际应用于 DeepSeek-V3 等大型 MoE 模型时，其性能远未达到实用水平。KTransformers 的性能分析显示，Fiddler 在 DeepSeek-V3 上仅能达到 70 tokens/s 的 Prefill 速度和 4.68 tokens/s 的 Decode 速度，GPU 利用率低于 30%。造成这种低性能的原因主要有两个：一是 CPU 侧的计算效率低下，现有实现未能充分利用 AMX 等高级指令集，PyTorch 默认的 AMX 内核（基于 oneDNN 库）仅能达到理论峰值的约 7%；二是 CPU-GPU 之间的同步开销过大，频繁的 CUDA Kernel 启动和设备间同步严重拖慢了整体流水线。

KTransformers 的设计目标正是系统性地解决这两个瓶颈，通过深度优化 CPU 侧计算内核与 CPU-GPU 协调机制，释放异构硬件的全部潜力，使得在资源受限的本地环境中高效运行超大 MoE 模型成为可能。

---

## 11.2 CPU-GPU 异构计算模型

CPU-GPU 异构推理的核心挑战在于如何合理地将模型的不同计算任务分配到最适合它们的硬件上，并在保证正确性的前提下最小化跨设备协调的开销。KTransformers 的异构计算模型围绕 MoE 模型的独特架构特征进行了精心设计。

### 11.2.1 计算划分原则：什么放 GPU，什么放 CPU？

KTransformers 的计算划分遵循一个核心原则：**按算术强度（Arithmetic Intensity, ARI）进行任务分配**。算术强度定义为单位内存访问字节所执行的浮点运算次数（FLOPs/Byte），它直接决定了一个计算任务更适合在计算密集型硬件（GPU）还是访存密集型硬件（CPU+DRAM）上执行。

回顾第 2 章的 Roofline 模型分析：GPU 拥有极高的计算吞吐量（如 A100 的 FP16 峰值为 312 TFLOPS）和 HBM 带宽（2 TB/s），适合执行高算术强度的任务；而 CPU+DRAM 系统虽然计算峰值较低，但拥有远大于 GPU 的内存容量，且通过多通道 DDR5 能提供可观的内存带宽（如双路 Xeon 服务器可达 400-500 GB/s）。

在 MoE Transformer 模型中，不同组件的算术强度差异显著。注意力（Attention）层的算术强度通常较高。在 Prefill 阶段处理长序列时，QKV 投影和输出投影涉及大规模矩阵乘法，每次内存访问可以支撑大量的浮点运算。即使在 Decode 阶段，注意力计算需要频繁访问 KV Cache，但 KV Cache 本身的参数量相对较小，适合存放在高带宽的 GPU HBM 中。更重要的是，DeepSeek-V3 采用的 MLA（Multi-head Latent Attention）机制通过将 KV Cache 压缩到低维潜空间，进一步减小了注意力层的显存占用，使其更适合放在 GPU 上。

相比之下，路由专家（Routed Experts）呈现出完全不同的特征。虽然单个专家的计算结构与标准 FFN 类似（由 Gate、Up、Down 三个线性投影组成），但在低并发场景下每次仅有少量 Token 被路由到每个专家。以 DeepSeek-V3 的 Decode 阶段为例，每步仅处理一个 Token，每层激活 8 个路由专家，每个专家执行的实际上是向量-矩阵乘法（GEMV），算术强度极低。然而，这些专家的参数总量却非常庞大——DeepSeek-V3 的 256 个路由专家的参数占据了模型总参数的绝大部分。将它们全部放在 GPU 显存中既不现实也不必要。

基于上述分析，KTransformers 确立了以下计算划分策略。GPU 负责执行注意力层的全部计算（包括 QKV 投影、注意力得分计算、输出投影、MLA 的吸收运算），共享专家（Shared Experts）的计算（因为共享专家对每个 Token 都被激活，其有效算术强度较高），路由门控网络（Gate Network）的计算（参数量极小），以及 LayerNorm、残差连接等轻量级操作。CPU 则负责路由专家（Routed Experts）的计算，利用 DRAM 的大容量存储全部路由专家的参数，利用 AMX/AVX-512 指令集执行矩阵运算。

这种划分方式的合理性在 DeepSeek-V3 模型上尤为明显。该模型有 2 个共享专家和 256 个路由专家，共享专家的参数量仅占专家总参数的不到 1%，而全部路由专家的参数量超过 600GB（BF16 精度）。将共享专家放在 GPU 上仅占用极少量显存，而将路由专家放在 CPU 则可以充分利用数百 GB 的 DRAM 空间。

### 11.2.2 Attention 在 GPU、Experts 在 CPU 的基本范式

理解了划分原则后，我们可以描述 KTransformers 中一个 MoE Transformer 层的完整执行流程。

在标准的 MoE Transformer 架构中（如图所示），每一层由注意力模块和 MoE 模块交替组成。MoE 模块内部又包含门控网络、共享专家和路由专家三个部分。KTransformers 的基本执行流程如下：

第一步，GPU 执行注意力模块。输入 Token 的隐状态首先经过 LayerNorm，然后进入注意力计算。QKV 投影、RoPE 位置编码、注意力得分计算、输出投影等全部在 GPU 上完成。注意力计算结束后，残差连接将注意力输出与输入相加，产生 MoE 模块的输入。

第二步，GPU 执行门控网络。门控网络是一个小型线性层，将每个 Token 的隐状态映射到一个长度为 256 的分数向量（对应 256 个路由专家），然后通过 Grouped Top-K 选出需要激活的 8 个路由专家及其对应的路由权重。

第三步，CPU 和 GPU 并行执行专家计算。GPU 知道了路由结果后，启动两个并行的计算流：GPU 上执行共享专家的前向计算（Gate 投影 → 激活函数 → Up 投影 → 逐元素相乘 → Down 投影），CPU 上执行被选中的路由专家的计算。CPU 控制线程从 GPU 接收路由信息和输入激活值，将路由专家任务分发到 CPU 工作线程的无锁任务队列中，工作线程并行执行各专家的矩阵运算。

第四步，合并专家输出。当 GPU 侧的共享专家计算和 CPU 侧的路由专家计算都完成后，控制线程将两者的输出在 GPU 上合并（加权求和），加上残差连接，形成下一层的输入。

整个流程的关键在于第三步中 CPU 和 GPU 的并行执行。由于共享专家在 GPU 上的计算与路由专家在 CPU 上的计算之间没有数据依赖关系（它们共享同一个输入但产生独立的输出），这两个计算可以同时进行。理想情况下，这种并行能够将 MoE 模块的执行时间降低到 CPU 和 GPU 计算时间中较长者，而不是两者之和。

然而实际情况并不如此理想。在 DeepSeek-V3 中，GPU 侧共享专家的计算时间仅占 MoE 层总执行时间的约 18%。这意味着即使共享专家与路由专家完美并行，CPU 侧的路由专家计算仍然是整个 MoE 层的主要瓶颈，GPU 在等待 CPU 完成时有大量空闲时间。这一观察直接催生了后文将要介绍的 Expert Deferral 机制（见 11.3.4 节）。

### 11.2.3 CPU-GPU 流水线化与延迟隐藏

在基本的"Attention 在 GPU、Experts 在 CPU"范式中，每一层的执行存在明确的串行依赖：第 $k$ 层的 MoE 输出是第 $k+1$ 层注意力的输入。这种依赖在每一层的边界处引入了 CPU-GPU 同步点，成为性能瓶颈的重要来源。

KTransformers 通过两层机制来隐藏这些同步开销。

第一层是异步 CPU-GPU 任务调度。在传统实现中，每当 CPU 和 GPU 之间需要交换数据或同步执行进度时，都需要显式地调用同步原语（如 `cudaStreamSynchronize`），这不仅引入了等待时间，还会打断 CUDA 的执行流，导致后续 Kernel 无法被提前排队。KTransformers 将同步操作封装为 `cudaLaunchHostFunc` 回调，使得同步逻辑在 CUDA Stream 内部作为一个"主机函数"被调用，而不是打断 Stream 的执行。具体来说，当 GPU 完成门控网络计算后，一个 Host Function 回调被触发，负责将路由信息传递给 CPU 工作线程并启动 CPU 侧计算。当 CPU 完成路由专家计算后，另一个 Host Function 回调将结果传回 GPU 用于合并。这种设计使得整个 Decode 阶段的动态形状 CPU-GPU 混合前向计算能够被捕获为一个单一的 CUDA Graph，从而消除了逐层的 Kernel Launch 开销。

第二层是全图 CUDA Graph 捕获。如第 16 章所述，CUDA Graph 能够将多个 GPU Kernel 的启动序列预先录制并作为一个整体重放，消除每次 Kernel 启动的 CPU 端开销。然而在 CPU-GPU 混合执行的场景中，传统的 CUDA Graph 实现面临严峻挑战：CPU 侧的操作会打断 CUDA Graph 的录制连续性，PyTorch 的 `torch.compile` 通常会在每层边界处将图打碎为多个片段，而 llama.cpp 则直接禁用了 CUDA Graph。KTransformers 通过上述的 Host Function 机制，将 CPU 操作作为 CUDA Stream 内的回调节点嵌入图中，从而将整个 Decode 阶段的逐 Token 前向计算封装为一个完整的 CUDA Graph 实例。实验表明，这一优化在 Decode 阶段带来了高达 1.23 倍的加速。

这两层机制相结合，有效地减少了 CPU-GPU 协调过程中的空闲等待，使得硬件利用率接近理论上限。但即便如此，在极大模型（如 DeepSeek-V3 671B）上，单批次 Decode 速度仍然被限制在约 5.87 tokens/s。为了进一步突破这一瓶颈，KTransformers 引入了更为激进的 Expert Deferral 策略，我们将在下一节中详细讨论。

---

## 11.3 MoE 专家卸载（Expert Offloading）核心技术

MoE 专家卸载是 KTransformers 系统的技术核心。不同于传统的权重卸载（将参数从 CPU 内存传输到 GPU 显存再执行计算）或 KV Cache 卸载（将不活跃的缓存数据移到 CPU），KTransformers 采用的是计算卸载（Computation Offloading）策略——专家参数始终驻留在 CPU 内存中，计算也直接在 CPU 上执行。这种策略避免了 PCIe 带宽成为瓶颈的问题（PCIe 4.0 仅有 32 GB/s 的双向带宽，远低于 CPU 的内存带宽），而是将性能挑战转移到了如何在 CPU 侧实现高效的矩阵计算上。

### 11.3.1 专家参数的 CPU 侧存储与按需加载

在 KTransformers 的设计中，路由专家的参数以量化格式（通常为 INT4 或 INT8）持久存储在 CPU 的 DRAM 中。对于 DeepSeek-V3 的 Q4_K_M 量化版本，全部路由专家参数约占 340GB，这对于配备 382GB 或更多 DRAM 的服务器而言完全可以容纳。

参数存储采用了精心设计的内存布局，以匹配后续计算内核的访问模式（详见 11.3.3 节）。每个专家的权重矩阵被预处理为与 AMX Tile 寄存器兼容的子矩阵格式，所有 Tile 按 64 字节缓存行对齐。量化的标度因子（Scale Factor）被独立存储以保持对齐。INT4 格式的 Tile 被打包为 INT8 大小的块，在计算时使用 SIMD 内联函数进行解包，确保最小化的运行时开销。

在推理过程中，每次仅有被门控网络选中的少量路由专家会参与计算。以 DeepSeek-V3 为例，每层 256 个路由专家中每个 Token 仅激活 8 个，因此每次推理步骤仅需从 DRAM 中读取约 3% 的专家参数。这种极端的稀疏访问模式使得 CPU 内存带宽足以支撑实时推理。对于双路 DDR5-4800 服务器（约 440 GB/s 的聚合内存带宽），单步 Decode 需要读取的专家参数量约为 10-15 GB（8 个专家，每个约 1.3-1.9GB，取决于量化精度），理论上仅需 25-35 毫秒的内存访问时间。

对于多 NUMA 节点的系统（如双路服务器有两个 NUMA 节点），KTransformers 并不简单地将不同的专家分配到不同的 NUMA 节点（即专家并行策略），而是采用了 NUMA 感知的张量并行策略。每个专家的权重矩阵按列/行维度在所有 NUMA 节点上进行切分，每个节点仅存储和计算自己的切片，最后通过轻量级的 Reduce-Scatter 操作合并部分结果。这种设计的优势在于：它保证每个 NUMA 节点的计算负载始终均衡（避免了专家并行中某些节点因被选中的专家恰好集中在该节点而过载的问题），且几乎消除了跨 NUMA 节点的远程内存访问。实验表明，相比于不感知 NUMA 拓扑的朴素基线，NUMA 感知张量并行在双路服务器上实现了高达 1.63 倍的 Decode 吞吐量提升。

### 11.3.2 基于 AMX 指令集的 CPU 高性能矩阵计算

充分发挥 CPU 的计算潜力是 KTransformers 的核心技术挑战之一。Intel AMX（Advanced Matrix Extensions）指令集是 KTransformers 在 Prefill 阶段实现高性能 CPU 矩阵计算的关键。

AMX 自 Intel 第 4 代 Xeon 可扩展处理器（代号 Sapphire Rapids）起引入，每个 AMX 使能的核心提供 8 个 Tile 寄存器，每个寄存器可存储一个 16 行 × 64 字节的子矩阵。AMX 指令能够在单条指令中完成两个 Tile 的乘法并将结果累加到目标 Tile 中，同时支持整个 Tile 与内存之间的单指令传输。在 INT8 精度下，单核 AMX 的理论峰值吞吐量可达数 TFLOPS 级别，远超传统 SIMD 指令的性能。

然而，KTransformers 团队的分析揭示了一个令人惊讶的事实：PyTorch 默认使用的 AMX 实现（通过 Intel oneDNN 库）仅能达到 AMX 理论峰值的约 7%。造成这一巨大差距的主要原因并非指令本身的效率问题，而是软件层面的内存布局未能匹配 AMX 的 Tiling 模式，导致缓存效率低下、内存带宽浪费严重，以及线程同步开销过大。

KTransformers 设计了一套专门为 AMX Tiling 模式优化的计算内核，其关键创新包括以下几个方面。

AMX Tiling 感知的内存布局方面，专家权重矩阵在模型加载阶段被预处理为与 AMX Tile 寄存器直接兼容的子矩阵排列。具体来说，权重被划分为 16×64 字节的块，每个块恰好对应一个 AMX Tile 的大小。块内数据按行优先排列并对齐到 64 字节缓存行边界。对于 INT4 量化权重，KTransformers 采用对称分组线性量化方案，将共享的标度因子单独存储以维持数据对齐，INT4 数据被打包为 INT8 格式并在计算时使用 SIMD 内联函数快速解包。这种预处理消除了推理时昂贵的矩阵转置和重塑操作。

缓存友好的 AMX 计算内核方面，基于优化的内存布局，KTransformers 的 AMX 内核通过多层循环嵌套充分利用 CPU 的缓存层次。其执行流程为：首先，专家权重矩阵被纵向分割为多个任务，通过动态调度分配到各 CPU 线程；输入激活值通常驻留在共享的 L3 缓存中。然后，每个任务将专家权重横向分割为恰好适配 L2 缓存容量的块。每个块由多个 AMX Tile 大小的子矩阵组成。输入和权重分别从 L3 和 DRAM 加载一次到 L2 缓存。最后，Tile 级别的计算使用 AMX 指令直接在 Tile 寄存器中完成乘加运算，中间结果缓冲在 L1 缓存中。

这种设计确保了每次数据从 DRAM 加载后能被最大程度地复用：权重数据从 DRAM 加载到 L2 后被多次用于与不同输入子块的乘法运算，输入数据从 L3 加载后同样被多次复用。动态任务调度进一步优先将针对同一专家的任务安排到同一线程执行，最大化缓存命中率。

实验结果证实了这些优化的效果：在 Intel Xeon 8452Y 处理器（36 核单路）上，KTransformers 的 AMX 内核在 DeepSeek-V3 的 MoE 层微基准测试中达到了 21.3 TFLOPS 的峰值吞吐量，相比 PyTorch 基于 oneDNN 的 AMX 基线实现了 3.98 倍的加速。

对于低算术强度的场景（如 Decode 阶段的单 Token 推理），AMX 指令反而是低效的。AMX 的 Tile 操作需要完整填充 16×64 字节的子矩阵，当每个专家仅处理一两个 Token 时，大量 Tile 空间被浪费，指令的启动开销也相对增大。KTransformers 因此设计了一个自适应的 AVX-512 内核，与 AMX 内核共享相同的内存布局，但使用更细粒度的 AVX-512 SIMD 指令进行向量化计算。微基准测试表明，当每个专家处理的 Token 数量少于 4 个时，AVX-512 内核始终优于 AMX 内核。KTransformers 在运行时根据实际的算术强度（由当前批次中分配给每个专家的 Token 数量决定）动态选择使用 AMX 或 AVX-512 内核。这种混合策略在 Decode 阶段相比纯 AMX 实现了高达 1.20 倍的加速，在 Prefill 阶段相比纯 AVX-512 实现了高达 10.81 倍的加速。

### 11.3.3 内存布局优化：数据局部性与 L1 Cache 命中率

内存布局优化是 KTransformers 实现高 CPU 计算效率的基石，值得进一步深入分析。

现代 CPU 的内存层次结构呈金字塔形。以 Intel Xeon 8452Y 为例，L1 数据缓存为每核 48 KB，L2 缓存为每核 2 MB，L3 缓存为共享 67.5 MB，DRAM 带宽约 220 GB/s（单路）。不同层级之间的带宽和延迟差异可达一个数量级以上。如果计算内核的数据访问模式不能良好地匹配缓存层次，数据会在各级缓存之间反复搬运（即缓存抖动），实际有效带宽将远低于理论值。

KTransformers 的内存布局优化策略基于对 MoE 计算模式的深入分析。在一个 MoE 层的矩阵乘法运算 $Y = X \cdot W$ 中（$X$ 为输入激活值矩阵，$W$ 为专家权重矩阵），三个矩阵的尺寸和访问模式各不相同。输入矩阵 $X$ 的行数等于分配到该专家的 Token 数量，列数等于隐状态维度（DeepSeek-V3 为 7168）。权重矩阵 $W$ 是模型参数，维度固定。输出矩阵 $Y$ 的维度由前两者决定。

KTransformers 的缓存优化采用了经典的分块（Tiling）策略，但与 AMX 的硬件 Tile 寄存器进行了精确的协同设计。权重矩阵被分为多个横向条带，每个条带的大小精确匹配 L2 缓存容量。在处理一个条带时，对应的输入子矩阵从 L3 缓存加载到 L1/L2，权重数据从 DRAM 流式加载到 L2。条带内部再分为 AMX Tile 大小的小块，每个小块的计算完全在 L1 缓存和 Tile 寄存器内完成。关键的设计约束是：每个小块的数据大小（输入子块 + 权重子块 + 输出累加缓冲区）不超过 L1 缓存容量，确保 Tile 级计算不会引发 L1 缓存逐出。

64 字节缓存行对齐确保每次内存访问恰好传输一整条缓存行，避免了部分缓存行读取导致的带宽浪费。预取指令（Prefetch）被精心安排，在当前 Tile 计算执行期间提前将下一个 Tile 的数据从 L2 加载到 L1，实现计算与数据传输的重叠。

对于多 NUMA 节点的系统，KTransformers 的 NUMA 感知策略确保每个 NUMA 节点仅访问本地 DRAM 中的权重切片。跨节点的远程内存访问延迟通常是本地访问的 2-3 倍，且会显著降低有效带宽。通过在模型加载阶段将每个专家的权重矩阵按行/列维度均匀分配到各 NUMA 节点，并在计算时确保每个节点的线程仅访问本地数据，KTransformers 几乎完全消除了远程内存流量。

### 11.3.4 异步预取与流水线调度

在前述小节中，我们分别讨论了 CPU 侧的计算优化和 CPU-GPU 协调机制。本节将介绍 KTransformers 的一项创新性优化——Expert Deferral（专家延迟执行）机制，它通过重构模型的执行顺序来实现更深层次的 CPU-GPU 计算重叠。

Expert Deferral 的核心洞察来自对 DeepSeek-V3 执行时间线的详细剖析。在标准的 Attention-MoE 交替执行模式中，一个 MoE 层的执行时间线大致如下：GPU 先完成注意力计算，然后执行门控网络确定路由，接着 CPU 开始执行路由专家计算，GPU 同时执行共享专家计算。由于共享专家仅占 GPU 执行时间的约 18%，GPU 在共享专家计算完成后便进入空闲等待状态，直到 CPU 完成所有路由专家计算。随后 CPU 和 GPU 合并输出，GPU 开始下一层的注意力计算，此时 CPU 又变为空闲。这种"你等我、我等你"的模式使得 CPU 利用率仅约 74%，GPU 利用率更是低至 28%。

Expert Deferral 的关键思想是：利用 Transformer 架构中残差连接的鲁棒性，有策略性地延迟部分路由专家的输出，使其不再反馈到紧邻的下一层注意力模块，而是推迟到更后面的层。这种延迟打破了路由专家计算与相邻注意力层之间的严格依赖关系，使得 CPU 可以在 GPU 执行注意力计算的同时继续执行前一层被延迟的路由专家计算。

具体而言，KTransformers 将每层的路由专家分为两类。Immediate Experts（即时专家）指其输出立即被下一层注意力模块消费的专家，遵循标准 Transformer 的数据流。Deferred Experts（延迟专家）指其输出被推迟到更后面的层消费的专家，其计算可以与后续层的 GPU 操作并行执行。

在标准 MoE 模型中，第 $k$ 层 MoE 模块的输出为：

$$O_k = I_k + S_k(I_k) + R^{\text{all}}_k(I_k)$$

其中 $I_k$ 是输入，$S_k$ 是共享专家计算，$R^{\text{all}}_k$ 是全部路由专家计算。

采用 Expert Deferral 后，第 $k$ 层的输出变为：

$$O_k = \begin{cases} I_k + S_k(I_k) + R^{\text{imm}}_k(I_k), & k = 1 \ I_k + S_k(I_k) + R^{\text{def}}_{k-1}(I_{k-1}) + R^{\text{imm}}_k(I_k), & 1 < k < L \ I_k + S_k(I_k) + R^{\text{def}}_{k-1}(I_{k-1}) + R^{\text{all}}_k(I_k), & k = L \end{cases}$$

其中 $R^{\text{imm}}_k$ 和 $R^{\text{def}}_k$ 分别表示第 $k$ 层即时专家和延迟专家的计算。注意最后一层不应用 Deferral，以保留完整的信息。

为了确定最优的延迟专家数量，KTransformers 对 DeepSeek-V3 进行了系统性的时间线分析。实验以 BF16 精度在单个 MoE 层上进行：

不延迟任何专家时（标准模式，8 个即时专家），CPU 利用率 74%，GPU 利用率 28%，CPU-GPU 重叠仅占总执行时间的 5%。延迟 2 个专家时（6 即时 + 2 延迟），GPU 注意力计算可以在 CPU 完成即时专家后立即开始，实现了部分 CPU-GPU 重叠，单层执行时间减少 19%。但延迟专家过早完成计算，CPU 仍有空闲周期。延迟 3 个专家时（5 即时 + 3 延迟），延迟专家的计算恰好在下一层即时专家开始时完成，CPU 被完全饱和。这是最优配置，单层执行时间减少 26%，端到端 Decode 吞吐量提升 33%。延迟 4 个专家时（4 即时 + 4 延迟），由于 CPU 已经饱和，进一步增加延迟专家数量不再带来吞吐量增益。

基于上述分析，KTransformers 对 DeepSeek-V3 的默认配置为 5 个即时专家 + 3 个延迟专家。对于其他 MoE 模型，一个通用的启发式规则是：延迟最少数量的专家以实现 CPU 的完全利用率，同时保证每层至少有 2 个即时专家以维持模型的稳定性和精度。

关于精度影响，KTransformers 在 HumanEval、MBPP、GSM8K、StrategyQA、LiveBench 等多个基准测试上进行了全面评估，结果表明 Expert Deferral 造成的平均精度下降不超过 0.5%。这种鲁棒性来源于 Transformer 架构中残差连接的固有特性——即使延迟专家的输出被推迟一层消费，残差路径仍然保证了信息的基本流通，延迟的专家输出在下一次被纳入时仍能有效地补充信息。

值得注意的是，Expert Deferral 仅在 Decode 阶段使用，不应用于 Prefill 阶段。在 Prefill 阶段，一个批次中包含大量 Token，它们通常会选择多样化的专家组合，导致即时专家和延迟专家几乎覆盖了所有专家，近乎翻倍了内存访问量，反而成为新的瓶颈，抵消了重叠执行的收益。

---

## 11.4 KTransformers 的系统架构

KTransformers 的系统架构设计在追求极致性能的同时，兼顾了灵活性、可扩展性和易用性。其架构演进经历了从早期的模块注入框架到后来的独立高性能内核库两个阶段。

### 11.4.1 Kernel 注入框架：可编程的算子替换

KTransformers 的一项核心设计理念是通过模块注入（Module Injection）实现性能优化与模型代码的解耦。这一设计使得研究人员能够在不修改原始模型代码的前提下，将标准 PyTorch 模块替换为经过硬件专门优化的高性能实现。

KTransformers 的注入框架建立在 HuggingFace Transformers 库之上。HuggingFace Transformers 是大模型社区中使用最广泛的模型接口库，几乎所有主流开源模型都提供了兼容的模型定义和权重格式。KTransformers 在此基础上增加了一个轻量级的注入层：在模型初始化过程中，框架遍历整个模型的模块树，根据预定义的规则将匹配的模块替换为优化后的版本，整个过程对上层应用完全透明。

注入规则通过 YAML 配置文件定义，每条规则包含匹配（Match）和替换（Replace）两个部分。匹配条件支持按模块名称的正则表达式匹配、按模块类名匹配或两者的组合。替换配置指定了新的模块类名、执行设备（CPU 或 CUDA）以及该模块所需的任何关键字参数。

以 DeepSeek-V3 的 INT4 量化部署配置为例，一个典型的 YAML 配置文件大致如下：首先，通过类匹配将所有 `DeepseekV3MoE` 实例替换为优化的 `FusedMoE` 模块，该模块封装了与 CPU 后端的异步通信逻辑，关键字参数指定使用 `hybrid_AMX_AVX512` 计算内核、对专家权重应用 INT4 量化，并启用 Expert Deferral 策略。然后，通过名称匹配将自注意力模块替换为 `FlashInferMLA` 模块，以利用 FlashInfer 的高性能 CUDA 内核实现支持矩阵吸收优化的 MLA 计算。最后，将所有 `torch.nn.Linear` 线性投影层替换为量化版本（如 GPU 侧使用 Marlin 内核的量化线性层）。

这种配置驱动的设计带来了显著的灵活性。用户可以通过简单修改 YAML 文件来调整部署策略，例如更改量化精度、切换注意力内核实现、启用或禁用 Expert Deferral 等，而无需修改任何 Python 或 C++ 代码。对于新模型的支持，通常仅需编写一份对应的 YAML 配置文件，而核心的优化内核可以跨模型复用。

在工程实现方面，KTransformers 的性能关键组件以约 11,000 行 C++ 代码实现（通过 pybind11 导出到 Python），集成脚本约 2,000 行 Python 代码。C++ 扩展包括 AMX/AVX-512 计算内核、融合 MoE 算子、异步 CPU-GPU 调度器、NUMA 感知的内存管理等模块。Python 层负责模型加载、配置解析、模块注入逻辑以及与 HuggingFace 接口的兼容适配。

### 11.4.2 配置驱动的灵活部署

KTransformers 的配置驱动架构使其能够适应多种不同的硬件环境和部署需求。

在硬件适配方面，KTransformers 支持多种 GPU（从消费级 RTX 4080/4090 到数据中心级 A100/H100）、多种 CPU（Intel Xeon 的 Sapphire Rapids、Emerald Rapids、Granite Rapids 系列，以及 AMD EPYC 系列），以及不同的内存配置（DDR5-4800 标准内存到 MRDIMM-8800 高带宽内存）。计算内核会根据检测到的硬件特性自动选择最优的指令集——有 AMX 支持的 CPU 优先使用 AMX 内核，仅有 AVX-512 的 CPU 回退到 AVX-512 内核，仅有 AVX2 的 CPU 使用 AVX2 内核。

在量化策略方面，KTransformers 支持多种量化组合。GPU 侧可选择 FP16/BF16 全精度、FP8、GPTQ INT4 等格式。CPU 侧支持 GGUF 格式的多种量化级别（Q4_K_M、Q8_0 等）以及在线量化（加载 BF16 权重后实时量化为 INT8 或 INT4）。用户还可以采用混合量化策略，例如注意力层和共享专家在 GPU 上使用 FP8 精度以获得更高的计算精度，而路由专家在 CPU 上使用 INT4 量化以节省内存容量。

在服务模式方面，KTransformers 支持单用户交互模式（通过 `local_chat` 脚本）和多并发服务模式（通过 `balance_serve` 后端）。多并发模式借鉴了 SGLang 的架构设计，实现了 C++ 层面的高性能异步并发调度，包括连续批处理（Continuous Batching）和分块 Prefill（Chunked Prefill），使得多个并发请求能够共享 GPU 资源，进一步提升整体吞吐量。

### 11.4.3 与 Hugging Face Transformers 的兼容层

KTransformers 的设计从一开始就以与 HuggingFace Transformers 生态的无缝兼容为重要目标。这种兼容性体现在以下几个层面。

在模型加载方面，KTransformers 直接使用 HuggingFace 的模型配置文件（`config.json`）和 Tokenizer，用户可以使用 HuggingFace 的标准模型标识符来指定模型（如 `deepseek-ai/DeepSeek-V3`），KTransformers 会自动下载或加载本地的模型配置。权重则从 GGUF 格式文件加载（对于 CPU 侧的量化权重）或从 HuggingFace 标准格式加载（对于 GPU 侧的权重）。

在 API 接口方面，KTransformers 暴露的推理接口与 HuggingFace Transformers 的 `generate` 方法兼容，用户可以使用熟悉的参数（如 `max_new_tokens`、`temperature`、`top_p` 等）控制生成行为。服务模式则提供 OpenAI 兼容的 REST API 接口，可以被任何支持 OpenAI API 的客户端或应用框架直接调用。

在模型支持方面，KTransformers 通过注入框架实现了对多种 MoE 模型的广泛支持。截至 2026 年初，已支持的模型包括 DeepSeek 系列（V2、V2.5、V3、R1、R1-0528）、Qwen 系列（Qwen2-57B-A14B、Qwen3-MoE）、Mixtral 系列（8×7B、8×22B）、LLaMA 4 Maverick、Kimi-K2 系列、GLM-4-MoE、GLM-5、MiniMax-M2.1/M2.5 等。每个新模型的支持通常只需要编写对应的 YAML 注入配置文件和（在必要时）适配少量模型特定的代码。

---

## 11.5 DeepSeek-V3/R1 671B 模型的单机部署案例

DeepSeek-V3/R1 671B 模型的本地部署是 KTransformers 最具代表性的应用场景，也是推动该项目获得广泛关注的核心案例。本节基于 KTransformers 官方文档和 SOSP 2025 论文中的实验数据，详细分析这一部署案例。

### 11.5.1 硬件配置需求分析

部署 DeepSeek-V3/R1 671B 模型需要满足一定的硬件最低要求，但这些要求远低于纯 GPU 方案。

在 GPU 方面，KTransformers 仅需一块 14GB 以上显存的 GPU。在实际部署中，24GB 显存的消费级 GPU（如 NVIDIA RTX 4090）是典型选择。GPU 显存用于存放注意力层参数、共享专家参数、KV Cache 以及中间激活值。使用 Q4_K_M 量化时，这些组件总共占用约 14GB 显存。若使用 FP8 精度的注意力层和共享专家（KTransformers V0.2.2+ 支持），精度更高但显存占用略增。上下文长度也会影响显存消耗：4K 上下文在 24GB 显存下可以舒适运行，8K 上下文需要精心管理显存（KTransformers V0.2.1 通过集成 SGLang 项目的 Triton MLA Kernel 实现了 8K 支持），而 139K 的长上下文则需要启用 Matrix Absorption MLA 和调小 Chunk Size 参数。

在 CPU 和内存方面，这是 KTransformers 部署中最关键的硬件要素。CPU 需要支持 AMX 指令集以获得最佳性能（Intel 第 4 代 Xeon 及以后），但也支持退回到 AVX-512 或 AVX2 指令集（AMD EPYC 系列也被支持）。内存容量是硬性约束：Q4_K_M 量化的 DeepSeek-V3 模型路由专家参数约 340GB。单路配置需要至少 382GB DRAM，双路配置（利用 NUMA 感知张量并行获得更高性能）建议 1TB DRAM。内存带宽直接决定 Decode 速度，DDR5-4800 是基本要求，而 MRDIMM-8800 等高带宽内存能带来显著的性能提升。

KTransformers 的 SOSP 论文中使用的主要测试平台配置为：CPU 为 Intel Xeon Gold 6454S（32 核/路，双路共 64 核，支持 AMX），内存为 1TB DDR5-4800（双路各 8 条），GPU 为 NVIDIA RTX 4090D（24GB VRAM）或 NVIDIA A100（40GB VRAM）。

### 11.5.2 性能实测：Token/s 与精度

KTransformers 在不同版本和配置下展示了显著的性能提升，以下是关键的性能数据。

在 Prefill（输入处理）性能方面，使用 Q4_K_M 量化和 AVX-512 内核（V0.2 版本），单路 32 核配置达到 54.21 tokens/s，双路 64 核配置达到 82.94 tokens/s。使用 BF16 精度在线量化为 INT8 配合 AMX 内核（V0.3 版本），双路配置在 2K 上下文下达到 255.26 tokens/s（8 专家模式）和 286.55 tokens/s（6 专家模式）。相比之下，llama.cpp 使用 232 核仅达到 10.31 tokens/s，KTransformers V0.3 实现了高达 27.79 倍的 Prefill 加速。

在 Decode（逐 Token 生成）性能方面，Q4_K_M 量化下单路 32 核配置达到 8.73 tokens/s，双路 64 核达到 12.21 tokens/s，启用 6 专家模式（Expert Deferral 的变体）可达 13.69 tokens/s。V0.2.1 版本通过集成 SGLang 的 Triton MLA Kernel 进一步提升至约 14-17 tokens/s（取决于上下文长度）。相比 llama.cpp 的 4.51 tokens/s，KTransformers 实现了 3.03 倍的 Decode 加速。

在 SOSP 论文中更系统的评测中，KTransformers 在多个模型上展示了一致的性能优势：对于全精度模型（BF16），相比 Fiddler 实现了 4.62-19.74 倍的 Prefill 加速和 2.42-4.09 倍的 Decode 加速；相比 llama.cpp 实现了 1.25-1.76 倍的 Decode 加速。启用 Expert Deferral 后，Decode 性能进一步提升至 1.66-4.90 倍（相比 Fiddler）。

在多并发场景下（V0.2.4 版本引入的 `balance_serve` 后端），在 Intel Xeon6 + MRDIMM-8800 平台上，通过增加并发请求数，总输出吞吐量从单并发的约 17 tokens/s 提升到 40 tokens/s。2026 年初的 kt-kernel 集成 SGLang 方案在 8 块 L20 GPU + Xeon Gold 6454S 配置下，DeepSeek-R1-0528 FP8 模型的 8 路并发总吞吐量达到 227.85 tokens/s（其中输出吞吐量 87.58 tokens/s）。

在精度方面，标准配置（不使用 Expert Deferral 或专家数量减少）下，KTransformers 的输出与原始模型完全一致，量化精度由所选的量化方法决定（Q4_K_M、INT8、FP8 等）。启用 Expert Deferral（延迟 3 个专家）时，在 HumanEval、MBPP、GSM8K、StrategyQA、LiveBench 等基准测试上的平均精度下降不超过 0.5%。使用 6 专家模式（即仅激活 Top-6 而非 Top-8 路由专家，这是 KTransformers 早期版本探索的一种加速策略）时，实验观察到输出质量没有显著变化，这得益于 DeepSeek-V3 模型本身对专家激活数量具有一定的鲁棒性。

### 11.5.3 与纯 GPU 方案的成本效益对比

KTransformers 的核心价值主张在于以极低的硬件成本实现可用的超大模型推理。下面从成本和性能两个维度进行对比分析。

在纯 GPU 部署方案中，运行 DeepSeek-V3 671B 的 BF16 版本需要至少 8 块 A100 80GB GPU（总显存 640GB），通过张量并行将模型分布到多块 GPU 上。一台配备 8 块 A100 的 DGX A100 服务器的采购成本约为 20-30 万美元。使用 H100 或 B200 等更新的 GPU 成本更高。即使采用 INT4 量化（约 340GB），仍需 5-6 块 A100 80GB 或 4-5 块 H100 80GB。这种配置下的 Decode 速度可以达到数十甚至上百 tokens/s（取决于批处理大小），但硬件成本极高。

在 KTransformers 方案中，典型的硬件配置为：一块 RTX 4090 GPU（约 2,000 美元）加上一台双路 Xeon 服务器（含 1TB DDR5 内存，约 8,000-15,000 美元，取决于 CPU 型号和内存等级）。总硬件成本约 10,000-17,000 美元，仅为纯 GPU 方案的 5%-10%。虽然 Decode 速度（10-17 tokens/s 单并发，40+ tokens/s 多并发）低于纯 GPU 方案，但在单用户交互场景下已经达到了较好的可用水平——10-17 tokens/s 的速度约等于人类快速阅读的速度，对于代码辅助、对话问答等场景已经足够流畅。

从每 Token 成本的角度看，如果以硬件采购成本分摊到使用寿命（假设 3 年）和实际利用率来计算，KTransformers 方案在低并发场景下具有显著的成本优势。纯 GPU 方案的高成本只有在高并发服务场景下才能被充分摊薄。对于个人研究者、小型团队或仅需低频调用的企业内部部署，KTransformers 提供了一个在成本和性能之间的务实平衡点。

当然，这一对比也需要考虑使用场景的差异。纯 GPU 方案适合高并发在线服务，能够通过批处理大幅提升吞吐量；KTransformers 方案更适合低并发的本地部署场景，其核心价值在于可访问性而非绝对性能。两种方案面向的用户群体和使用场景有明确的区分。

---

## 11.6 KTransformers 的局限性与发展方向

尽管 KTransformers 在 MoE 模型的本地异构推理领域取得了突破性进展，它仍然存在一些固有的局限性。理解这些局限性有助于读者在实际部署时做出明智的选择，也指明了系统未来的演进方向。

### 11.6.1 吞吐量限制：单用户场景适用性

KTransformers 最根本的局限在于其吞吐量受限于 CPU 的计算能力和内存带宽，难以达到纯 GPU 方案在高并发下的性能水平。

在 Decode 阶段，每次生成一个 Token 都需要从 DRAM 中读取被激活的路由专家参数，并在 CPU 上完成矩阵计算。即使经过 AMX 优化和 Expert Deferral，这一过程的速度仍然被 DRAM 带宽和 CPU 计算吞吐量所限制。对于单用户场景，DeepSeek-V3 的典型 Decode 速度在 10-17 tokens/s 范围内（取决于具体配置），这对于交互式使用是可接受的，但与纯 GPU 方案数十到上百 tokens/s 的性能差距明显。

增加并发请求数可以在一定程度上提升总吞吐量（从单并发约 17 tokens/s 提升到多并发约 40 tokens/s），这是因为多个请求可能激活不同的专家，允许更好地利用 CPU 计算资源和内存带宽。然而，多并发场景下每个请求的个体延迟会增加，且随着并发数的增长，CPU 计算资源和内存带宽很快会达到饱和。

在 Prefill 阶段，KTransformers 的 AMX 优化内核已经展现了很强的性能（V0.3 达到 286 tokens/s），但长上下文 Prefill 仍然受限于 GPU 显存容量——KV Cache 的显存占用随上下文长度线性增长，在 24GB 显存的 GPU 上限制了可处理的最大上下文长度。

这些限制意味着 KTransformers 最适合以下使用场景：个人开发者或研究人员的本地推理工具、低并发的内部部署服务、模型行为研究和调试（需要访问中间状态）、以及对数据隐私有严格要求的应用。对于需要服务大量并发用户的生产级在线推理服务，纯 GPU 或 GPU 集群方案仍然是更合适的选择。

### 11.6.2 Dense 模型支持的挑战

KTransformers 的核心优势建立在 MoE 模型的稀疏激活特性之上。对于 Dense 模型（如 LLaMA 系列、Qwen Dense 模型），这种优势大幅减弱甚至不存在。

Dense 模型的每一层都需要访问全部 FFN 参数，不存在专家路由的稀疏性。这意味着在 Decode 阶段，每步生成一个 Token 都需要读取整个 FFN 层的参数（而不是仅 3% 的专家参数），所需的内存带宽大幅增加。以 LLaMA-3 70B 为例，每层 FFN 参数约 200MB（INT4 量化），32 层总计约 6.4GB。单步 Decode 需要从 DRAM 读取全部这些参数，在 220 GB/s 的单路内存带宽下约需 29 毫秒，对应约 34 tokens/s——这与 llama.cpp 在类似硬件上的性能接近，KTransformers 的 AMX 优化在此场景下能提供的加速有限，因为瓶颈在于内存带宽而非计算。

此外，Dense 模型的 FFN 层不像 MoE 的路由专家那样存在可并行于注意力计算的自然分界点，Expert Deferral 等依赖于 MoE 结构的优化策略无法应用。

因此，KTransformers 目前主要定位为 MoE 模型的推理引擎，对 Dense 模型的支持并非其设计重点。

### 11.6.3 与 SGLang 的集成方向

KTransformers 最重要的发展方向之一是与 SGLang 推理引擎的深度集成。2025 年 10 月，KTransformers 团队在 SGLang 仓库中开启了正式的集成提案（Issue #11425），将 KTransformers 的 CPU 端内核能力以 kt-kernel 库的形式整合到 SGLang 的推理后端中。

这一集成的动机在于发挥两个系统各自的优势。SGLang 拥有成熟的 GPU 端推理能力，包括高效的调度器、RadixAttention 缓存机制、结构化生成支持、多 GPU 张量并行等生产级特性。KTransformers 则拥有领先的 CPU 端 MoE 计算能力，包括 AMX 优化内核、NUMA 感知内存管理、Expert Deferral 等。两者结合，目标是实现"GPU 张量并行 + CPU-GPU 混合专家并行"的统一推理架构，在这种架构中，Dense 层（注意力、共享专家）受益于多 GPU 的高吞吐执行，而路由专家则灵活地在 CPU 和 GPU 之间调度，最大化硬件利用率。

集成的技术路线图包括多个阶段。第一阶段是基础集成，包括压缩张量格式支持、AMX 内核集成和 CUDA Graph 支持，已通过初始 PR 完成。后续阶段将逐步添加混合量化配置支持、更多权重格式（GPTQ、AWQ）、热度感知的专家分布策略、Expert Deferral 机制、投机解码支持以及更多模型的适配。

这一集成方向反映了推理引擎生态的一个重要趋势：不同引擎不再是完全竞争的替代品，而是可以在各自的优势领域进行互补和协作。KTransformers 的 CPU 端优化能力作为一个独立的内核库（kt-kernel），可以被包括 SGLang 在内的多个推理框架所集成，从而将 CPU-GPU 异构推理的优化普惠到更广泛的用户群体。

KTransformers 项目也在积极拓展其他方向。在模型覆盖方面，持续支持新发布的 MoE 模型（如 Kimi-K2 系列、GLM-5、MiniMax-M2.5 等），通常在模型发布的当天或次日即提供支持。在功能增强方面，KTransformers 推出了 kt-sft 模块，利用 CPU-GPU 异构计算实现超大 MoE 模型的本地微调——例如仅需 70GB GPU 显存 + 1.3TB RAM 即可对 DeepSeek-V3 671B 进行 LoRA 微调，吞吐量约 40 tokens/s。在硬件平台扩展方面，除了 Intel x86 + NVIDIA GPU 的主要平台外，KTransformers 还增加了对 AMD GPU（ROCm）、Intel Arc GPU（XPU）以及华为昇腾 NPU 的支持。

---

## 本章小结

KTransformers 代表了大模型推理领域中一种独特而重要的技术路线——通过充分利用 CPU 的大内存容量和现代指令集的计算能力，与 GPU 形成互补，使得在资源受限的环境中运行超大 MoE 模型成为可能。

本章的核心要点可以概括如下。在设计动机方面，MoE 模型的稀疏激活特性使其天然适合 CPU-GPU 异构推理。KTransformers 的目标是解决此前异构方案中 CPU 计算效率低下和 CPU-GPU 同步开销过大两个核心瓶颈。在计算划分方面，KTransformers 按照算术强度进行任务分配：高算术强度的注意力层和共享专家放在 GPU，庞大但稀疏激活的路由专家放在 CPU。在 CPU 计算优化方面，AMX Tiling 感知的内存布局、缓存友好的分块计算策略、以及 AMX/AVX-512 的动态选择机制，使得 CPU 端 MoE 计算性能达到了 PyTorch 基线的 3.98 倍。在 CPU-GPU 协调方面，基于 Host Function 的异步调度和全图 CUDA Graph 捕获消除了同步开销。Expert Deferral 机制通过重构执行顺序，将 CPU 利用率从 74% 提升到近 100%，带来高达 45% 的额外吞吐量提升，精度损失不超过 0.5%。在工程实现方面，模块注入框架和 YAML 配置驱动的设计使得系统灵活易扩展，与 HuggingFace 生态无缝兼容。在实际部署方面，KTransformers 使得在一块 24GB 消费级 GPU 加上约 400GB DRAM 的单机上运行 DeepSeek-V3/R1 671B 模型成为可能，Decode 速度达到 10-17 tokens/s（单用户），硬件成本仅为纯 GPU 方案的 5%-10%。

KTransformers 的成功也揭示了一个更深层的启示：推理优化不仅仅是关于如何让 GPU 跑得更快，也是关于如何更智慧地利用系统中所有可用的计算和存储资源。随着模型规模持续增长和 MoE 架构的日益普及，CPU-GPU 异构推理有望成为推理优化工具箱中不可或缺的一个组成部分。
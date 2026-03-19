# 第9章 vLLM：高吞吐推理引擎

vLLM 是当今大模型推理领域最具影响力的开源推理引擎之一。它诞生于一篇关于 KV Cache 高效管理的学术论文，在短短两年多的时间里迅速成长为支撑数百家企业生产环境的工业级系统。本章将从历史演进、架构设计、核心子系统、关键优化技术、分布式推理、多模态支持、性能调优等多个维度，对 vLLM 进行深入而系统的剖析。读者在阅读本章之前，应已掌握前八章中关于 Transformer 推理计算分析、KV Cache 原理、注意力机制优化、量化技术、投机解码以及调度与批处理的基础知识。本章将展示这些技术如何在一个真实的、高度工程化的推理系统中被有机地整合在一起。

---

## 9.1 vLLM 的发展历程：从 PagedAttention 论文到生产级系统

vLLM 的故事始于加州大学伯克利分校 Sky Computing Lab（后更名为 LMSYS）的一个核心观察：在大模型推理服务中，KV Cache 的显存管理是制约系统吞吐量的首要瓶颈。2023 年 6 月，Woosuk Kwon 等人发表了论文"Efficient Memory Management for Large Language Model Serving with PagedAttention"，提出了借鉴操作系统虚拟内存分页机制来管理 KV Cache 的思路。这篇论文的核心洞察在于：传统推理系统为每个请求预分配连续的、按最大可能序列长度估算的 KV Cache 空间，这导致了严重的内存碎片化和资源浪费。通过将 KV Cache 切分为固定大小的物理块（Block），并使用类似页表（Page Table）的映射结构进行动态管理，系统可以将 KV Cache 的显存利用率从此前普遍不足 40% 的水平提升至接近 96%。

伴随论文发布的开源项目 vLLM 立即引起了工业界的广泛关注。在 PagedAttention 的基础上，vLLM 实现了连续批处理（Continuous Batching），使得系统能够在每个推理迭代粒度上动态地插入新请求和移除已完成的请求，从而大幅提升了 GPU 的计算利用率。早期的 vLLM 相比当时广泛使用的 HuggingFace Transformers 原生推理和 NVIDIA FasterTransformer，在吞吐量上实现了数倍乃至一个数量级的提升，这使其迅速成为开源 LLM 推理的事实标准之一。

从 2023 年下半年到 2024 年，vLLM 经历了快速的功能迭代。社区和核心团队持续加入了对更多模型架构的支持（从最初的 LLaMA、GPT-2 等扩展到 Mixtral、Qwen、DeepSeek 等数十种架构），引入了张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism）以支持多 GPU 推理，集成了多种量化格式（GPTQ、AWQ、FP8 等），并实现了投机解码（Speculative Decoding）、自动前缀缓存（Automatic Prefix Caching）、Chunked Prefill 等一系列重要优化。与此同时，vLLM 项目的治理结构也日趋成熟，围绕它形成了活跃的开源社区，多家云服务商（如 Anyscale、RunPod、Together AI 等）将 vLLM 集成到其推理服务产品中。

然而，随着功能的不断叠加，vLLM 早期的架构（后来被称为 V0 架构）逐渐暴露出设计层面的问题。大量条件分支和历史兼容代码使得系统的可维护性下降，调度器和模型执行器之间的 IPC（进程间通信）开销在高吞吐场景下成为瓶颈，且多种优化技术之间的组合变得越来越复杂。2024 年末至 2025 年初，vLLM 团队启动了代号为 V1 的重大架构重构，目标是在不牺牲功能广度的前提下，实现更简洁、更高性能的统一架构。V1 架构的核心变化包括：将调度器与模型执行器合并到同一进程以消除 IPC 开销、默认启用 Chunked Prefill、重新设计 KV Cache 管理器使其天然支持前缀缓存、以及引入基于 Torch.compile 的图编译优化。

截至本书写作时（2025 年初），vLLM 正处于 V0 到 V1 的过渡期。V1 架构已在主要场景中展现出显著的性能优势，部分配置下相比 V0 有 1.3 到 1.7 倍的吞吐提升。同时，vLLM 团队还在推进 Disaggregated Prefill（Prefill-Decode 分离）、Expert Parallelism、NIXL 高性能网络传输层等面向更大规模部署的功能。从一篇学术论文到支撑全球众多组织关键推理负载的生产级系统，vLLM 的发展历程本身就是大模型推理优化这一领域快速演进的缩影。

---

## 9.2 整体架构设计

理解 vLLM 的架构设计是深入掌握其内部工作机制的前提。本节将分别介绍 V0 架构和 V1 架构的设计思路，帮助读者理解架构演进背后的工程权衡。

### 9.2.1 V0 架构：Engine + Worker + ModelRunner

vLLM V0 架构采用了经典的前后端分离设计，核心组件包括 LLMEngine（或 AsyncLLMEngine）、Scheduler、Worker 和 ModelRunner。这套架构的设计理念是将请求管理与模型执行解耦，使二者可以在不同的进程中独立运行。

**LLMEngine** 是 V0 架构的顶层入口，负责接收用户请求、协调调度器和工作进程、并向上层返回生成结果。对于离线批量推理场景，用户直接调用 `LLM` 类的 `generate()` 方法，该方法内部将请求提交给 LLMEngine 并同步等待结果。对于在线推理服务场景，vLLM 提供了 `AsyncLLMEngine`，它是 LLMEngine 的异步封装，与基于 FastAPI 的 HTTP 服务器配合工作，能够并发处理大量推理请求。`AsyncLLMEngine` 内部维护一个事件循环，不断从请求队列中取出待处理请求，驱动调度器产生执行计划，将计划发送给 Worker 执行，然后收集结果并通过 Server-Sent Events（SSE）流式地返回给客户端。

**Scheduler** 是 V0 架构中的核心决策组件。它维护三个请求队列：`waiting` 队列存放新到达但尚未获得 KV Cache 资源的请求，`running` 队列存放正在进行推理的活跃请求，`swapped` 队列存放因显存不足而被换出到 CPU 内存的请求。在每个调度步骤中，Scheduler 执行以下决策流程：首先检查 `swapped` 队列，如果有请求可以被换回且显存充足，则优先恢复这些请求；然后检查 `running` 队列中的请求是否有足够的显存继续生成下一个 Token，如果显存不足，则按照策略（通常是后进先出）将部分请求抢占并移入 `swapped` 队列；最后检查 `waiting` 队列，在剩余显存和批量大小限制内尽可能多地准入新请求。Scheduler 的输出是一个 `SchedulerOutput` 对象，包含本次迭代需要执行 Prefill 的请求列表和需要执行 Decode 的请求列表，以及对应的 Block Table 映射信息。

**Worker** 是模型执行的载体。在单 GPU 场景下只有一个 Worker，在多 GPU 张量并行场景下，每个 GPU 对应一个 Worker 进程。Worker 内部持有 `ModelRunner`，后者负责实际的模型前向计算。在 V0 架构中，Engine（包含 Scheduler）运行在主进程中，而 Worker 运行在由 Ray 或多进程框架管理的独立进程中。Engine 与 Worker 之间通过进程间通信（IPC）传递调度指令和执行结果。这种设计的优点是清晰地分离了调度逻辑和计算逻辑，使得调度器可以在 CPU 上进行复杂的资源管理决策，而不阻塞 GPU 计算。但其缺点也十分明显：IPC 本身带来不可忽略的延迟开销，尤其是在批量较小、每次迭代计算量较轻的场景下，IPC 开销可能占据总延迟的显著比例。

**ModelRunner** 是 Worker 内部负责模型前向计算的组件。它持有模型参数（已加载到 GPU 上），管理输入张量的准备（将 Token ID、位置编码、Block Table 等信息组装为模型输入），调用模型的 `forward()` 方法执行前向传播，然后从输出 logits 中采样得到下一个 Token。ModelRunner 还负责管理 CUDA Graph 的捕获和回放：对于固定形状的 Decode 批次，它预先捕获一组 CUDA Graph，在运行时直接回放以消除反复的 Kernel Launch 开销。

V0 架构的这套 Engine-Scheduler-Worker-ModelRunner 层级结构，在 vLLM 早期的快速发展中发挥了重要作用。它的模块化设计使得社区能够相对独立地在不同层面添加新功能——例如，新的调度策略只需修改 Scheduler，新的模型架构只需在 ModelRunner 中添加对应的模型实现，量化和投机解码等优化可以分别在模型层和 ModelRunner 层接入。然而，随着这些功能的不断增加，V0 架构中各模块之间的交互变得越来越复杂，大量的 `if-else` 分支散布在代码中，维护成本急剧上升。

### 9.2.2 V1 架构重构：简化与高性能的统一

V1 架构的设计目标可以用两个关键词概括：简化（Simplification）与性能（Performance）。vLLM 团队在总结 V0 运营经验后，确定了 V1 的几项核心设计原则。

第一个原则是**消除不必要的 IPC 开销**。在 V1 架构中，Scheduler 和 ModelRunner 被合并到同一个进程中运行。这意味着调度器的输出可以直接作为模型执行器的输入，无需经过序列化、进程间传递和反序列化的过程。对于单 GPU 场景，这一改变带来了显著的延迟降低。对于多 GPU 场景，V1 在每个 Worker 进程内部都运行一个本地的调度逻辑副本，Worker 之间仅同步必要的控制信息，最大限度地减少跨进程通信。

第二个原则是**以 Chunked Prefill 作为默认执行模式**。在 V0 中，Chunked Prefill 是一个需要显式启用的可选功能，且与其他功能的组合存在兼容性问题。V1 将 Chunked Prefill 作为基础执行模型：所有请求的 Prefill 和 Decode 在统一的迭代循环中被处理，长 Prefill 请求被自动分块，短 Prefill 和 Decode Token 被打包到同一个批次中。这样做的好处是消除了 V0 中 Prefill 批次和 Decode 批次之间的切换开销，使 GPU 始终有稳定且适量的工作负载，显著改善了 Decode 阶段的尾部延迟（tail latency）。

第三个原则是**重新设计 KV Cache 管理器，使前缀缓存成为一等公民**。V0 的 Block Manager 经历了从 V1 版到 V2 版的演变（注意这里的 Block Manager 版本号与 vLLM 整体的 V0/V1 架构版本号不同），但前缀缓存始终是作为附加功能叠加在基础分页逻辑之上的，需要通过命令行参数 `--enable-prefix-caching` 显式启用。V1 架构中的新 KV Cache 管理器从设计之初就原生支持前缀缓存：每个物理块都通过其内容的哈希值（Content Hash）进行标识，相同内容的 KV Cache 块自然地被不同请求共享。这使得 Automatic Prefix Caching（APC）成为默认行为，无需额外配置。新管理器还引入了更高效的引用计数和 LRU 淘汰机制，在高并发场景下的管理开销更低。

第四个原则是**深度集成编译优化**。V1 架构与 PyTorch 的 `torch.compile` 紧密集成，利用 TorchDynamo 进行计算图捕获，通过 TorchInductor 生成优化的 GPU Kernel。这使得 vLLM 能够自动享受算子融合、内存布局优化等编译器优化带来的性能提升，同时减少对手写 CUDA Kernel 的依赖，提高代码的可维护性和跨硬件的可移植性。

从部署者的视角来看，V1 架构的一个重要变化是命令行接口和配置参数的简化。许多在 V0 中需要精心调节的参数（如 `max_num_batched_tokens`、`enable_chunked_prefill`、`enable_prefix_caching` 等）在 V1 中要么已有合理的默认值，要么被自动推断，用户通常只需指定模型路径和基本的硬件配置即可获得接近最优的性能。这种"开箱即用"的设计理念大大降低了 vLLM 的使用门槛。

从实现层面看，V1 架构的核心执行循环可以概括为以下步骤：（1）调度器基于当前 KV Cache 资源状况和等待队列中的请求，决定本次迭代要处理的 Token 集合（包括新请求的 Prefill Token 和活跃请求的 Decode Token）；（2）将这些 Token 打包为一个批次，准备输入张量；（3）调用模型的 `forward()` 执行前向计算；（4）从输出 logits 中采样得到新 Token；（5）更新请求状态，检查终止条件，将已完成的请求从活跃队列中移除。这个循环看似简单，但其内部的每一步都经过精心优化：调度步骤与前一轮的采样步骤重叠执行以隐藏延迟，输入张量的准备尽可能利用预分配的缓冲区以避免反复的显存分配，CUDA Graph 在 Decode 阶段被自动使用以消除 Kernel Launch 开销。

值得注意的是，V1 架构在写作本书时仍在积极开发中，部分功能（如某些量化格式的支持、LoRA 适配器的热切换等）还在从 V0 向 V1 迁移的过程中。vLLM 团队通过环境变量 `VLLM_USE_V1=1`（后来改为默认启用）控制 V0 和 V1 之间的切换，确保用户在过渡期间可以根据需要选择合适的架构。

---

## 9.3 核心子系统解析

本节深入剖析 vLLM 的四个核心子系统：Scheduler、Block Manager、ModelRunner、以及 Tokenizer/Detokenizer。这些子系统的协同工作构成了 vLLM 推理引擎的完整执行管线。

### 9.3.1 Scheduler：请求调度与抢占

Scheduler 是 vLLM 的"大脑"，它在每个推理迭代中做出资源分配决策，决定哪些请求可以执行、哪些请求需要等待、哪些请求需要被暂时搁置。Scheduler 的设计质量直接影响系统的吞吐量、延迟分布和公平性。

在 V0 架构中，Scheduler 的核心数据结构是三个请求队列。`waiting` 队列（对应代码中的 `self.waiting`）使用双端队列（deque）存储所有等待被调度的新请求，按照到达顺序排列。`running` 队列（`self.running`）存储当前正在 GPU 上执行推理的活跃请求。`swapped` 队列（`self.swapped`）存储因显存压力而被换出到 CPU 内存的请求，其 KV Cache 已从 GPU 显存复制到主机内存。

每个调度步骤的执行逻辑遵循严格的优先级顺序。首先，Scheduler 尝试恢复 `swapped` 队列中的请求——这些请求此前因显存不足被换出，恢复它们的优先级高于接纳新请求，因为它们已经产生了部分计算结果，换入成本低于重新计算。恢复过程需要检查 GPU 上是否有足够的空闲物理块来容纳被换回的 KV Cache，以及当前批量大小是否允许增加新的 running 请求。其次，Scheduler 检查 `running` 队列中的请求能否继续执行。每个 Decode 步骤需要为每个活跃请求分配一个新的物理块（如果当前 Block 已满），如果可用物理块不足以支撑所有活跃请求的下一个 Token 生成，Scheduler 需要执行抢占。最后，Scheduler 从 `waiting` 队列中按 FCFS（先到先服务）顺序准入新请求，直到显存预算或批量大小上限被耗尽。

抢占（Preemption）是 Scheduler 中最复杂的机制之一。当显存不足时，Scheduler 需要选择一个或多个活跃请求暂时搁置，以释放它们占用的 KV Cache 物理块给其他请求使用。vLLM 的默认抢占策略是后进先出（LIFO）——最晚被调度的请求最先被抢占。这一策略的直觉是：后到达的请求通常已生成的 Token 较少，其 KV Cache 占用的显存较小，换出的成本较低。抢占有两种实现方式：**Swap**（将 KV Cache 从 GPU 复制到 CPU 内存，待条件允许时再复制回来）和 **Recompute**（直接丢弃 KV Cache，待请求被重新调度时从头计算 Prefill 阶段）。Swap 方式保留了已有的计算结果，但需要额外的 CPU 内存和 PCIe 带宽；Recompute 方式不需要额外内存，但会产生重复计算。vLLM 根据请求的 KV Cache 大小和当前的硬件条件自动选择抢占方式。

在 V1 架构中，Scheduler 的设计进行了显著简化。由于 Chunked Prefill 成为默认行为，Scheduler 不再需要区分"Prefill 批次"和"Decode 批次"——所有 Token（无论是 Prefill Token 还是 Decode Token）都在统一的框架下被调度。V1 的 Scheduler 使用一个统一的 Token 预算（Token Budget）来控制每个迭代的工作量：`max_num_batched_tokens` 指定每个迭代最多处理的 Token 总数，Scheduler 在这个预算内同时安排 Decode Token（每个活跃请求一个）和 Prefill Token（新请求或续传的 Prefill 块）。这种统一调度避免了 V0 中因长 Prefill 请求独占 GPU 而导致 Decode 请求延迟飙升的问题。

V1 的 Scheduler 还引入了与 KV Cache 管理器的更紧密集成。在 V0 中，Scheduler 需要显式地向 Block Manager 请求和释放物理块，这涉及复杂的状态同步。V1 中，KV Cache 管理器维护一个全局的块池（Block Pool），Scheduler 只需查询池中的可用块数量即可做出调度决策，块的分配和回收通过引用计数自动完成。这大大简化了 Scheduler 的代码复杂度。

### 9.3.2 Block Manager：KV Cache 的分页管理

Block Manager 是 PagedAttention 在工程层面的具体实现，负责管理 KV Cache 的物理块分配、映射表维护、以及块的共享与回收。它是 vLLM 区别于其他推理引擎的核心组件之一。

KV Cache 的分页管理以**物理块（Physical Block）**和**逻辑块（Logical Block）**的两层抽象为基础。每个物理块是 GPU 显存中一段连续的存储空间，可以容纳固定数量的 Token 的 Key 和 Value 张量。这个固定数量即 `block_size`，vLLM 默认使用 16，即每个物理块存储 16 个 Token 的 KV Cache。以一个具有 32 层、每层 32 个注意力头、每个头维度为 128 的模型为例，使用 FP16 精度时，一个 `block_size=16` 的物理块占用的显存为 $2 \times 32 \times 32 \times 128 \times 16 \times 2 = 8\text{MB}$（因子 2 分别对应 Key 和 Value，以及 FP16 的每元素 2 字节）。逻辑块是请求视角的抽象：每个请求看到的是一个从 0 开始编号的连续逻辑块序列，逻辑块到物理块的映射通过 **Block Table** 维护。Block Table 类似于操作系统中的页表，每个请求拥有一个 Block Table，记录其每个逻辑块对应的物理块编号。

当一个新请求到达时，Block Manager 为其 Prompt 中的 Token 分配所需数量的物理块。例如，一个包含 50 个 Prompt Token 的请求在 `block_size=16` 的设置下需要 $\lceil 50/16 \rceil = 4$ 个物理块。这 4 个物理块从全局空闲块池中分配，可以是不连续的——这正是分页管理的核心优势。在后续的 Decode 阶段，每当最后一个物理块被填满时（即新生成的 Token 需要的存储空间超出了当前块的剩余容量），Block Manager 再分配一个新的物理块。这种按需分配的方式避免了预分配整个最大序列长度所需内存的浪费。

**Copy-on-Write（写时复制）**机制是 Block Manager 支持 Beam Search 和 Parallel Sampling 等场景的关键。在 Beam Search 中，多个 Beam 可能共享相同的 Prompt 和部分生成前缀。如果为每个 Beam 独立复制一份完整的 KV Cache，显存消耗将随 Beam 数量线性增长。通过 Copy-on-Write，多个 Beam 可以共享相同的物理块，只有当某个 Beam 需要修改（即写入新的 KV 值到）一个被共享的物理块时，才会触发复制——将该物理块的内容复制到一个新分配的物理块中，然后更新对应 Beam 的 Block Table。这使得 Beam Search 的显存开销从 $O(B \times L)$（B 为 Beam 数，L 为序列长度）降低到接近 $O(L + B \times \Delta L)$（$\Delta L$ 为各 Beam 分叉后的增量长度）。

在 V0 架构中，Block Manager 经历了两个版本的演变。第一版 Block Manager（BlockManagerV1）使用显式的 `PhysicalTokenBlock` 和 `LogicalTokenBlock` 对象来管理块，每个块对象包含引用计数、所属请求等元信息。这种面向对象的设计清晰但在高并发场景下产生了可观的 Python 对象管理开销。第二版 Block Manager（BlockManagerV2，有时也被称为 BlockManagerV2 或 SelfAttnBlockSpaceManager）改用基于整数数组的块管理，通过 NumPy 数组而非 Python 对象来记录块状态，减少了 Python GIL 和垃圾回收的影响。

V1 架构的 KV Cache 管理器在 BlockManagerV2 的基础上做了进一步的简化和优化。最显著的变化是原生支持内容哈希（Content Hash）索引：每个物理块在其内容确定后（即块被填满或请求完成后），会计算一个基于块内 Token ID 序列的哈希值。当新请求的某个逻辑块的哈希值与已有物理块的哈希值匹配时，就直接复用该物理块，而不是重新分配和计算。这就是 Automatic Prefix Caching 的实现基础，我们将在 9.4.2 节中详细讨论。

### 9.3.3 ModelRunner：模型执行与 CUDA Graph

ModelRunner 是 vLLM 中直接与 GPU 交互的组件，负责将 Scheduler 的调度决策转化为实际的 GPU 计算操作。它的核心职责包括：输入张量准备、模型前向计算、输出采样、以及 CUDA Graph 管理。

**输入张量准备**是 ModelRunner 在每个推理迭代开始时的首要工作。Scheduler 输出的是逻辑层面的信息——哪些请求需要处理、每个请求处理多少个 Token、Block Table 映射等。ModelRunner 需要将这些信息转化为模型 `forward()` 方法所需的具体张量。主要的输入张量包括：`input_ids`（本次迭代需要处理的所有 Token 的 ID，按请求顺序排列）、`positions`（每个 Token 的位置编码索引，用于计算 Rotary Positional Embedding 等）、`block_tables`（当前批次中每个请求的 Block Table，告诉 Attention Kernel 去哪些物理块中读取历史 KV Cache）、以及 `slot_mapping`（指定当前生成的 KV Cache 应写入哪些物理块的哪些位置）。在 Chunked Prefill 模式下，批次中可能同时包含 Prefill Token 和 Decode Token，ModelRunner 还需要准备 `seq_lens` 张量来标识每个请求已有的上下文长度，以便 Attention Kernel 正确地计算注意力分数。

**模型前向计算**是 ModelRunner 的核心功能。vLLM 内部维护了一套模型定义层（位于 `vllm/model_executor/models/` 目录下），这些模型定义在 PyTorch 的 `nn.Module` 基础上针对推理场景进行了优化。以 LLaMA 模型为例，vLLM 的 `LlamaForCausalLM` 实现与 HuggingFace 原版的关键区别在于：（1）Attention 层使用 PagedAttention Kernel 而非标准的 `scaled_dot_product_attention`，以支持分页的 KV Cache 读写；（2）QKV 投影矩阵被融合为一个大的矩阵乘法，减少 Kernel Launch 次数；（3）Gate 和 Up 投影同样被融合；（4）使用 FlashInfer 或 FlashAttention 等高效 Attention Kernel 作为后端。模型的前向计算流程是逐层执行的：输入 Token Embedding → 逐层的 LayerNorm → Attention → Residual Connection → LayerNorm → FFN → Residual Connection → 最终 LayerNorm → LM Head → Logits。

**输出采样**在模型前向计算得到 logits 张量后执行。vLLM 的采样器（Sampler）支持多种采样策略，包括贪婪解码（Greedy Decoding，即选择概率最高的 Token）、Top-K 采样、Top-P（Nucleus）采样、温度缩放（Temperature Scaling）等，以及它们的组合。采样器还支持 `repetition_penalty`（重复惩罚）、`frequency_penalty`（频率惩罚）和 `presence_penalty`（存在惩罚）等参数来控制生成的多样性。对于 Beam Search 场景，采样器返回每个 Beam 的 Top-K 候选 Token 及其对数概率。在 V1 架构中，采样操作被进一步优化：对于简单的贪婪解码场景，直接使用 `torch.argmax` 在 GPU 上完成，避免将完整的 logits 张量传回 CPU。

**CUDA Graph** 是 ModelRunner 的一项重要优化。在 Decode 阶段，由于每个请求每次只生成一个 Token，单次前向传播的计算量相对较小，此时 GPU Kernel Launch 的 CPU 端开销可能成为整体延迟的显著组成部分。CUDA Graph 允许将一系列 GPU 操作（Kernel Launch、内存复制等）记录为一个图（Graph），然后在后续迭代中一次性回放整个图，从而将多次 Kernel Launch 的 CPU 开销压缩为一次图回放的开销。

vLLM 的 CUDA Graph 使用策略如下。在系统启动时（或在首次遇到特定批量大小时），ModelRunner 会为一组预定义的批量大小（通常是 1、2、4、8 … 直到某个最大值的 2 的幂次序列，以及一些中间值）分别捕获 CUDA Graph。捕获过程是：用虚拟输入张量执行一次完整的模型前向传播，CUDA 运行时记录所有的 GPU 操作形成一个图。在后续的推理迭代中，如果当前批次的实际大小恰好匹配某个预捕获的图，就直接使用图回放。如果不匹配，则选择不小于实际大小的最接近的预捕获图——多余的"填充"位置使用虚拟 Token，其计算结果将被忽略。CUDA Graph 的一个限制是它要求每次回放时的输入张量形状与捕获时完全一致，这就是为什么 vLLM 需要为多个不同的批量大小分别捕获图。

vLLM 提供了 `--enforce-eager` 命令行参数来禁用 CUDA Graph，强制使用 PyTorch 的 Eager Mode 执行。这在调试和性能分析时很有用，因为 CUDA Graph 的"黑盒"特性使得逐 Kernel 的 profiling 变得困难。在 V1 架构中，CUDA Graph 的管理进一步与 `torch.compile` 集成——`torch.compile` 在图捕获阶段自动应用算子融合等优化，使得图回放时执行的是经过编译器优化的 Kernel 序列。

### 9.3.4 Tokenizer 与 Detokenizer

Tokenizer 和 Detokenizer 在推理引擎中承担着文本与 Token ID 之间的转换工作。虽然它们的计算量远小于模型前向传播，但在高吞吐场景下，如果处理不当，它们同样可能成为性能瓶颈。

**Tokenizer** 将用户输入的文本字符串转换为模型能够理解的 Token ID 序列。vLLM 支持多种 Tokenizer 后端：默认使用 HuggingFace Transformers 提供的 Tokenizer（通过 `AutoTokenizer.from_pretrained()` 加载），也支持 SentencePiece 和 Tiktoken 等后端。对于高吞吐在线服务场景，vLLM 使用 `TokenizerGroup` 来管理多个 Tokenizer 实例，以支持并发的 Tokenization 请求。Tokenization 是一个 CPU 密集型操作（尤其是对于 BPE 类 Tokenizer），在处理大量短请求的场景下，其开销不可忽视。

**Detokenizer** 的任务看似简单——将模型生成的 Token ID 转换回文本字符串——但在流式输出场景下存在微妙的复杂性。问题在于，许多 Tokenizer（尤其是基于 BPE 的 Tokenizer）的 Token 不是一对一映射到可打印字符的。一个字符可能由多个 Token 组成（如 UTF-8 多字节字符），或一个 Token 可能跨越多个字符。如果简单地将每个生成的 Token 独立解码并立即发送给用户，可能产生乱码或不正确的文本。vLLM 的 Detokenizer 实现了**增量式解码（Incremental Decoding）**：它维护每个请求的已解码文本缓冲区，每次新生成一个 Token 后，将完整的 Token ID 序列（或其后缀）一起解码，然后与缓冲区中的已有文本比较，只输出新增的部分。这种方式确保了流式输出的文本在任何截断点都是合法的 UTF-8 文本。

在 V1 架构中，Detokenizer 被移到一个独立的线程中异步执行，使其不阻塞主推理循环。模型前向计算和采样完成后，新生成的 Token ID 被放入一个队列，Detokenizer 线程从队列中取出并异步解码。这一设计进一步减少了每个推理迭代的关键路径长度。

---

## 9.4 vLLM 的关键优化技术

在前几节介绍的核心架构之上，vLLM 集成了一系列关键优化技术来提升推理性能。本节将逐一深入讨论这些技术在 vLLM 中的具体实现。

### 9.4.1 PagedAttention 的工程实现细节

第 5 章已经从原理层面介绍了 PagedAttention 的分页思想，本节聚焦于其在 vLLM 中的工程实现细节。

PagedAttention 的核心挑战在于修改标准 Attention Kernel 以支持非连续的 KV Cache 读取。在标准 Attention 实现中，一个请求的所有 Key 和 Value 张量存储在连续的内存区域中，Kernel 可以通过简单的指针算术进行高效的内存访问。而在 PagedAttention 中，KV Cache 被分散存储在多个不一定连续的物理块中，Kernel 需要根据 Block Table 的映射关系动态地确定每个 Token 的 KV 数据所在的物理地址。

vLLM 中 PagedAttention Kernel 的 Decode 阶段实现（即每次只处理一个新 Token 的 Query）是理解其工程细节的最佳入口。这个 Kernel 接收以下输入：当前 Token 的 Query 向量、所有物理块中存储的 Key Cache 和 Value Cache（以一个大张量的形式存在，通过块号索引访问）、以及当前请求的 Block Table。Kernel 的执行逻辑如下。每个 CUDA Thread Block 负责处理一个注意力头的一个块范围内的 Key-Value 对。对于给定的 Query 向量 $\mathbf{q}$，Kernel 遍历 Block Table 中的所有物理块，从每个物理块中读取 Key 向量，计算 $\mathbf{q} \cdot \mathbf{k}^T$ 得到注意力分数，然后通过 Online Softmax 算法（参见 4.2.1 节）进行归一化，最后与对应的 Value 向量加权求和。

这个 Kernel 的性能关键在于如何最大化 GPU 的内存带宽利用率。在 Decode 阶段，Attention 计算是典型的访存密集型操作——需要从 HBM 中读取大量的 KV Cache 数据，但每个 Query 向量只执行一次点积计算。为了提高带宽利用率，vLLM 的 PagedAttention Kernel 采用了以下优化策略：（1）将每个物理块内的 Key 向量预先转置存储（Key Cache 的内存布局为 `[num_blocks, num_heads, head_dim/x, block_size, x]`，其中 `x` 是一个用于向量化内存访问的因子），使得计算 Query-Key 点积时的内存访问模式对 GPU 缓存更友好；（2）使用 Warp-Level 的并行化，每个 Warp 协同处理一个注意力头的部分 KV 数据，通过 Warp Shuffle 指令高效地进行 Warp 内通信；（3）利用 Shared Memory 缓存 Query 向量和中间结果，减少对 HBM 的重复访问。

在 Prefill 阶段，计算模式与 Decode 不同：Query 不再是单个向量，而是一个矩阵（包含 Prompt 中所有 Token 的 Query）。此时 Attention 计算的 Arithmetic Intensity 更高，属于计算密集型操作。vLLM 在 Prefill 阶段通常不使用自己的 PagedAttention Kernel，而是使用 FlashAttention 或 FlashInfer 等更适合计算密集场景的 Kernel。FlashInfer 库专为推理场景设计，原生支持分页的 KV Cache 布局，提供了高效的 Paged FlashAttention 实现。vLLM 可以通过 `--backend` 参数选择 Attention 后端。

随着 vLLM 从 V0 演进到 V1，以及 FlashInfer 等外部库的成熟，vLLM 内部的 Attention Kernel 策略也在不断调整。V1 架构倾向于使用统一的 Attention 后端（如 FlashInfer），同时处理 Prefill 和 Decode Token，利用 FlashInfer 提供的灵活的 Ragged Tensor 接口来处理 Chunked Prefill 中不规则的输入形状。

### 9.4.2 Prefix Caching（Automatic Prefix Caching）

在许多实际应用场景中，不同的推理请求共享相同的前缀。例如，在聊天应用中，同一个系统 Prompt（System Prompt）被附加到每个用户的消息前面；在代码补全场景中，多个请求可能基于相同的代码文件上下文；在 RAG（检索增强生成）应用中，不同请求可能检索到相同的文档片段作为上下文。如果每个请求都从头计算这些共享前缀的 KV Cache，会造成大量的重复计算。

vLLM 的 Automatic Prefix Caching（APC）机制自动检测并复用这些共享前缀的 KV Cache，无需用户显式管理。其核心思想是：以物理块的内容（即块内 Token ID 序列）作为块的唯一标识符，当两个请求的某个逻辑块包含完全相同的 Token 序列时，它们可以共享同一个物理块。

APC 的实现依赖于内容哈希表（Content Hash Table）。每个物理块在被填满后，Block Manager 计算该块中所有 Token ID 的哈希值（注意，哈希值的计算不仅包含当前块的 Token ID，还包含该块在请求中的位置和前面所有块的 Token 内容，以确保不同上下文位置的相同 Token 序列不会被错误地视为同一块）。这个哈希值作为 Key 存入哈希表，物理块的编号作为 Value。当新请求到达时，Block Manager 逐块计算其 Prompt 的内容哈希值，并在哈希表中查找匹配。如果命中，则直接将该物理块的引用计数加 1，并在新请求的 Block Table 中记录这个物理块的编号，无需重新计算 KV Cache；如果未命中，则分配新的物理块并执行正常的 Prefill 计算。

APC 的缓存淘汰采用 LRU（Least Recently Used）策略。当显存不足需要回收物理块时，Block Manager 优先淘汰引用计数为 0（即当前没有活跃请求使用）且最近最久未被访问的物理块。被淘汰的块从哈希表中移除，其物理内存回归空闲池。

APC 在以下场景中能带来显著的性能提升。在长系统 Prompt 场景中，假设系统 Prompt 包含 2000 个 Token，使用 `block_size=16` 时对应 125 个物理块，第一个请求完成后这些块被缓存；后续请求如果使用相同的系统 Prompt，可以直接跳过这 2000 个 Token 的 Prefill 计算，将 TTFT（首 Token 延迟）从数百毫秒甚至秒级降低到仅覆盖用户输入部分的微秒级。在多轮对话场景中，连续的对话轮次共享越来越长的历史上下文，APC 使得每一轮只需计算新增内容的 KV Cache。在多请求并发场景中，使用相同 Few-Shot 示例或文档上下文的请求群体可以共享大量的 KV Cache 物理块，有效降低整体的显存消耗，从而允许更大的并发批量。

在 V0 架构中，APC 需要通过 `--enable-prefix-caching` 参数显式启用，因为它会引入额外的哈希计算和缓存管理开销。在 V1 架构中，APC 成为默认行为，KV Cache 管理器从设计之初就以内容哈希为核心索引机制，即使在没有共享前缀的场景下，其开销也被优化到可以忽略的水平。

### 9.4.3 Chunked Prefill 的实现与调优

Chunked Prefill 是 vLLM V1 架构的核心执行策略之一，它解决了传统推理引擎中长 Prefill 请求阻塞 Decode 请求的问题。

在传统的推理调度中，Prefill 和 Decode 交替执行：当一个长 Prompt 的请求到达时，引擎需要在一个推理迭代中完成整个 Prompt 的 Prefill 计算。对于包含数千甚至数万 Token 的长 Prompt，一次 Prefill 可能持续数百毫秒到数秒。在此期间，所有正在 Decode 阶段的请求都必须等待，导致它们的 Inter-Token Latency 出现巨大的尖峰。这对于在线服务来说是不可接受的，因为用户会明显感受到生成过程中的"卡顿"。

Chunked Prefill 的核心思想是将长 Prefill 请求的 Prompt 分割为多个较小的块（Chunk），在多个推理迭代中逐块处理。每个迭代中，一部分 Token 预算分配给 Prefill 块，剩余预算分配给 Decode Token，二者在同一个批次中混合执行。例如，一个 4096 Token 的 Prompt 可以被分为 4 个 1024 Token 的块，在 4 个迭代中逐一计算，每个迭代中同时处理其他请求的 Decode Token。

在 vLLM 的 V1 实现中，Chunked Prefill 的关键参数是 `max_num_batched_tokens`，它控制每个推理迭代中处理的最大 Token 总数。Scheduler 在每个迭代的调度流程如下：首先为所有活跃的 Decode 请求各预留 1 个 Token 的预算；然后将剩余的 Token 预算分配给等待 Prefill 的请求——如果某个请求的剩余 Prefill Token 数超过了可用预算，就只取前面的部分作为一个 Chunk，其余部分在下一个迭代继续处理。

Chunked Prefill 的实现涉及一个重要的技术细节：当一个 Prefill 请求被分块时，前面 Chunk 计算产生的 KV Cache 需要被保存下来，以便后续 Chunk 的 Attention 计算能够看到完整的上下文。这意味着即使 Prefill 尚未完成，请求已经开始占用 KV Cache 物理块。Block Manager 需要为这些"正在进行中"的 Prefill 请求分配和管理物理块。在 V1 的实现中，这与正常的块分配逻辑统一处理——每个 Chunk 计算完成后，其 KV Cache 被写入对应的物理块，下一个 Chunk 的 Attention Kernel 通过 Block Table 访问这些已有的块以及正在计算的新块。

`max_num_batched_tokens` 的取值是影响性能的关键调优点。较大的值允许每个迭代处理更多的 Token，有利于提高 GPU 的计算利用率（因为更大的批次意味着更高的 Arithmetic Intensity），但也会增加单次迭代的延迟，对 Decode 请求的 ITL 产生更大的影响。较小的值使每次迭代更快完成，Decode 请求获得更低的 ITL，但可能导致 GPU 的计算单元利用不充分。在实践中，vLLM V1 的默认值通常设为 2048 到 8192 之间（具体取决于模型和硬件），用户可以根据延迟敏感程度进行调整。对于延迟敏感的在线服务，倾向于使用较小的值（如 2048）；对于吞吐优先的离线批量处理，可以使用较大的值。

Chunked Prefill 还为 Prefill-Decode 分离（PD 分离）提供了重要的基础设施支持。在 PD 分离架构中，Prefill 节点和 Decode 节点各自独立运行，Prefill 节点生成 KV Cache 并传输给 Decode 节点。Chunked Prefill 使得 Prefill 节点可以按块生成并逐块传输 KV Cache，实现计算与传输的流水线化，降低端到端延迟。

### 9.4.4 Speculative Decoding 支持

投机解码（Speculative Decoding）是加速自回归生成的一项重要技术（详见第 7 章），vLLM 从较早期就提供了对投机解码的原生支持。

vLLM 的投机解码实现遵循标准的 Draft-Then-Verify 框架。系统维护两个模型：一个较小的 Draft 模型用于快速地生成若干候选 Token（通常 3 到 7 个），一个较大的 Target 模型用于并行地验证这些候选 Token。验证过程利用了 Transformer 的并行化特性——Target 模型可以在一次前向传播中同时处理所有候选 Token（类似于 Prefill 阶段处理一个短序列），然后通过接受-拒绝采样（Acceptance-Rejection Sampling）确定哪些候选 Token 可以被接受。被接受的 Token 直接作为最终输出，无需逐个生成，从而实现加速。

在 vLLM 中启用投机解码需要指定 Draft 模型和每次投机的 Token 数量。通过命令行参数 `--speculative-model` 指定 Draft 模型的路径或名称，`--num-speculative-tokens` 指定每次 Draft 的 Token 数量（即投机长度 $K$）。例如：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B \
    --speculative-model meta-llama/Llama-3-8B \
    --num-speculative-tokens 5
```

这个配置使用 Llama-3-8B 作为 Draft 模型，在每个 Decode 步骤中先用 Draft 模型快速生成 5 个候选 Token，然后用 Llama-3-70B 一次性验证。

vLLM 的投机解码实现需要解决几个工程挑战。第一个挑战是 Draft 模型和 Target 模型的 KV Cache 管理。由于 Draft 模型生成的候选 Token 中只有部分会被接受，Draft 模型的 KV Cache 需要根据验证结果进行修剪——被拒绝的 Token 对应的 KV Cache 必须被丢弃。vLLM 通过在 Block Manager 中维护 Draft 模型和 Target 模型各自的 Block Table 来处理这个问题。

第二个挑战是批处理的适配。在非投机解码模式下，一个批次中的所有请求在每个迭代只生成一个 Token。而在投机解码模式下，批次中的不同请求可能接受了不同数量的候选 Token（有的接受了全部 5 个，有的只接受了 2 个），导致序列长度的增长不均匀。vLLM 的 Scheduler 需要考虑到这种不均匀性，在分配 KV Cache 块时预留足够的空间以容纳最大可能的接受长度。

第三个挑战是 Draft 模型与 Target 模型共享同一个 GPU 的资源竞争问题。在 vLLM 的标准实现中，Draft 模型和 Target 模型运行在同一组 GPU 上（Draft 模型较小，不需要额外的 GPU）。Draft 阶段和 Verify 阶段交替执行，这意味着 GPU 在 Draft 阶段可能利用不充分（因为 Draft 模型较小）。一些高级实现允许将 Draft 模型部署在独立的 GPU 上，或使用异步管线使 Draft 和 Verify 部分重叠，但这增加了系统复杂度。

除了传统的独立小模型作为 Drafter 之外，vLLM 还探索支持了其他 Draft 策略。例如，基于 N-gram 的投机（`--speculative-model [ngram]`）不使用任何额外的模型，而是根据已生成的 Token 序列中出现的 N-gram 模式来预测下一个 Token。这种方式的优点是完全没有 Draft 模型的显存和计算开销，缺点是对于复杂的生成任务接受率较低。vLLM 还支持 MLP Speculator（如 EAGLE 的变体），它使用一个轻量级的 MLP 头来预测多个未来的 Token，在 Draft 开销和接受率之间取得更好的平衡。

投机解码的加速效果高度依赖于 Draft 模型的接受率（Acceptance Rate），即 Draft 模型生成的候选 Token 被 Target 模型接受的比例。接受率受到多个因素的影响：Draft 模型与 Target 模型的能力差距（差距越大，接受率越低）、生成内容的难度（事实性内容的接受率通常高于创造性内容）、采样温度（温度越低、即生成越确定性时接受率越高）。在实践中，当接受率低于约 40% 到 50% 时，投机解码的加速可能被 Draft 阶段的额外开销抵消，导致总体性能反而下降。因此，vLLM 的文档建议用户在特定的模型组合和使用场景下实测投机解码的效果，而不是盲目启用。

### 9.4.5 Guided Decoding（结构化输出）

在许多实际应用中，用户需要模型的输出严格遵循特定的格式——例如有效的 JSON 对象、符合某个 JSON Schema 的数据、SQL 查询语句、或正则表达式约束的文本。vLLM 的 Guided Decoding 功能通过在采样阶段施加约束来保证输出的结构化合规性。

Guided Decoding 的基本原理是：在每个采样步骤，根据当前的约束状态，动态地计算一个"允许 Token 集合"（Allowed Token Set），然后将不在该集合中的 Token 的 logits 设置为负无穷（或一个极小的值），使它们的采样概率为零。这确保了每一步生成的 Token 都符合约束条件。

对于 JSON Schema 约束，vLLM 集成了外部的结构化生成库来完成约束状态的维护和允许 Token 集合的计算。主要支持的后端包括 Outlines 和 lm-format-enforcer。Outlines 的工作原理是将 JSON Schema 编译为一个有限状态机（Finite State Machine, FSM）或上下文无关文法（Context-Free Grammar, CFG），然后在每个采样步骤根据 FSM/CFG 的当前状态确定合法的下一个 Token 集合。lm-format-enforcer 采用类似但不同的实现方式，在某些场景下具有更好的性能。

通过 vLLM 的 OpenAI-Compatible API，用户可以使用 `response_format` 参数指定期望的输出格式。例如，要求模型输出一个符合特定 JSON Schema 的 JSON 对象：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Give me info about the Eiffel Tower."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "landmark_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "location": {"type": "string"},
                    "height_meters": {"type": "number"},
                    "year_built": {"type": "integer"}
                },
                "required": ["name", "location", "height_meters", "year_built"]
            }
        }
    }
)
```

Guided Decoding 的性能开销主要来自两个方面。第一是约束编译的一次性成本：将 JSON Schema 或正则表达式编译为 FSM/CFG 需要一定的预处理时间（通常在毫秒到百毫秒级别），但这只在首次遇到新约束时发生，编译结果会被缓存。第二是每步采样时计算允许 Token 集合的增量成本：这取决于约束的复杂度和 Vocabulary 的大小，对于简单约束通常只增加微秒级的开销，但对于非常复杂的嵌套 JSON Schema 可能更高。

vLLM 的 V1 架构对 Guided Decoding 进行了进一步优化，包括将约束状态的更新与模型前向计算重叠执行（在 GPU 计算模型前向的同时，CPU 异步地更新约束 FSM 状态并预计算下一步的 Token Mask），以及批量化处理同一约束下多个请求的 Token Mask 计算。

---

## 9.5 分布式推理支持

当模型规模超出单个 GPU 的显存容量，或当吞吐量需求超出单个 GPU 的计算能力时，就需要使用分布式推理。vLLM 提供了多种并行策略的支持，用户可以根据模型特点和硬件配置灵活选择。

### 9.5.1 Tensor Parallelism

张量并行（Tensor Parallelism, TP）是 vLLM 支持最成熟的分布式推理策略，也是大多数用户在多 GPU 推理时的首选方案。TP 将模型的每一层内部的参数矩阵沿特定维度切分到多个 GPU 上，每个 GPU 持有每一层的一个"切片"，所有 GPU 协同完成一层的计算后再进入下一层。

vLLM 的 TP 实现遵循 Megatron-LM 风格的切分方案。对于 Attention 层，QKV 投影矩阵按注意力头的维度切分——如果模型有 32 个注意力头且使用 TP=4，则每个 GPU 负责 8 个注意力头的计算。Attention 计算是本地的（每个 GPU 只计算其负责的注意力头），但输出投影矩阵的 All-Reduce 需要在所有 GPU 之间进行通信以汇总结果。对于 FFN 层（以 LLaMA 的 SwiGLU FFN 为例），Gate 和 Up 投影矩阵按列切分，每个 GPU 计算中间隐层的一个切片，然后 Down 投影矩阵按行切分，每个 GPU 的局部结果通过 All-Reduce 汇总。每一层的计算总共需要 2 次 All-Reduce 操作（Attention 输出投影后一次，FFN 输出投影后一次），这是 TP 的主要通信开销。

在 vLLM 中启用 TP 非常简单，只需指定 `--tensor-parallel-size`（简写 `-tp`）参数：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B \
    --tensor-parallel-size 4
```

vLLM 自动处理模型参数的切分和加载——它会检测到模型的参数维度和切分规则，在加载模型权重时直接将每个 GPU 应加载的部分读入对应的 GPU 显存，无需先加载完整模型再切分。

TP 的效率高度依赖于 GPU 间的互联带宽。All-Reduce 操作需要在所有 TP 组的 GPU 之间交换数据，数据量与每层的隐层维度成正比。对于隐层维度为 8192 的模型，每次 All-Reduce 传输约 $2 \times 8192 \times 2 = 32\text{KB}$（FP16，因子 2 来自 All-Reduce 的通信模型），这在 NVLink（带宽 900 GB/s for NVLink 4.0）上几乎可以忽略，但在 PCIe（带宽 64 GB/s for PCIe 5.0 x16）上可能成为瓶颈。因此，TP 最适合在同一服务器内通过 NVLink 互联的 GPU 之间使用。

vLLM 使用自定义的 All-Reduce 实现（Custom All-Reduce）来优化 NVLink 场景下的通信效率。与 NCCL 提供的通用 All-Reduce 相比，vLLM 的自定义实现针对小数据量的 All-Reduce（推理场景下 Decode 阶段的 All-Reduce 数据量通常很小）进行了优化，利用了 CUDA IPC（Inter-Process Communication）和直接的 P2P 内存访问来减少延迟。这一优化在 Decode 阶段尤为重要，因为此时每个 GPU 的计算量很小，All-Reduce 的延迟占总迭代时间的比例较大。

### 9.5.2 Pipeline Parallelism

流水线并行（Pipeline Parallelism, PP）将模型的不同层分配到不同的 GPU 上——例如，一个 80 层的模型使用 PP=4 时，GPU 0 负责第 0-19 层，GPU 1 负责第 20-39 层，GPU 2 负责第 40-59 层，GPU 3 负责第 60-79 层。数据按层序依次在 GPU 之间流动。

PP 的主要优势是通信模式简单——每个 GPU 只需与其上游和下游的邻居 GPU 通信（传递一层的激活值），通信量与 TP 的 All-Reduce 相当，但不需要所有 GPU 之间的全对全通信。这使得 PP 适合 GPU 之间通过带宽较低的 PCIe 或跨节点的 InfiniBand 互联的场景。

然而，PP 在推理场景中有一个固有的效率问题：流水线气泡（Pipeline Bubble）。在训练中，通过微批次（Microbatch）划分可以有效地填充流水线，使所有 GPU 保持忙碌。但在推理的 Decode 阶段，每个迭代的工作量非常小（每个请求只生成一个 Token），且计算必须严格按层序执行——Token 必须依次经过所有层才能得到最终的 logits 进行采样——这意味着在任何时刻，只有一个 GPU 在执行计算，其他 GPU 都在空闲等待。

vLLM 通过以下方式缓解 PP 的效率问题。对于较大的批次，一个迭代内多个请求的计算可以形成微流水线——当 GPU 0 完成第一个微批的前 20 层计算并将结果发送给 GPU 1 后，GPU 0 可以立即开始处理第二个微批。但这种微流水线化的效果取决于批次大小和每级 GPU 的计算延迟，在 Decode 阶段（计算量小）的收益有限。

在 vLLM 中，PP 通过 `--pipeline-parallel-size`（简写 `-pp`）参数启用：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B \
    --pipeline-parallel-size 4
```

PP 也可以与 TP 组合使用。例如，对于需要 8 个 GPU 的超大模型，可以使用 TP=4 × PP=2——每 4 个 GPU 组成一个 TP 组处理半个模型的每一层，两个 TP 组通过 PP 串联。这种组合通常将 TP 部署在高速 NVLink 互联的同一台服务器内的 GPU 上，PP 部署在跨服务器的 GPU 之间。

### 9.5.3 Data Parallelism

数据并行（Data Parallelism, DP）是最简单的多 GPU 推理策略：运行多个完全独立的模型副本，每个副本处理不同的请求。请求通过负载均衡器（Load Balancer）分发到不同的模型副本。

在 vLLM 中，DP 通常不需要特殊的引擎级支持——用户只需在多个 GPU 上分别启动多个 vLLM 实例，然后通过一个外部的负载均衡器（如 Nginx、HAProxy 或 Kubernetes Ingress）将请求分发到这些实例。每个实例独立运行，拥有自己的 Scheduler、Block Manager 和 ModelRunner。

DP 的优势是实现简单、线性扩展能力强（N 个副本理论上可以提供 N 倍的吞吐量），且完全没有 GPU 间通信开销。其劣势是每个模型副本需要完整的模型参数副本，因此模型必须能够放入单个 GPU（或单个 TP 组）的显存中。此外，不同副本之间的 KV Cache 不共享，这意味着前缀缓存等优化的效果被分散——如果一个请求被路由到副本 A 并生成了前缀缓存，另一个具有相同前缀的请求如果被路由到副本 B 则无法享受缓存命中。

为了缓解这个问题，vLLM 支持缓存感知的请求路由（Cache-Aware Routing）。当使用多个 DP 实例时，负载均衡器可以根据请求的前缀内容将其路由到最可能有缓存命中的实例。例如，使用一致性哈希（Consistent Hashing）将相同的系统 Prompt 或文档前缀映射到同一个实例。vLLM 的文档和社区提供了与各种负载均衡器集成的缓存感知路由方案。

DP 也可以与 TP/PP 组合使用。例如，在一个有 16 个 GPU 的集群中，可以运行 2 个 DP 副本，每个副本使用 TP=8。vLLM V1 架构正在探索在引擎内部原生支持 DP——即一个 vLLM 进程管理多个模型副本，内部实现请求分发和缓存共享。

### 9.5.4 Expert Parallelism（MoE 模型）

对于 Mixture of Experts（MoE）模型，vLLM 支持 Expert Parallelism（EP），将不同的专家（Expert）分布到不同的 GPU 上。

MoE 模型的每一层包含多个并行的 FFN 专家（例如 DeepSeek-V3 有 256 个 Routed Expert），但每个 Token 只激活其中的少数几个（如 8 个）。EP 将这些专家分布到多个 GPU 上——如果使用 EP=8 且模型有 256 个专家，每个 GPU 持有 32 个专家。当某个 Token 需要被路由到分布在不同 GPU 上的专家时，需要通过 All-to-All 通信将 Token 的激活值发送到对应的 GPU，计算完成后再通过 All-to-All 将结果收集回来。

EP 的通信开销与 Token 的路由模式密切相关。如果大多数 Token 恰好被路由到同一个 GPU 上的专家（即路由局部性好），则 All-to-All 通信量较小；如果路由分散，则通信量较大。vLLM 利用 DeepSeek 开源的 DeepEP 库来高效地实现 MoE 的 All-to-All 通信，该库针对 NVLink 和 InfiniBand 拓扑进行了优化。

EP 通常与 TP 或 DP 组合使用。例如，一种常见的配置是对 Attention 层使用 TP，对 MoE 层使用 EP，两者的并行度可以不同。vLLM 允许通过参数灵活配置这种混合并行策略。

### 9.5.5 Disaggregated Prefill 与 NIXL

Disaggregated Prefill（PD 分离）是 vLLM 正在积极开发的一项分布式优化，旨在从架构层面解决 Prefill 和 Decode 阶段的资源需求冲突（详见第 8 章 8.4 节）。

在 PD 分离架构中，集群中的 GPU 被划分为两类角色：**Prefill 节点**专门负责处理新请求的 Prefill 计算，**Decode 节点**专门负责已启动请求的逐 Token 生成。当一个新请求到达时，它被路由到一个 Prefill 节点，该节点完成 Prompt 的 KV Cache 计算后，将 KV Cache 传输到一个 Decode 节点，由后者继续 Decode 阶段的生成。

这种分离的核心价值在于：Prefill 阶段是计算密集型的，受益于高计算吞吐的 GPU 配置和大批量并行；Decode 阶段是访存密集型的，受益于高内存带宽和大显存容量。将两者分离后，可以针对各自的特点进行独立的资源配置和优化——例如，Prefill 节点可以使用 FP8 量化来最大化计算吞吐，Decode 节点可以使用较高的显存利用率来支持更多并发请求。更重要的是，分离后 Decode 节点上不再有长时间的 Prefill 计算来阻塞 Decode 请求，从而根本上消除了 Prefill 对 Decode 延迟的干扰。

PD 分离的关键技术挑战是 KV Cache 从 Prefill 节点到 Decode 节点的高效传输。对于一个长 Prompt（如 4096 个 Token）和一个大模型（如 70B 参数），KV Cache 的数据量可能达到数 GB，跨节点传输这些数据的延迟和带宽消耗不可忽视。vLLM 采用了 **Layerwise KV Cache Transfer**（逐层传输）策略来解决这个问题：Prefill 节点在计算完每一层的 KV Cache 后立即开始传输，而不是等到所有层都计算完毕。这样，后续层的计算与前面层的 KV Cache 传输可以重叠进行，显著降低端到端的传输延迟。

为了支撑高效的 KV Cache 传输，vLLM 团队开发了 **NIXL**（Network Interface for Cross-node LLM serving）——一个专为 LLM 推理服务设计的高性能网络传输层。NIXL 利用 RDMA（Remote Direct Memory Access）实现 GPU 显存之间的零拷贝传输，绕过 CPU 和操作系统内核，直接在 RDMA NIC 和 GPU 之间建立数据通路。NIXL 还支持 GPUDirect RDMA，使得数据可以直接从一个节点的 GPU 显存传输到另一个节点的 GPU 显存，无需经过主机内存的中转。

在 vLLM 的 PD 分离实现中，系统维护一个全局的请求管理器，负责将新请求路由到负载最低的 Prefill 节点，并在 Prefill 完成后将请求及其 KV Cache 的位置信息传递给选定的 Decode 节点。Decode 节点通过 NIXL 从 Prefill 节点的 GPU 显存中拉取（Pull）或接收 Prefill 节点推送（Push）的 KV Cache 数据，然后开始 Decode 阶段。

---

## 9.6 多模态模型推理支持

随着视觉语言模型（Vision-Language Model, VLM）如 LLaVA、QWen-VL、InternVL 等的兴起，支持多模态输入成为推理引擎的重要能力。vLLM 从较早版本开始就逐步扩展了对多模态模型的支持。

多模态推理的核心挑战在于处理非文本模态（主要是图像和视频）的输入。在典型的 VLM 架构中，图像首先通过一个视觉编码器（如 CLIP ViT、SigLIP 等）转换为一系列视觉 Token 的嵌入向量，这些嵌入向量然后与文本 Token 的嵌入向量拼接在一起，作为 Transformer 解码器的输入。从推理引擎的视角看，关键差异在于 Prefill 阶段：标准文本模型的 Prefill 只需处理文本 Token 的嵌入，而多模态模型需要首先执行图像编码（计算密集但只需执行一次），然后将视觉 Token 和文本 Token 一起送入 Transformer。

vLLM 的多模态支持涉及以下几个方面。在 API 层面，vLLM 的 OpenAI-Compatible API 支持 `image_url` 字段，用户可以在请求中传入图像的 URL 或 Base64 编码数据。在输入处理层面，vLLM 内部维护了一个多模态输入处理管线——接收到包含图像的请求后，图像数据被预处理（缩放、归一化等）并传入视觉编码器，得到的视觉 Token 嵌入被注入到对应的位置。在 KV Cache 管理层面，视觉 Token 和文本 Token 的 KV Cache 统一管理，不做区分。在 Scheduler 层面，视觉 Token 参与 Token 预算的计算——一张高分辨率图像可能对应数百个视觉 Token，这些都计入 `max_num_batched_tokens` 的预算。

vLLM 支持的多模态模型覆盖了主流架构，包括 LLaVA 系列、Qwen-VL 系列、InternVL 系列、Pixtral、Molmo 等。对于视频理解模型，vLLM 支持将视频帧序列作为多个图像输入处理。

多模态推理的性能优化重点在于视觉编码阶段的高效处理（如图像编码的批量化、视觉 Token 的缓存）以及视觉 Token 对 KV Cache 显存的影响管理。在多轮对话场景中，如果用户在多个轮次中引用同一张图像，APC 可以自动缓存视觉 Token 的 KV Cache，避免重复编码。

---

## 9.7 性能调优指南

vLLM 提供了丰富的配置参数来适应不同的部署场景。本节将重点讨论最常用和最关键的调优参数，帮助读者在实际部署中获得最佳性能。

### 9.7.1 关键参数解析

**`max_num_batched_tokens`** 控制每个推理迭代中处理的最大 Token 总数。在 V1 架构中（默认启用 Chunked Prefill），这个参数决定了每次迭代的工作量上限。较大的值有利于吞吐量（GPU 计算利用率更高），但会增加单次迭代的延迟，对 Decode 请求的 ITL 产生影响。较小的值有利于延迟（每次迭代更快完成），但可能导致 GPU 计算利用不充分。vLLM V1 对这个参数设置了合理的默认值（通常为 8192），大多数场景下不需要手动调整。对于延迟极敏感的场景（如实时对话），可以考虑降低到 2048；对于吞吐优先的离线批处理，可以提高到 16384 或更高。

**`max_num_seqs`**（或 `max_num_requests`）控制同时处理的最大请求数。这直接影响批量大小的上限。较大的值允许更多请求并发执行，有利于吞吐量，但也意味着每个请求可获得的 KV Cache 显存份额更少。vLLM 的默认值通常为 256，对于大多数在线服务场景已足够。

**`gpu_memory_utilization`** 控制 vLLM 可以使用的 GPU 显存比例，默认值为 0.9（即 90%）。vLLM 在启动时会根据模型参数大小和这个比例计算出可用于 KV Cache 的显存量，然后确定物理块的数量。较高的值允许更多的 KV Cache 块（支持更大的批量或更长的序列），但如果设置得太高，可能导致与 CUDA 运行时或其他进程争抢显存。如果遇到 OOM（Out of Memory）错误，可以适当降低这个值。如果系统是 GPU 的独占用户且需要最大化吞吐，可以尝试提高到 0.95。

**`max_model_len`** 指定模型支持的最大上下文长度。vLLM 默认使用模型配置中的 `max_position_embeddings` 值，但用户可以手动降低它以节省 KV Cache 显存。例如，如果一个模型支持 128K 上下文但实际请求不会超过 8K，设置 `--max-model-len 8192` 可以显著增加可用的 KV Cache 块数量（因为 vLLM 在某些计算中会以 `max_model_len` 作为基准进行显存规划）。

**`block_size`** 指定每个 KV Cache 物理块容纳的 Token 数量，默认值为 16。较小的 `block_size` 可以减少最后一个块的内部碎片（internal fragmentation），但会增加 Block Table 的大小和块管理的开销。较大的 `block_size` 减少管理开销但增加碎片。在大多数场景下，默认值 16 是一个良好的平衡。

**`dtype`** 指定模型参数和计算的精度。vLLM 支持 `float16`（FP16）、`bfloat16`（BF16）和 `float32`（FP32）。对于大多数推理场景，`bfloat16` 是推荐的选择——它的数值范围与 FP32 相同（有利于数值稳定性），精度略低于 FP16 但在大模型推理中差异通常可以忽略。如果模型原生使用 FP16 训练且已知在 FP16 下表现良好，使用 `float16` 也是合理的。使用 `float32` 会使显存消耗翻倍且计算速度大幅下降，仅在调试数值问题时使用。

**`swap_space`** 指定每个 GPU 可用于 KV Cache 换出（Swap）的 CPU 内存量（单位 GB），默认值为 4。这个参数影响抢占机制的行为——当 GPU 显存不足时，被抢占的请求的 KV Cache 会被换出到这块 CPU 内存中。较大的 swap space 允许更多请求被暂存而非重新计算，但会占用主机内存。

### 9.7.2 量化策略选择

vLLM 支持多种量化方案，选择合适的量化策略需要权衡模型精度、推理速度和显存节省。

**FP8 量化**（W8A8-FP8）是 H100 及后续 GPU 上推荐的首选量化方案。它将权重和激活都量化为 FP8 格式，利用 H100 Tensor Core 的原生 FP8 支持实现接近 2 倍的计算吞吐提升。FP8 量化的精度损失通常非常小（在大多数基准测试上与 FP16 几乎无差异），且不需要复杂的校准（Calibration）过程。在 vLLM 中，可以通过 `--quantization fp8` 加载 FP8 量化的模型。

**GPTQ** 和 **AWQ** 是最成熟的训练后量化（PTQ）方案，通常将权重量化到 INT4 精度（W4A16）。它们需要一个小的校准数据集来计算量化参数，但量化后的模型可以在 INT4 精度下运行，显存消耗降至 FP16 的约 1/4。vLLM 通过集成高效的量化 GEMM Kernel（如 Marlin Kernel）来加速 INT4 量化模型的推理。Marlin Kernel 针对 Ampere 和 Hopper 架构的 GPU 进行了深度优化，在 W4A16 的量化推理中可以接近甚至达到 FP16 推理的吞吐量，同时节省显存。在 vLLM 中，加载 GPTQ 或 AWQ 量化模型时只需指定模型路径（模型的量化配置信息包含在模型文件中），vLLM 自动选择合适的 Kernel。

**SmoothQuant**（W8A8-INT8）将权重和激活都量化为 INT8，通过将激活中的异常值"平滑"到权重中来改善激活的可量化性。它的精度通常优于 W4A16 方案，但显存节省较少（约为 FP16 的 1/2）。SmoothQuant 适合对精度要求较高但仍需一定显存节省的场景。

选择量化策略的一般原则是：如果硬件支持 FP8（H100/H200/B200），优先使用 FP8 量化；如果需要在较旧的 GPU（A100 等）上部署大模型且显存紧张，使用 GPTQ 或 AWQ 的 W4A16 量化；如果精度要求极高且显存允许，使用无量化的 BF16 推理。

### 9.7.3 CUDA Graph 与 `enforce_eager`

如 9.3.3 节所述，CUDA Graph 通过将多个 GPU Kernel 的 Launch 操作合并为一次 Graph Replay 来降低 Decode 阶段的延迟。vLLM 默认启用 CUDA Graph，这对于大多数场景是最优选择。

用户可能需要禁用 CUDA Graph（`--enforce-eager`）的场景包括：（1）调试和 profiling——CUDA Graph 将多个 Kernel 打包为一个不可分割的单元，使得 Nsight Systems 等工具难以分析每个 Kernel 的性能；（2）动态模型行为——某些模型使用了 CUDA Graph 不支持的动态控制流（如条件执行、动态形状），在这种情况下 vLLM 会自动回退到 Eager Mode，但显式指定 `--enforce-eager` 可以避免 Graph 捕获失败的尝试开销；（3）显存受限——CUDA Graph 的捕获需要为每个预定义的批量大小分别记录一份 Graph 副本，这会占用额外的 GPU 显存。在显存非常紧张的部署中，禁用 CUDA Graph 可以释放这部分显存用于 KV Cache。

在 V1 架构中，CUDA Graph 与 `torch.compile` 的集成使其更加高效——`torch.compile` 在图捕获阶段自动进行算子融合，使得 Graph 中的 Kernel 数量更少、每个 Kernel 的效率更高。这进一步增大了 CUDA Graph 带来的加速幅度。

一些额外的性能调优建议如下。对于**高并发在线服务**场景，推荐使用 V1 架构（默认 Chunked Prefill + APC），设置适中的 `max_num_batched_tokens`（如 4096-8192），使用 FP8 或 BF16 精度，启用 CUDA Graph。对于**离线批量推理**场景，推荐增大 `max_num_batched_tokens` 和 `max_num_seqs` 以最大化吞吐量，可以容忍较高的单请求延迟。对于**长上下文场景**，设置合理的 `max_model_len`（不超过实际需求），考虑使用 KV Cache 量化（`--kv-cache-dtype fp8_e4m3` 或 `fp8_e5m2`）来降低 KV Cache 的显存占用，必要时使用更大的 `gpu_memory_utilization`。对于**MoE 模型**（如 DeepSeek-V3/R1），考虑使用 Expert Parallelism 结合 Tensor Parallelism 来高效地分布专家，利用 FP8 量化来最大化 Decode 吞吐。

---

## 9.8 vLLM 源码导读与 Mini-vLLM 实验

深入理解 vLLM 的最佳方式是阅读其源码并动手实验。本节为读者提供一份源码导读路线图，并介绍一个 Mini-vLLM 教学实验的设计。

**源码导读路线图。** vLLM 的代码仓库（`github.com/vllm-project/vllm`）结构清晰，建议按以下顺序阅读。

第一站是**入口点与 API 层**。从 `vllm/entrypoints/openai/api_server.py` 开始，了解 HTTP 服务器如何接收请求并将其转化为内部的推理请求对象。然后查看 `vllm/entrypoints/llm.py` 中的 `LLM` 类，了解离线推理的入口。

第二站是**引擎层**。阅读 `vllm/engine/` 目录下的代码。V0 的 `llm_engine.py` 展示了 Engine-Scheduler-Worker 的完整交互流程。V1 的 `engine_core.py`（路径可能因版本而异）展示了简化后的核心执行循环。重点关注 `step()` 方法——这是每个推理迭代的入口。

第三站是**调度器**。`vllm/core/scheduler.py`（V0）或 V1 对应的调度器文件展示了三队列调度、抢占策略、Token 预算分配的完整逻辑。这是理解 vLLM 资源管理策略的关键文件。

第四站是**Block Manager**。`vllm/core/block_manager.py` 及其子模块展示了分页 KV Cache 管理的实现。重点关注物理块的分配与回收、Block Table 的维护、以及 Prefix Caching 的哈希表逻辑。

第五站是**模型执行层**。`vllm/worker/` 目录下的 `model_runner.py` 展示了输入张量准备、模型前向调用、CUDA Graph 管理和输出采样的完整流程。`vllm/model_executor/models/` 目录下包含各种模型的推理优化实现，选择一个熟悉的模型（如 `llama.py`）深入阅读，理解 PagedAttention Kernel 调用、QKV 融合、SwiGLU 实现等推理优化细节。

第六站是**Attention 后端**。`vllm/attention/` 目录下展示了不同 Attention Kernel 后端（FlashAttention、FlashInfer、PagedAttention 等）的抽象接口和具体实现。理解这些后端如何与分页的 KV Cache 布局交互是掌握 vLLM 底层性能的关键。

**Mini-vLLM 教学实验。** 为了帮助读者从零开始理解推理引擎的核心原理，我们建议读者尝试实现一个简化版的 Mini-vLLM，它包含以下核心功能：

实验一：实现一个基本的 KV Cache Block Manager。用 Python 实现一个简化的分页 KV Cache 管理器，包括物理块池的初始化、块的分配与回收、Block Table 的维护。不需要真正的 GPU 内存操作，使用 NumPy 数组模拟即可。验证 Copy-on-Write 在 Parallel Sampling 场景中的正确性。

实验二：实现一个基本的 Continuous Batching Scheduler。在实验一的基础上实现三队列调度逻辑（waiting、running、swapped），包括请求准入、Token 预算分配和简单的抢占策略。模拟多个请求并发到达并跟踪其状态变化。

实验三：将 Scheduler 与真实的模型推理连接。使用 HuggingFace Transformers 加载一个小模型（如 GPT-2 或 TinyLlama），在实验一和二的基础上实现完整的推理循环——Scheduler 产生调度决策，ModelRunner 准备输入并执行模型前向传播，采样器生成下一个 Token，然后反馈给 Scheduler 更新状态。

实验四：实现 Prefix Caching。在 Block Manager 中加入内容哈希索引，当新请求的前缀与已有缓存匹配时跳过对应的 Prefill 计算。使用一组具有共同系统 Prompt 的请求来测试缓存命中率和性能提升。

实验五：实现 Chunked Prefill。修改 Scheduler 使其将长 Prefill 请求分块，与 Decode Token 混合在同一迭代中处理。比较启用和禁用 Chunked Prefill 时 Decode 请求的 ITL 分布。

通过这五个渐进式实验，读者可以建立起对推理引擎核心机制的直觉理解。每个实验的代码量在 200-500 行 Python 之间，总计约 1500-2000 行，覆盖了 vLLM 最核心的设计思想。完整的实验代码和参考实现将在本书的配套代码仓库中提供。

---

**本章小结。** vLLM 从 PagedAttention 的核心创新出发，经过 V0 到 V1 的架构演进，已经发展为一个功能丰富、性能卓越的生产级推理引擎。它的核心设计理念——通过借鉴操作系统的虚拟内存管理来高效管理 KV Cache、通过连续批处理最大化 GPU 利用率、通过 Chunked Prefill 统一 Prefill 和 Decode 的执行、通过 Automatic Prefix Caching 自动复用计算结果——代表了当前大模型推理系统的最佳实践之一。同时，vLLM 在分布式推理（TP/PP/EP/PD 分离）、量化支持、投机解码、结构化输出、多模态推理等方面的全面能力，使其能够适应从单 GPU 本地部署到大规模分布式服务的各种场景。在下一章中，我们将转向 SGLang——一个在设计哲学上与 vLLM 互补的推理引擎，它以结构化生成语言和智能缓存为核心特色，在特定场景下展现出独特的优势。
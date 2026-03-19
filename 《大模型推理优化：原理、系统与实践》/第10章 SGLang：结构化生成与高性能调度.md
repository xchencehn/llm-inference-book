# 第10章 SGLang：结构化生成与高性能调度

---

大模型推理引擎的设计空间中，存在一个根本性的张力：用户希望以灵活的、程序化的方式编排多步骤的 LLM 调用（例如多轮对话、思维链、工具调用、结构化 JSON 输出），而底层运行时追求的是极致的批处理效率和资源利用率。大多数推理引擎从运行时出发，将用户交互视为一个个独立的 Completion 请求；而大多数应用框架从编程接口出发，将底层推理视为可以随意调用的黑盒函数。二者之间的鸿沟导致了大量的优化机会被白白浪费——例如多次调用之间的前缀重复计算、结构化输出中的冗余 Token 生成、以及多分支推理中的 KV Cache 复用失败。

SGLang（Structured Generation Language）正是为了弥合这一鸿沟而设计的系统。它由加州大学伯克利分校 LMSYS 团队于 2024 年初提出，在设计上做出了一个独特的选择：同时构建一套前端领域特定语言（DSL）和一套高性能后端运行时，并让二者深度协同。前端通过显式的程序结构暴露出多调用之间的依赖关系和复用机会，后端利用这些信息在 KV Cache 管理、调度策略和约束解码等层面做出全局优化。这种"语言-运行时协同设计"的哲学使 SGLang 在多类典型工作负载上取得了显著的性能优势。

本章将从 SGLang 的设计哲学出发，依次深入解析其前端语言、后端运行时架构、核心优化技术、对 Reasoning Model 和多模态模型的支持，以及与 KTransformers 的混合部署实践。最后，我们将通过 Mini-SGLang 教学实现和源码导读，帮助读者建立对这一系统的深入理解。

---

## 10.1 SGLang 的设计哲学：前端语言 + 后端运行时

理解 SGLang 的设计，首先需要理解它试图解决的核心问题。在典型的 LLM 应用中，一个用户请求往往不是单次生成就能完成的。以一个常见的"带有结构化输出的多轮推理"场景为例：用户提交一段长文档，系统需要先提取关键信息，然后基于提取结果生成一段摘要，接着将摘要格式化为特定的 JSON Schema，最后对 JSON 中的某些字段进行二次验证。这个过程涉及至少四次 LLM 调用，每次调用都共享一部分上下文前缀，且后续调用依赖前序调用的输出。

在传统的推理服务架构中，这四次调用会被当作四个独立的 HTTP 请求发送到推理引擎。每次请求都需要重新处理完整的提示词（即便大部分前缀相同），每次请求之间的 KV Cache 无法复用，而且客户端需要在每次调用之间做字符串拼接和条件判断。这种"无状态调用"模式的效率损失是系统性的。


SGLang 的设计哲学可以用一句话概括：**通过前端语言与后端运行时的协同设计，系统性地挖掘 LM 程序（Language Model Programs）中的结构化信息，实现全局优化**。

这一哲学体现在三个层面。第一，前端以 Python 嵌入式 DSL 的形式暴露出多次 LLM 调用之间的依赖关系、并行机会和前缀共享模式，使得后端能够"看到"用户意图的结构，而非被迫将每个请求视为孤立的黑盒。第二，后端运行时据此实现了基于 Radix Tree 的 KV Cache 自动复用（RadixAttention）、基于压缩有限状态机的快速约束解码、以及前端提示（Frontend Hint）驱动的缓存调度策略。第三，前端和后端可以独立使用——前端可以对接 OpenAI API 等黑盒端点，后端也可以作为标准的 OpenAI-Compatible 推理服务器单独运行——但二者结合时会产生协同增益。

SGLang 最初的论文发表于 2023 年 12 月（Zheng et al., arXiv:2312.07104），由来自斯坦福大学和加州大学伯克利分校的研究团队联合开发。论文的第一版在标准的 LLM 工作负载上实现了相比 vLLM 和 Guidance 高达 6.4 倍的吞吐量提升。此后，SGLang 经历了多个重要版本迭代：v0.3（2024 年 9 月）引入了 DeepSeek MLA 优化和 Torch.compile 集成，v0.4（2024 年 12 月）实现了零开销批调度器、缓存感知负载均衡器和 XGrammar 快速结构化输出，后续版本进一步增加了 PD 分离、EAGLE 投机解码、KTransformers 异构集成等功能。截至本书撰写时，SGLang 已经成长为一个功能完备的生产级推理引擎，被广泛部署在 LMSYS Chatbot Arena 等大规模服务中。

---

## 10.2 前端：结构化生成语言

### 10.2.1 SGLang DSL：gen、select、fork、join

SGLang 的前端是一种嵌入在 Python 中的领域特定语言（Domain-Specific Language, DSL）。它不改变 Python 的语法，而是通过一组精心设计的原语（primitives）来控制 LLM 的生成过程。这些原语可以与 Python 的原生控制流（if/else、for 循环、函数调用）无缝混合，使得开发者能够用熟悉的编程范式构建复杂的 LM 程序。

SGLang 的核心原语包括以下几类：

**状态管理与文本追加。** SGLang 使用一个 prompt state 对象 `s` 来管理当前的提示词状态。开发者可以用 `+=` 运算符或 `extend` 函数向状态中追加字符串。例如，`s += "You are a helpful assistant.\n"` 会将这段系统提示追加到当前上下文中。`s["variable_name"]` 用于获取之前生成的结果。

**生成原语 `gen`。** `gen` 是最核心的原语，用于调用 LLM 生成内容并将结果存储在一个命名变量中。它支持多种控制参数：`stop` 指定停止条件，`max_tokens` 限制生成长度，`regex` 指定输出必须满足的正则表达式约束（用于结构化输出）。例如：

```python
s += "Name: " + sgl.gen("name", stop="\n")
s += "Age: " + sgl.gen("age", regex=r"\d{1,3}")
```

这段代码先让模型生成一个名字（遇到换行停止），然后生成一个年龄（必须是 1 到 3 位数字）。`gen` 调用是非阻塞的，允许后续的 Python 代码继续执行，类似于异步 CUDA 内核启动。

**选择原语 `select`。** `select` 让模型从给定的候选列表中选择概率最高的选项。这在多选题、分类任务等场景中非常有用。例如：

```python
s += "Is the essay related to the image? "
s += sgl.select("related", choices=["Yes", "No"])
```

`select` 的实现会将每个候选项分别拼接到当前上下文后计算对数概率，然后选择概率最高的选项。这比让模型自由生成再做字符串匹配要可靠得多。

**并行控制原语 `fork` 和 `join`。** `fork` 创建当前提示词状态的多个并行副本，`join` 等待所有副本完成并收集结果。这使得程序内部的并行化变得简洁。例如，在 Branch-Solve-Merge 提示技术中：

```python
forks = s.fork(3)
for i, f in enumerate(forks):
    f += f"Evaluate from dimension {i}: "
    f += sgl.gen("judgment", max_tokens=256)
forks.join()
```

这段代码将当前上下文分叉为三个副本，每个副本独立评估一个维度，然后合并结果。在运行时层面，三个分叉的请求共享相同的前缀 KV Cache，通过 RadixAttention 实现零冗余复用。

**多模态原语。** SGLang 还提供了 `image` 和 `video` 原语来处理多模态输入，与文本原语无缝组合。

一个完整的 SGLang 程序示例是论文中的多维度作文评判器。这个程序接收一张图片和一篇作文，首先判断作文是否与图片相关，如果相关则分叉为三个维度进行并行评估，合并评判后生成摘要，最后按照正则表达式约束输出一个字母等级。整个程序用 SGLang 编写只需约 20 行代码，而等价的 OpenAI API 调用代码需要约 2.1 倍的行数，还要处理大量的字符串拼接和手动并行控制。

### 10.2.2 多调用协同的计算图表达

SGLang 程序的执行有两种模式：解释器模式和编译器模式。

在解释器模式下，每个 SGLang 函数运行在一个后台线程中的流执行器（stream executor）上。提示词状态被视为一个异步流，`extend`、`gen`、`select` 等原语被提交到流中进行异步执行。这类似于 CUDA 编程中的异步内核启动——调用不会阻塞 Python 线程的执行，只有在显式获取生成结果（例如读取 `s["name"]`）时才会发生同步。这种设计天然支持程序内部的并行化：多个 `fork` 分支可以同时提交生成请求，运行时批处理器会将它们打包在一起高效执行。

在编译器模式下，SGLang 程序可以被追踪（trace）为计算图，然后由图执行器进行更深层次的优化。图表示使得运行时能够提前分析所有调用之间的依赖关系和共享模式，从而做出更优的调度决策。例如，编译器可以识别出哪些生成调用共享相同的前缀，并提前向运行时发送"Frontend Hint"，确保 Radix Tree 中的缓存结构被正确建立。

从运行时优化的角度看，SGLang 程序的多调用结构暴露出三类关键优化机会。第一类是 KV Cache 复用机会：多次调用共享的前缀（如系统提示、Few-shot 示例、对话历史）可以通过 RadixAttention 自动识别和复用，避免重复计算。第二类是约束解码加速机会：当输出被正则表达式或 JSON Schema 约束时，压缩有限状态机可以在确定性转移路径上一次性解码多个 Token。第三类是 API 投机执行机会：在调用黑盒 API（如 OpenAI GPT-4）时，SGLang 可以让前一个生成调用在遇到停止条件后继续生成几个额外的 Token，并尝试将这些额外输出与后续原语模板匹配，从而节省一次 API 调用的延迟和输入 Token 费用。

### 10.2.3 与 LangChain/LlamaIndex 等框架的对比

理解 SGLang 在 LLM 编程生态中的定位，有助于把握其独特价值。LLM 编程系统可以按抽象层次分为高层和低层两类。

高层框架如 LangChain 和 LlamaIndex 提供预定义的抽象组件（如 Chain、Agent、Retriever），开发者通过组装这些组件来构建应用。DSPy 更进一步，通过自动提示优化来改善效果。这些框架关注的是应用层面的编程便利性，通常将底层推理引擎视为可以随意替换的黑盒。它们的优势在于快速搭建原型和丰富的生态集成，但劣势在于对运行时性能缺乏控制——每次 LLM 调用都是一次独立的 API 请求，框架无法跨调用优化。

低层系统如 LMQL、Guidance 和 SGLang 则直接操控提示词和生成过程。LMQL 是一种查询语言，提供了类 SQL 的声明式语法来控制 LLM 生成，但其运行时优化有限，主要使用 Hugging Face Transformers 作为后端。Guidance 提供了一种模板化的语法来交织文本和生成指令，但在批处理和并行化方面存在不足。

SGLang 的独特之处在于它是唯一一个**同时设计前端语言和后端运行时**的低层系统。这种协同设计的优势在于：前端可以向运行时传递结构化的优化提示（如前缀共享信息），而运行时的优化能力（如 RadixAttention）又反过来使得前端的高层抽象（如 `fork`）不会引入性能损失。实际基准测试表明，这种协同设计带来的性能差异是巨大的——在 Few-shot 学习、多轮对话、Tree-of-Thought 等典型多调用场景中，SGLang 的吞吐量可以达到 LangChain + vLLM 组合的数倍。

值得注意的是，高层框架和低层系统并不矛盾——DSPy 已经将 SGLang 作为可选后端集成，从而在保持 DSPy 自动提示优化能力的同时享受 SGLang 的运行时加速。

---

## 10.3 后端运行时架构

### 10.3.1 整体架构：Tokenizer → Scheduler → ModelRunner → Detokenizer

SGLang 的后端运行时（SGLang Runtime, SRT）是一个完整的推理服务系统，其整体架构遵循标准的多阶段流水线设计，但在每个阶段都嵌入了针对 LM 程序的特殊优化。

系统的最外层是一个 FastAPI 服务器，提供 OpenAI-Compatible 的 HTTP API（包括 `/v1/chat/completions`、`/v1/completions` 和 `/generate` 等端点），支持流式和非流式输出。请求到达后首先进入 Tokenizer 模块，将输入文本转换为 Token 序列。对于多模态模型，Tokenizer 还会处理图像和视频的预处理和 Token 化。

Token 化后的请求进入核心调度器（Scheduler）。调度器运行在 CPU 上，负责以下关键职责：管理请求队列和生命周期（新到请求的加入、已完成请求的退出、被抢占请求的暂存）、进行 KV Cache 分配和回收（通过 RadixAttention 的 Radix Tree 进行前缀匹配和缓存管理）、组装每个迭代步（iteration step）的批次（batch），以及准备 ModelRunner 所需的全部元数据（Token ID、Position ID、Block Table 等）。

组装好的批次被送入 ModelRunner 执行。ModelRunner 管理模型的前向计算过程，包括 CUDA Graph 的捕获和回放、Attention Kernel 的调度（使用 FlashInfer 或 FlashAttention 作为后端）、以及采样（Sampling）逻辑。ModelRunner 输出的是每个请求下一个 Token 的 ID。

生成的 Token ID 被送入 Detokenizer，增量地转换回文本字符串，然后通过流式响应返回给客户端。

这个流水线的关键特点在于调度器和 ModelRunner 之间的交互方式。在 SGLang 的架构中，调度器和 ModelRunner 运行在同一个进程中（不像某些系统将二者分离到不同进程），这减少了进程间通信的开销。同时，调度器直接管理 Radix Tree 和内存池，无需像 vLLM V0 那样在独立的 Engine 进程和 Worker 进程之间传递复杂的调度决策。

### 10.3.2 零开销 CPU Scheduler 设计

在 LLM 推理中，虽然 GPU 承担了模型前向计算的主要负载，但 CPU 上的工作同样不可忽视。批调度、内存分配、Radix Tree 操作、Token 处理等 CPU 端开销如果处理不当，会导致 GPU 在每个迭代步之间出现空闲间隙（idle gap），严重影响整体吞吐量。研究表明，一个未经优化的推理引擎可能将多达一半的时间花费在 CPU 开销上。

SGLang v0.4 引入的零开销批调度器（Zero-Overhead Batch Scheduler）通过一个简洁而有效的思想解决了这个问题：将 CPU 调度与 GPU 计算在时间上重叠（overlap）。具体来说，调度器总是"领先一步"运行——当 GPU 正在执行第 N 批次的前向计算时，CPU 调度器已经在为第 N+1 批次做准备工作，包括从等待队列中选择请求、进行 Radix Tree 的前缀匹配、分配 KV Cache 物理块、以及组装批次元数据。当 GPU 完成第 N 批次后，第 N+1 批次的全部准备工作已经就绪，可以立即启动，从而消除了 GPU 空闲时间。

这一思想最初由 NanoFlow（2024）提出，SGLang 在工程实现上做了精细的处理。实现的核心挑战在于解决数据依赖：第 N+1 批次的组装需要知道第 N 批次中哪些请求已经完成、每个请求生成了什么 Token。SGLang 通过创建"future token"占位符来解决这个问题——调度器在组装下一批次时，假设当前批次中的请求会继续存在，并为它们预留位置。当 GPU 计算完成并产出实际 Token 后，再通过精心设计的 CUDA Event 和同步机制来更新状态。如果某个请求实际上已经完成（生成了 EOS Token），则在下一轮调度时将其移除。

使用 NVIDIA Nsight Systems 进行的性能分析（profiling）验证了零开销的声明：在连续的五个解码步骤中，GPU 上没有出现任何空闲时间，所有 CPU 开销都被完全隐藏在 GPU 计算的背后。实测结果表明，零开销调度器在 SGLang v0.3 基础上带来了约 1.1 倍的吞吐量提升，相比其他推理引擎则有 1.3 倍的提升，且加速效果在小模型和大张量并行度场景下最为显著。

这一优化默认启用，用户无需做任何配置。如需进行对照实验（ablation study），可以在启动服务器时添加 `--disable-overlap` 参数来禁用重叠调度。

### 10.3.3 RadixAttention 的实现细节

RadixAttention 是 SGLang 运行时最核心的创新，也是其性能优势的主要来源之一。本节深入解析其数据结构设计、操作流程和缓存策略。关于 RadixAttention 的原理性介绍已在第 5 章中给出，这里侧重于系统实现层面的细节。

**数据结构设计。** RadixAttention 使用 Radix Tree（基数树）来管理所有请求的 KV Cache。Radix Tree 是 Trie（前缀树）的空间效率变体，其边可以标注不仅仅是单个元素，还可以是任意长度的元素序列。在 SGLang 的实现中，Radix Tree 的节点代表 Token 序列的分叉点，边上标注从父节点到子节点的 Token 子序列，对应的 KV Cache 张量以分页布局（paged layout）存储在 GPU 显存中，每个页的大小对应一个 Token。

Radix Tree 本身存储在 CPU 内存中，维护开销极小。每个节点维护一个引用计数器（reference counter），记录有多少正在运行的请求正在使用该节点对应的 KV Cache。当引用计数为零时，节点成为可淘汰的候选。

**操作流程。** 当一个新请求到达时，运行时执行以下步骤：首先，在 Radix Tree 中进行前缀匹配（prefix matching），找到与新请求的输入 Token 序列匹配的最长前缀。匹配成功的部分可以直接复用已有的 KV Cache，无需重新计算。然后，对不匹配的后缀部分进行正常的 Prefill 计算。计算完成后，新的 KV Cache 被插入到 Radix Tree 中，并且生成结果的 KV Cache 也会被保留（而不是在请求结束后丢弃），以便后续请求复用。

这个过程可以用一个具体的例子来说明。假设系统先后处理了同一个用户的两轮对话。第一轮的完整上下文（系统提示 + 用户消息 + 模型回复）被作为一条边插入 Radix Tree。当第二轮到来时，其输入包含第一轮的完整历史加上新的用户消息。Radix Tree 的前缀匹配会发现第一轮的部分完全匹配，于是复用其 KV Cache，只需对新的用户消息部分执行 Prefill。这直接减少了 Prefill 的计算量和延迟。

**节点分裂与合并。** 当不同的请求共享部分但不完全相同的前缀时，Radix Tree 会自动分裂节点。例如，两个不同的对话都以相同的系统提示开头但有不同的用户消息时，系统提示对应的边会被分裂为两部分，使得两个对话可以共享系统提示的 KV Cache。

**LRU 淘汰策略。** GPU 显存是有限的，不可能无限保留所有请求的 KV Cache。SGLang 实现了一个 LRU（Least Recently Used）淘汰策略，优先淘汰最近最少使用的叶子节点。通过先淘汰叶子节点，系统保留了公共祖先节点（它们可能被更多后续请求复用），直到这些祖先节点本身也变成叶子节点并被淘汰。

重要的是，SGLang 并没有为缓存预分配固定大小的内存池。缓存的 Token 和正在运行的请求共享同一个内存池，系统动态地在缓存和新请求之间分配内存。当有足够多的等待请求需要运行时，系统会淘汰所有缓存的 Token 以腾出空间给更大的批次。这确保了缓存不会以牺牲批处理效率为代价。

**缓存感知调度。** SGLang 定义缓存命中率为"已缓存的提示 Token 数量 / 总提示 Token 数量"。当等待队列中有多个请求时，执行顺序会显著影响缓存命中率。如果调度器在不相关的请求之间频繁切换，会导致缓存抖动（cache thrashing）和低命中率。SGLang 设计了一种缓存感知调度算法：在批处理场景中，按匹配前缀长度对请求排序，优先执行匹配前缀最长的请求（最长共享前缀优先策略）。论文证明，在离线场景中，这种策略等价于对 Radix Tree 的深度优先搜索（DFS）遍历顺序，可以达到最优缓存命中率。实测中，这种调度策略平均达到了最优命中率的 96%。

**前端提示机制。** 在 `fork` 原语的执行中，SGLang 前端会先将共享前缀作为"提示"发送给运行时，确保 Radix Tree 中正确地建立了前缀节点。然后再发送各分叉的后缀部分。这种"Frontend Hint"机制是前端-运行时协同设计的一个典型体现——前端的程序结构信息被用于指导运行时的缓存管理。

**分布式扩展。** RadixAttention 可以扩展到多 GPU 场景。在张量并行中，每个 GPU 维护一个分片的 KV Cache，但 Tree 操作是相同的，不需要额外同步。在数据并行中，SGLang 的缓存感知负载均衡器（Cache-Aware Load Balancer）在路由层维护一个近似的 Radix Tree，预测每个 Worker 的缓存命中率，并将请求路由到命中率最高的 Worker，从而在多实例间也能充分利用缓存。

---

## 10.4 核心优化技术

### 10.4.1 RadixAttention 与前缀缓存

RadixAttention 的缓存复用能力已在 10.3.3 节中详细介绍。这里从应用场景的角度，总结其在不同工作负载中的实际表现。

在 5-shot MMLU 基准测试中，所有试题共享相同的 5-shot 示例前缀。RadixAttention 只需在第一个试题中计算这部分前缀的 KV Cache，后续所有试题直接复用。这不仅提高了吞吐量（通过减少总内存占用来允许更大的批次），还降低了延迟（通过减少 Prefill 计算量来加快首 Token 生成）。在 20-shot HellaSwag 中，RadixAttention 实现了两层共享：Few-shot 示例的前缀被所有题目共享，而每道题的题干前缀又被其多个选项共享。

在多轮对话场景中，每一轮新对话都共享之前所有轮次的完整历史。对于短输出的多轮对话，RadixAttention 的加速效果非常显著，因为 Prefill 在总延迟中占主导地位；对于长输出的多轮对话，由于 Decode 时间占主导，加速比相对较小。

在 Tree-of-Thought 和 Self-Consistency 等需要对同一问题采样多个答案的场景中，RadixAttention 允许所有采样分支共享问题前缀的 KV Cache，与 `fork` 原语形成完美配合。

在 SGLang 部署在 LMSYS Chatbot Arena 的一个月生产实践中，观察到 LLaVA-Next-34B 的 RadixAttention 缓存命中率为 52.4%，Vicuna-33B 的缓存命中率为 74.1%。缓存命中主要来自常见的系统消息、频繁被重用的示例图片、以及多轮对话的历史记录。对于 Vicuna-33B，缓存命中使首 Token 延迟平均降低了 1.7 倍。

SGLang 还实现了层次化缓存（Hierarchical Cache）——当 GPU 显存不足以容纳所有缓存时，可以将不常用的 KV Cache 卸载到 CPU 内存中，在需要时再载回 GPU。这进一步扩展了缓存的有效容量，尤其在长上下文场景中价值显著。

### 10.4.2 Constrained Decoding：Jump-Forward 与 Grammar-Guided

在许多实际应用中，LLM 的输出需要符合特定格式：JSON Schema、SQL 语句、代码片段、或自定义的结构化模板。约束解码（Constrained Decoding）通过在生成过程中限制每一步允许的 Token 集合来保证输出格式正确性。

**传统方法的局限。** 传统的约束解码方法将约束（通常用正则表达式或上下文无关文法表示）转换为有限状态机（FSM）。在每个解码步骤中，系统维护 FSM 的当前状态，获取从当前状态可达的下一状态对应的合法 Token 集合，将不合法 Token 的概率设为零（masking），然后从合法 Token 中采样。这种方法逐 Token 工作，即使在确定性路径上（只有一个合法的下一个 Token）也需要执行完整的前向计算来"生成"这个 Token。

**压缩有限状态机。** SGLang 提出了压缩有限状态机（Compressed Finite State Machine）来解决这一效率问题。其核心思想是对 FSM 进行静态分析，将相邻的单一转移边（只有一个合法下一状态的边）压缩为一条边。这样，当解码过程进入一条压缩边时，系统知道接下来的多个 Token 是确定性的，可以在一个前向计算步骤中一次性解码多个 Token（Jump-Forward），而无需逐个生成。

例如，在 JSON 解码中，当模型刚生成完一个字段的值后，下一段内容可能是 `", "next_field": "` 这样的固定模板。在传统方法中，这个模板需要多个解码步骤，每步一个 Token。在压缩 FSM 中，这整段模板被识别为一条压缩边，可以在一步中跳过。

**XGrammar 集成。** 从 SGLang v0.4 开始，系统集成了 XGrammar 作为新的语法后端。XGrammar 是由 MLC 团队开发的高效结构化生成引擎，支持基于下推自动机（Pushdown Automaton, PDA）的批量约束解码。相比原始的压缩 FSM，XGrammar 在以下方面做了进一步优化：支持上下文无关文法（CFG），表达能力更强；实现了高效的 Token 掩码预计算，减少了每步的 CPU 开销；支持批量处理，不同请求可以有不同的语法约束而不影响批处理效率。根据基准测试，SGLang + XGrammar 在 JSON 解码任务上比其他开源解决方案快多达 10 倍。

在实际使用中，用户可以通过启动参数 `--grammar-backend xgrammar` 启用 XGrammar 后端，然后在请求中通过 `response_format` 参数指定 JSON Schema 或正则表达式约束。

### 10.4.3 Speculative Decoding（EAGLE 集成）

投机解码（Speculative Decoding）是加速自回归生成延迟的重要技术，其原理已在第 7 章详细介绍。SGLang 选择集成 EAGLE 系列算法（EAGLE、EAGLE-2、EAGLE-3）作为其投机解码的核心方案，这是因为 EAGLE 的特征层投机策略在接受率和草稿开销之间取得了优异的平衡。

**EAGLE 在 SGLang 中的集成方式。** 在 SGLang 中使用 EAGLE 投机解码时，需要指定一个已训练好的 EAGLE Draft 模型路径。启动时通过参数配置：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-70B-Instruct \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path path/to/eagle-draft-model \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8
```

其中 `--speculative-num-steps` 控制每次投机的步数（Draft 模型连续生成多少个候选 Token），`--speculative-eagle-topk` 控制每一步保留的候选 Token 数量，二者共同决定了候选 Token 树的大小。

**Token Tree Verification。** EAGLE 生成的不是一条线性的候选序列，而是一棵候选 Token 树。SGLang 的 ModelRunner 使用批量验证（batch verification）的方式一次性验证整棵树：将树中所有候选 Token 按照适当的位置编码和注意力掩码排列，然后用目标模型执行一次前向计算。验证结果确定树中哪条路径被接受，从而一次性推进多个 Token。

**EAGLE-3 的进步。** 2025 年发布的 EAGLE-3 在前代基础上做出了重要改进：放弃了特征预测（feature prediction），转而直接进行 Token 预测；用多层特征融合替代了单层顶层特征依赖。这些改进使得 EAGLE-3 在 SGLang 中实现了 2.7 到 3.5 倍的推理速度提升，并且在服务场景中可以将吞吐量翻倍。

**SpecForge 训练框架。** 为了简化 EAGLE Draft 模型的训练过程，SGLang 团队还发布了 SpecForge，一个专门为 SGLang 投机解码设计的训练框架，使得用户可以为自己的目标模型快速训练高质量的 Draft 模型。

### 10.4.4 PD 分离与多节点推理

Prefill-Decode 分离（PD Disaggregation）是大规模部署中的重要架构模式，其动机和原理已在第 8 章讨论。SGLang 实现了完整的 PD 分离支持，允许将 Prefill 和 Decode 分配到不同的 GPU 实例上执行。

**架构设计。** SGLang 的 PD 分离采用了前端路由 + 后端 Worker 的架构。一个前端路由器接收所有请求，将需要 Prefill 的请求路由到 Prefill Worker，Prefill 完成后，KV Cache 通过高速网络（如 NVLink、InfiniBand 或 RDMA）传输到 Decode Worker，Decode Worker 接管后续的逐 Token 生成。KV Cache 的传输支持逐层流水线化（Layerwise Transfer），即 Prefill Worker 每计算完一层的 KV Cache 就立即传输，而不是等待所有层全部完成后一次性传输，从而将传输延迟与计算时间重叠。

**灵活的配置方式。** SGLang 支持多种 PD 分离拓扑：节点内分离（同一服务器上的不同 GPU 分别承担 Prefill 和 Decode）、节点间分离（不同服务器分别承担两个角色）、以及更复杂的多 Prefill 对多 Decode 的配置。用户可以根据工作负载特征（输入长度分布、输出长度分布、并发量）灵活调整 Prefill Worker 和 Decode Worker 的数量比例。

**与 DeepSeek 模型的优化。** 在 DeepSeek 系列模型的大规模部署中，SGLang 将 PD 分离与专家并行（Expert Parallelism）相结合。在 Decode 端，SGLang 使用数据并行注意力（DP Attention）来减少 MLA 带来的 KV Cache 重复问题，同时使用专家并行来高效分布 MoE 层的计算。SGLang 团队在博客中展示了使用 PD 分离 + 大规模专家并行部署 DeepSeek-V3/R1 的实践，这是截至目前开源社区中最高效的 DeepSeek 部署方案之一。

### 10.4.5 Torch.compile 深度集成

Torch.compile 是 PyTorch 2.0 引入的即时编译器，能够将 Python 编写的模型代码通过 TorchDynamo 图捕获和 TorchInductor 代码生成自动编译为高效的 GPU 代码。SGLang 从 v0.3 版本开始深度集成了 Torch.compile，使其成为提升推理性能的重要手段。

**集成方式。** SGLang 的 Torch.compile 集成不是简单地对整个模型调用 `torch.compile()`，而是针对推理场景做了精细的适配。模型的前向计算被标注了适当的编译边界，确保 TorchDynamo 能够捕获完整的计算图而不会在不支持的操作处中断（graph break）。对于推理特有的操作（如 Paged Attention Kernel、RoPE 位置编码等），SGLang 通过 Custom Operator 注册的方式让 TorchDynamo 正确处理这些操作。

**性能收益。** 在 SGLang v0.3 的发布公告中，Torch.compile 带来了高达 1.5 倍的推理速度提升。加速主要来自三个方面：自动算子融合（例如将 RMSNorm、残差连接和 RoPE 融合为一个 Kernel）、减少 GPU Kernel Launch 开销、以及更高效的内存访问模式。

**与分段 CUDA Graph 的结合。** SGLang 将 Torch.compile 与分段 CUDA Graph（Piecewise CUDA Graph）相结合。CUDA Graph 将一系列 GPU 操作录制为一个图，后续可以一次性回放，消除了逐个 Kernel 启动的 CPU 开销。分段 CUDA Graph 将不同的 Token 数量区间对应不同的预编译图，覆盖了从小批次解码到大批次 Prefill 的各种场景。Torch.compile 生成的优化代码作为这些 CUDA Graph 的内容被录制，二者协同实现了极致的 Kernel 启动效率和计算效率。

**使用方式。** 用户在启动服务器时添加 `--enable-torch-compile` 即可启用。需要注意的是，首次启动时需要经历编译预热（compilation warmup），这可能需要几分钟的额外时间。编译完成后的性能提升是持久的，不影响后续请求的延迟。

---

## 10.5 Reasoning Model 支持

### 10.5.1 Thinking / Non-Thinking 模式

随着 DeepSeek-R1、QwQ-32B、Qwen3 等推理模型（Reasoning Model）的出现，推理引擎需要支持一种新的输出模式：模型先生成一段内部"思考"（thinking）内容，然后生成最终的答案。思考内容和最终答案之间通常由特定的标记（tag）分隔，例如 `<think>` 和 `</think>`。

SGLang 提供了对推理模型的原生支持，关键在于两个方面。

**Thinking / Non-Thinking 模式切换。** 对于支持混合模式的模型（如 DeepSeek-V3.1/V3.2、Qwen3），用户可以在请求中通过参数控制是否启用思考模式。对于 DeepSeek-V3 系列，使用 `thinking` 参数（布尔值）；对于 Qwen3 系列，使用 `enable_thinking` 参数。当禁用思考模式时，模型跳过思考阶段直接生成答案，适用于简单查询或对延迟敏感的场景。

**流式输出中的推理内容分离。** 在流式输出模式下，SGLang 能够实时区分思考内容和最终答案，将它们分别通过 `reasoning_content` 和 `content` 字段返回给客户端。这遵循了 DeepSeek API 建立的标准接口设计。用户可以选择将思考内容流式传输（实时显示思考过程），或缓冲到最后一个思考 chunk 一次性返回。

### 10.5.2 Reasoning Parser 机制

不同的推理模型使用不同的标记格式来分隔思考内容和答案。SGLang 设计了一个可扩展的 Reasoning Parser 框架来统一处理这些差异。

**内置 Parser。** SGLang 内置了多种 Parser：`deepseek-r1` 用于 DeepSeek-R1 系列（包括 R1、R1-0528 以及各种蒸馏变体），处理 `<think>` 和 `</think>` 标记；`deepseek-v3` 用于 DeepSeek-V3.1/V3.2 系列，支持 `thinking` 参数控制；`qwen3` 用于 Qwen3 系列模型，支持 `enable_thinking` 参数；此外还有 `kimi_k2`（用于 Kimi K2 Thinking 模型）和 `gpt-oss`（用于 OpenAI GPT OSS 模型）等。

**模型特定行为。** 不同模型的思考标记行为存在细微差异。例如，DeepSeek-R1 原版不生成 `<think>` 开始标记，直接进入思考内容；而 DeepSeek-R1-0528 则生成完整的 `<think>` 和 `</think>` 标记对。Parser 框架统一处理了这些差异，用户只需在启动服务器时指定 `--reasoning-parser` 参数即可。

**可扩展性。** 对于未来的新推理模型，开发者可以通过继承 `BaseReasoningFormatDetector` 类来实现新的 Parser，并注册到框架中。这种设计确保了 SGLang 能够快速适配新出现的推理模型。

**使用方式。** 在启动时通过 `--reasoning-parser deepseek-r1` 指定 Parser，然后通过 OpenAI-Compatible API 的 `separate_reasoning` 参数（默认启用）来控制是否分离推理内容。支持非流式和流式两种方式。

---

## 10.6 多模态与视觉语言模型支持

SGLang 从最初版本就将多模态支持作为核心特性之一。在前端，`image` 和 `video` 原语允许开发者将视觉输入自然地嵌入到提示词中。在运行时，SGLang 对多模态模型的推理做了系统性的优化。

**多模态模型的 KV Cache 复用。** RadixAttention 对多模态输入的支持方式是：计算输入图像或视频的哈希值，将其作为 Radix Tree 中的键。当多个请求包含相同的图像时（例如关于同一张图片的多个问题），它们的视觉 Token 对应的 KV Cache 可以被自动复用。在 SGLang 论文的 LLaVA-bench-in-the-wild 基准测试中，多个关于同一图片的问题共享了图像 Token 的 KV Cache，结合高效的推理运行时，SGLang 在多模态基准上实现了相比 Hugging Face Transformers 高达 6 倍的吞吐量提升。

**支持的模型范围。** SGLang 支持广泛的视觉语言模型（VLM），包括 LLaVA 系列、Qwen-VL 系列、Gemma 3 Vision、InternVL、NVILA 等。SGLang 团队还发布了详细的教程，指导如何将新的 VLM 模型集成到 SGLang 中。

**编码器优化。** 对于 VLM 中的视觉编码器（如 ViT），SGLang 支持编码器-解码器分离的部署模式（Encoder/Prefill/Decode 分离，简称 E/P/D 模式），允许将计算密集的视觉编码分配到专门的资源上，避免影响文本推理的吞吐量。此外，SGLang 还支持视觉编码器的数据并行（DP Encoder），将多个请求的图像编码并行化，减少 TTFT。

---

## 10.7 SGLang 与 KTransformers 的混合部署集成

SGLang 与 KTransformers 的集成代表了一个重要的发展方向：将高性能推理引擎的调度和服务能力与异构计算的硬件利用能力相结合。这一集成工作始于 2025 年下半年，目标是在 SGLang 的框架内支持 CPU/GPU 混合推理，特别是针对 MoE 模型的专家卸载场景。

**集成动机。** KTransformers 的核心优势在于其高效的 CPU 端 MoE 专家计算（基于 AMX 指令集）和精心设计的 CPU-GPU 流水线调度。但作为一个独立系统，KTransformers 在服务层面的功能（如连续批处理、缓存管理、负载均衡、API 兼容性）相对有限。SGLang 则在服务层面非常成熟，但默认假设所有计算在 GPU 上完成。将二者集成，意味着 SGLang 可以利用 KTransformers 的 CPU Kernel 来处理 MoE 专家计算，同时保留自身在调度、缓存和服务方面的全部优化。

**技术方案。** KTransformers 以"库后端"（library backend）的形式被集成到 SGLang 中。具体来说，SGLang 的模型层中，Dense 部分（Attention、LayerNorm 等）仍然在 GPU 上执行，而 MoE 层中的专家计算被分派到 CPU 和 GPU 上混合执行。SGLang 通过配置参数控制有多少专家放在 GPU 上、多少放在 CPU 上。GPU 上的专家使用标准的 CUDA Kernel，CPU 上的专家使用 KTransformers 优化的 AMX Kernel。

**性能表现。** 根据 SGLang 团队在 GitHub Issue 中公布的初步基准数据，在搭载双路 Intel Xeon Platinum 8452Y CPU（36 核 × 2，1 TB DDR5）和单张 NVIDIA A100（40 GB）的服务器上，对 DeepSeek-V3-0324 模型的推理测试显示，集成后的系统在 Prefill 阶段相比 Llama.cpp 和 Fiddler 取得了一致性的领先，CPU 端 MoE Kernel 达到了 21.3 TFLOPS，是 PyTorch 基线的 3.98 倍。在 Decode 阶段，相比 Fiddler 取得了 2.42 到 4.09 倍的加速。

**发展路线。** 截至本书撰写时，这一集成仍在活跃开发中。路线图包括：支持更多权重格式（GPTQ、AWQ）、热度感知的专家分布策略、专家延迟（Expert Deferral）机制、以及投机解码支持。长期来看，SGLang + KTransformers 的组合有望使普通消费级硬件也能高效运行超大规模 MoE 模型。

---

## 10.8 Mini-SGLang：5000 行代码的推理引擎教学实现

Mini-SGLang 是 SGLang 团队于 2025 年 12 月发布的教学项目。它将 SGLang 从近 30 万行 Python 代码精炼为约 5000 行，保留了核心设计和关键优化，同时大幅降低了理解门槛。

**设计目标。** Mini-SGLang 有两个主要目标。第一个是教育目的：为初学者提供一个清晰、高度模块化的代码库，使其能够理解现代 LLM 推理引擎的核心组件。第二个是研究原型：为 ML 和系统研究者提供一个即用的高性能框架，可以在不深入复杂生产代码的情况下快速验证新的优化想法。

**保留的核心特性。** 尽管代码量缩减了 60 倍，Mini-SGLang 仍然实现了以下关键特性：

Radix Attention——完整的 Radix Tree 数据结构和 KV Cache 复用逻辑，包括 LRU 淘汰策略。开发者可以在短短几百行代码中看到 Radix Tree 的节点分裂、前缀匹配和缓存回收的全部流程。

Chunked Prefill——将长的 Prefill 请求分块处理，控制显存峰值并减少对 Decode 请求的干扰。

Overlap Scheduling——与 SGLang 完全相同的零开销调度器设计。CPU 调度与 GPU 计算重叠，NVIDIA Nsight Systems 的 profiling 确认 GPU 没有空闲间隙。

Tensor Parallelism——支持多 GPU 张量并行推理，使用 NCCL 进行通信。

高性能 Kernel——集成 FlashAttention-3（用于 Prefill）和 FlashInfer（用于 Decode）的 Attention Kernel。

OpenAI-Compatible API——提供标准的 HTTP API 接口，包括在线基准测试工具。

**性能对比。** Mini-SGLang 的性能接近甚至匹配完整版 SGLang。在使用 Qwen3-32B 模型、4 路张量并行部署在 4 张 H200 GPU 上的在线服务基准测试中，Mini-SGLang 在吞吐量、P90 TTFT 和 TBT 三个指标上均与 SGLang 几乎一致。在离线吞吐量测试中，Mini-SGLang 相比 Nano-vLLM 基线也展现了一致的优势。

**代码结构。** Mini-SGLang 的代码组织与 SGLang 保持相同的高层架构：前端 API Server、Tokenizer Server、以及每个 GPU 对应一个后端 Scheduler。这种"缩略版"的代码结构使得读者可以先在 Mini-SGLang 中建立全局理解，再到完整的 SGLang 中深入某个特定模块。

**交互式 Shell 模式。** Mini-SGLang 还提供了一个简单的命令行交互模式，用户可以直接在终端与模型对话，方便测试和调试。

---

## 10.9 SGLang 源码导读与动手实验

本节为读者提供深入 SGLang 源码的入口指引和一组动手实验，旨在将前述的概念性知识转化为具体的代码理解。

**源码组织结构。** SGLang 的源码仓库（`github.com/sgl-project/sglang`）主要由以下几个目录组成。`python/sglang/srt/` 是后端运行时的核心代码，其中 `managers/scheduler.py` 包含调度器的全部逻辑（请求队列管理、Radix Tree 操作、批次组装、重叠调度）；`model_executor/` 包含 ModelRunner 的实现（CUDA Graph 管理、前向计算调度）；`layers/` 包含各种算子层的实现（Attention、MoE、Sampling 等）；`mem_cache/` 包含 Radix Tree 和内存池的实现。`python/sglang/lang/` 是前端 DSL 的实现，包括解释器和编译器。`python/sglang/srt/parser/` 包含 Reasoning Parser 的实现。

**推荐的阅读路径。** 对于初学者，建议从 Mini-SGLang 入手，按照以下路径阅读：首先理解 Scheduler 的主循环（`event_loop_normal`），看清每个迭代步中调度器如何选择请求、匹配缓存、组装批次；然后深入 ModelRunner 的 `forward` 方法，理解 CUDA Graph 是如何被选择和回放的；接着阅读 Radix Tree 的实现（`radix_cache.py`），理解节点插入、前缀匹配和 LRU 淘汰的细节；最后回到完整的 SGLang 代码，对比每个模块的完整实现与 Mini-SGLang 中的精简版。

**动手实验一：RadixAttention 缓存效果验证。** 启动一个 SGLang 服务器（使用较小的模型如 Llama-3.2-3B-Instruct），向其发送一组共享前缀的请求（例如使用相同系统提示的多轮对话），通过 `/get_cache_report` 端点或日志观察缓存命中率。然后对比关闭 Radix Cache（`--disable-radix-cache`）后的吞吐量和 TTFT 差异。

**动手实验二：零开销调度器验证。** 使用 NVIDIA Nsight Systems 对 SGLang 进行 profiling，对比启用和禁用（`--disable-overlap`）重叠调度时的 GPU 利用率和 Kernel 间隙。具体命令：

```bash
# 启用重叠调度（默认）
nsys profile python -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct
# 禁用重叠调度
nsys profile python -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct --disable-overlap
```

对比两次 profiling 的时间线，观察 GPU 是否存在空闲间隙。

**动手实验三：结构化输出性能测试。** 构造一个 JSON 解码基准测试，要求模型输出符合特定 JSON Schema 的内容。分别使用默认语法后端和 XGrammar 后端（`--grammar-backend xgrammar`），对比解码速度。

**动手实验四：投机解码加速测试。** 如果有 EAGLE Draft 模型可用，在 SGLang 中启用 EAGLE 投机解码，对比启用和禁用投机解码时的单请求延迟和服务吞吐量。观察不同 `--speculative-num-steps` 设置对接受率和整体速度的影响。

**动手实验五：Mini-SGLang 扩展实验。** Fork Mini-SGLang 仓库，尝试在其中实现一个简单的新功能，例如：在 Radix Tree 中添加缓存命中率统计、实现一种新的调度策略（如 SJF），或集成一种新的采样方法。Mini-SGLang 的模块化设计使得这类修改可以在数小时内完成。

---

## 本章小结

SGLang 代表了一种"语言-运行时协同设计"的推理系统范式。其前端 DSL 通过 `gen`、`select`、`fork`、`join` 等原语，使开发者能够以程序化的方式编排复杂的 LM 程序，同时向后端暴露出丰富的优化信息。后端运行时以 RadixAttention 为核心，通过 Radix Tree 实现了跨调用、跨请求的 KV Cache 自动复用，配合零开销批调度器、压缩有限状态机约束解码、EAGLE 投机解码、PD 分离、Torch.compile 深度集成等一系列优化技术，在广泛的工作负载上实现了领先的性能。

SGLang 的快速演进也体现了推理系统领域的发展趋势：从关注单一请求的计算效率，扩展到关注多请求、多调用间的系统级优化；从纯 GPU 推理，扩展到与 KTransformers 的 CPU-GPU 异构集成；从文本生成，扩展到多模态和推理模型的原生支持。Mini-SGLang 作为一个 5000 行代码的精炼实现，则为理解和研究现代推理引擎提供了宝贵的入口。

在后续章节中，我们将在 KTransformers（第 11 章）和 KLLM（第 12 章）的讨论中继续看到 SGLang 的身影——它作为服务层与其他系统协同工作的能力，正是其开放架构设计的体现。
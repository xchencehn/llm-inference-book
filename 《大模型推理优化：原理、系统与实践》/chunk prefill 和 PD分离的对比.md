


# Chunked Prefill 与 P/D 分离（Prefill-Decode Disaggregation）详细对比

## 一、前置知识：LLM 推理的两个阶段

在深入对比之前，需要先理解 LLM 推理天然分为两个阶段：

**Prefill（预填充）阶段**：处理用户输入的全部 prompt token，并行计算所有 token 的 Key/Value 向量存入 KV Cache，最终生成第一个输出 token。该阶段是 **计算密集型（compute-bound）**，能充分利用 GPU 算力。

**Decode（解码）阶段**：利用已缓存的 KV Cache，逐 token 自回归生成后续输出。每一步只处理一个新 token，该阶段是 **访存密集型（memory-bound）**，GPU 算力严重空闲。

两个阶段的资源需求截然不同，这正是这两种优化技术的出发点。

---

## 二、Chunked Prefill（分块预填充）

### 2.1 核心思想

Chunked Prefill 源自 Microsoft 的 **Sarathi** 论文（2023），其核心思想是：**不再一次性完成一个请求的全部 Prefill，而是将长 prompt 切成固定大小的 chunk（如 512~8192 tokens），每次只处理一个 chunk，并与正在 Decode 的请求混合在同一个 batch 中执行**。

这本质上是一种 **“PD 聚合”（PD Aggregation）** 策略——Prefill 和 Decode 仍然运行在**同一个 GPU 实例**上，但通过分块调度来缓解相互干扰。

### 2.2 解决的问题

在 continuous batching 的 “Prefill-first” 策略下，当一个新请求到达时，系统会立即执行其完整 Prefill。由于 Prefill 高度占用 GPU 算力，在 Prefill 执行期间，所有正在 Decode 的请求只能完成一个 decode step，用户会感受到**明显的输出"卡顿"（decode stall）**，尾部 ITL（inter-token latency）飙升。

Chunked Prefill 通过把大的 Prefill 切成多个小块，每个小块和 Decode 请求并行执行，使得 Decode 请求在每个 chunk 间隔都能产出一个 token，从"完全停顿"变成"略有减速"。

### 2.3 关键特性

**Chunk Size 是核心调优参数**：chunk 越小，对 Decode 干扰越小（TPOT 越好），但 Prefill 本身会变慢（TTFT 变差），且 GPU 利用率不一定最优。chunk 越大，TTFT 越好但 TPOT 越差。典型范围 512~8192 tokens，vLLM V1 中默认开启。

**提升吞吐量**：由于 Prefill 是 compute-bound 而 Decode 是 memory-bound，将两者混合在同一 batch 中恰好互补，可以同时充分利用 GPU 的计算能力和显存带宽。TNG 公司实测在标准 vLLM 部署中，chunked prefill 带来了约 **+50% 的总 token 吞吐量提升**。

**无需额外硬件或网络**：只是调度层面的优化，仍然是单实例部署，不涉及 KV Cache 传输，部署运维简单。

### 2.4 局限性

Prefill 和 Decode 仍然共享同一 GPU，**无法完全消除阶段间干扰**。chunk 越小干扰越小，但实践中很难找到最优 chunk size，尤其在请求长度变化大的环境中。TaiChi 论文的实验表明，在 TPOT 约束较紧时，chunked prefill 的 SLO 达标率仅 16%（平衡 SLO 场景下）。

---

## 三、P/D 分离（Prefill-Decode Disaggregation）

### 3.1 核心思想

P/D 分离的思路更加激进：**将 Prefill 和 Decode 物理分离到不同的 GPU 实例（甚至不同的机器）上运行**。Prefill 实例专门处理 prompt 计算并生成 KV Cache，然后通过高速网络（NVLink、InfiniBand、RDMA 等）将 KV Cache 传输到 Decode 实例，由 Decode 实例独立完成后续 token 生成。

该思路由 **DistServe**（UCSD Hao Zhang 团队，OSDI 2024）和 **Splitwise**（Microsoft，2024）两篇论文独立且几乎同时提出。

### 3.2 解决的问题

P/D 分离从根本上**消除了 Prefill 和 Decode 之间的干扰**。Decode 实例不再受到任何 Prefill 计算的打扰，可以保持稳定且极低的 ITL/TPOT，解决了尾部延迟抖动问题。同时，两个阶段可以独立扩缩容、独立调参（如使用不同的 tensor parallelism 策略）。

### 3.3 关键特性

**彻底消除干扰**：Decode 实例 100% 的 GPU 时间都用于 Decode，TPOT 可控且稳定。DistServe 论文的实验表明在严格 TPOT SLO 下 goodput 最多提升 4.48×。

**独立资源配置与扩缩容**：可以为 Prefill 配备更多计算资源（如更大的 TP），为 Decode 配备更多实例或更大显存，根据负载动态调整 Prefill:Decode 实例比例。

**适合大规模生产部署**：Moonshot AI 的 Mooncake 系统在其 Kimi 产品的大规模生产环境中采用了 P/D 分离架构，设计了以 KV Cache 为中心的调度系统；LMSYS/SGLang 团队部署 DeepSeek-V3/R1 时也采用了 P/D 分离 + 大规模 Expert Parallelism 的方案。

### 3.4 局限性

**不提升总吞吐量**：vLLM 文档明确指出 “Disaggregated prefill DOES NOT improve throughput”。由于 Decode 实例独占 GPU 但计算利用率低（memory-bound），整体 GPU 利用率反而可能低于 chunked prefill。

**KV Cache 传输开销**：需要高速网络支持。如果网络带宽不足或延迟过高，KV Cache 传输本身会成为瓶颈，性能甚至会下降（BentoML 实测在配置不当时性能可下降 20-30%）。好消息是 GQA、MLA 等技术大幅减少了 KV Cache 大小，且 NIXL、NVLink 等使传输开销趋于可忽略。

**TTFT 可能变差**：由于只有部分实例处理 Prefill，在高并发下 Prefill 队列排队时间增加，导致 TTFT 上升。TaiChi 论文实验表明，在严格 TTFT 约束下，P/D 分离的 SLO 达标率仅 42%。

**部署复杂度高**：需要管理多种角色的实例、KV Cache 传输通道、路由调度，运维成本显著增加。

---

## 四、系统化对比

|维度|Chunked Prefill（PD 聚合）|P/D 分离（PD 解聚）|
|---|---|---|
|**架构**|单实例，Prefill 和 Decode 共享 GPU|多实例，Prefill 和 Decode 运行在不同 GPU|
|**干扰消除**|缓解但不消除，仍有 decode 延迟波动|彻底消除，Decode 实例不受 Prefill 影响|
|**TTFT 表现**|优秀（所有实例都能处理 Prefill）|较差（仅部分实例处理 Prefill，高并发下排队）|
|**TPOT/ITL 表现**|有改善但仍受 chunk 干扰，尾部 ITL 不稳定|极好且稳定（独占 Decode GPU）|
|**总吞吐量**|提升显著（compute+memory bound 互补）|不提升甚至略降（GPU 利用率不如聚合模式）|
|**GPU 利用率**|高（混合 batch 互补利用）|较低（Decode 实例 memory-bound 导致算力空闲）|
|**调优难度**|chunk size 难以选择最优值|P:D 实例比例需要根据负载精细调整|
|**部署复杂度**|低（单实例，无需 KV 传输）|高（多角色实例 + 高速网络 + 路由调度）|
|**网络需求**|无额外要求|需要高速互联（NVLink/RDMA/InfiniBand）|
|**最佳场景**|紧 TTFT + 宽 TPOT SLO，或追求最大吞吐|紧 TPOT + 宽 TTFT SLO，或追求稳定低延迟|
|**代表论文**|Sarathi (MSR, 2023), Sarathi-Serve (OSDI 2024)|DistServe (OSDI 2024), Splitwise (MSR, 2024), Mooncake (2024)|

---

## 五、业界实际使用情况

### 5.1 两种技术的关系：不是替代而是互补

TaiChi 论文（华为云，2025）给出了最清晰的总结：**Chunked Prefill 在紧 TTFT + 宽 TPOT 场景下最优；P/D 分离在紧 TPOT + 宽 TTFT 场景下最优；在平衡 SLO 场景下两者都不理想**。因此业界逐渐走向**混合方案**——在 P/D 分离架构的基础上，每个实例内部仍然使用 chunked prefill 来提升利用率。

### 5.2 典型生产实践

**Moonshot AI (Kimi)**：全面采用 P/D 分离架构（Mooncake 系统），以 KV Cache 为中心进行全局调度，利用 CPU/DRAM/SSD 作为多级 KV Cache 池。

**LMSYS / SGLang 团队**：在部署 DeepSeek-V3/R1 时采用 P/D 分离 + 大规模 Expert Parallelism，达到 52.3K input tokens/s、22.3K output tokens/s 的速度。

**TNG Technology Consulting**：在 24 块 H100 GPU 的集群上全面启用 chunked prefill，在所有 vLLM 部署中获得 +50% 吞吐提升，作为默认配置。

**Together AI**：提出 Cache-aware Prefill-Decode disaggregation (CPD)，在 P/D 分离基础上增加专门的 pre-prefill 层处理缓存未命中的请求，实测提升 40%。

---

## 六、框架支持情况

### 6.1 Chunked Prefill 支持

|框架|状态|
|---|---|
|**vLLM**|V1 中**默认开启**，无法关闭。chunk size 可通过 `--max-num-batched-tokens` 配置。|
|**SGLang**|支持，通过 `--enable-mixed-chunk` 启用 Sarathi-Serve 风格的 chunked prefill 调度。|
|**TensorRT-LLM**|支持，称为 “Chunked Context”，通过 in-flight batching 实现。|
|**Ray Serve (vLLM backend)**|继承 vLLM 的 chunked prefill 能力。|

### 6.2 P/D 分离支持

|框架|状态|
|---|---|
|**vLLM**|实验性支持。通过 `--kv-transfer-config` 配置。支持 6+ 种 Connector：NixlConnector、LMCacheConnector、P2pNcclConnector、MooncakeConnector、ExampleConnector、MultiConnector、OffloadingConnector、FlexKVConnector 等。|
|**SGLang**|支持，已有完整 PD Disaggregation 路线图（GitHub Issue #4655），使用 Mooncake 进行 KV Cache 传输，已在 AMD GPU 和 NVIDIA GPU 上验证。还扩展到 EPD（Encoder-Prefill-Decode 三阶段分离），用于多模态模型。|
|**TensorRT-LLM**|支持，有专门的 Disaggregated Serving 模块，支持 NIXL、服务发现、动态扩缩容、请求取消等生产级功能。还探索了 SM 级别的分离（intra-GPU disaggregation）。|
|**NVIDIA Dynamo**|这是 NVIDIA 专门为 P/D 分离等分布式推理场景设计的开源框架，原生支持 disaggregated serving，提供 Planner 自动规划 P:D 实例比例，与 vLLM/TensorRT-LLM 作为后端引擎配合。|
|**llm-d**|Red Hat/IBM/Google 联合推动的 **Kubernetes 原生** LLM 推理框架，内置 P/D 分离支持，通过 sidecar 编排 Prefill/Decode 实例，支持 NIXL 和 TCP/RDMA 传输，与 KServe 深度集成。|
|**Ray Serve**|原生支持 P/D 分离，通过独立的 Prefill Deployment 和 Decode Deployment 实现，底层可选 vLLM 引擎。|
|**Mooncake（月之暗面）**|生产级 KV Cache 中心化分离架构，已开源 Mooncake Transfer Engine 并集成到 SGLang 和 vLLM 中作为 Connector。|

---

## 七、前沿趋势

**混合模式成为主流**：TaiChi（华为云）提出统一 PD 聚合与 PD 解聚的混合架构，通过 P-heavy 和 D-heavy 两类实例搭配调度，在平衡 SLO 下 goodput 提升 77%。Nexus 论文则探索 intra-GPU 级别的 PD 分离——在同一块 GPU 内部将 SM 划分为 Prefill 组和 Decode 组。

**KV Cache 传输技术快速演进**：NVIDIA 开源的 NIXL（Inference Xfer Library）成为 KV 传输的标准层，支持 UCX、GDS 等多种后端，多个框架已采用。CXL、NVMe-oF 等新互联技术也在被探索。

**Cache-aware 调度**：Together AI 的 CPD、Mooncake 的 Conductor 等系统开始将 Prefix Cache 命中率纳入 P/D 调度决策——如果 Decode 节点本地已有 KV Cache 缓存，就不需要远程 Prefill，直接本地完成。

**与 MoE 模型的结合**：DeepSeek 系列的 MoE 架构下，P/D 分离与 Expert Parallelism 的结合成为部署大模型的标准范式。SGLang 在部署 DeepSeek-V3 时采用 PD 分离 + 8-way EP，LMSYS 博客有详细的实践分享。

---

## 八、总结建议

如果你的场景是**单卡或少量 GPU、追求最大吞吐、请求延迟要求不苛刻**，Chunked Prefill 是首选——它零成本启用、部署简单、效果显著，vLLM V1 已经默认开启。

如果你的场景是**大规模多 GPU 集群、对 decode 延迟（TPOT/ITL）有严格 SLA 要求、有高速互联网络**，P/D 分离是更好的选择——它能彻底消除阶段干扰并独立扩缩容，但需要更复杂的基础设施。

在实际的大规模生产环境中，**两者通常结合使用**：在 P/D 分离的架构中，每个 Prefill-heavy 或 Decode-heavy 实例内部仍然可以启用 chunked prefill 来最大化单实例利用率。





你这个问题特别好，直击了 P/D 分离在**同构集群**下"到底图什么"的本质。很多人确实有这个困惑——都是一样的 H100，把 Prefill 和 Decode 拆到不同卡上，同一张卡的算力和显存又没变，收益从何而来？

让我把真正的关键优化点拆开讲。

---

## 核心矛盾不是"硬件不匹配"，而是"两种负载无法同时做到最优"

异构硬件（P 用高算力卡、D 用高带宽卡）确实是 P/D 分离的一个远期愿景，Splitwise 论文就提过用 H100 做 Prefill、A100 做 Decode 这种搭配。但现实中几乎所有大规模部署都是同构集群，P/D 分离照样大幅有效。原因在于：**P/D 分离的核心收益不在于硬件异构，而在于"调度解耦"和"干扰消除"**。具体有以下几个层面：

---

### 1. 消除干扰 → 尾部延迟从"不可控"变成"可控"

这是同构集群下 P/D 分离最直接、最有价值的收益。

在聚合模式下（即使开了 chunked prefill），Prefill chunk 仍然会挤占 Decode 的 GPU 时间。一个 1024 token 的 chunk 可能让当次 batch 的 decode step 耗时从 40ms 涨到 80-100ms，用户体验到的就是 token 流忽快忽慢。更糟的是 **P99 尾部延迟极难控制**——某个瞬间来了一批长 prompt 请求，所有正在 decode 的用户都会被拖慢，你事后看 TPOT 分布会发现一条很长的尾巴。

P/D 分离后，Decode 实例上 **永远不会跑任何一个 Prefill token**。每个 decode step 的耗时几乎恒定，TPOT 方差极小。对于在线聊天、代码补全这类用户体验敏感的场景，这种"延迟可预测性"的价值远超纸面吞吐数字。

DistServe 的作者 Hao Zhang 在回顾博文中也说得很清楚：2025 年 P/D 分离突然在业界爆发，核心原因就是越来越多企业把 LLM 做进了产品——**一旦是面向用户的服务，延迟 SLO 就不是"越快越好"，而是"必须达标"**。goodput（满足 SLO 的吞吐）才是真正的优化目标，而 P/D 分离在严格 TPOT SLO 下 goodput 提升 2.0x~4.48x。

---

### 2. 独立并行策略 → 同样的卡可以用不同的"姿势"

这是很多人忽略的一个点。同样一堆 H100：

**Prefill 阶段**：是 compute-bound，一个请求可能有几千个 token 需要并行计算。此时用**更大的 TP（tensor parallelism）** 可以加速单次 Prefill 的速度，减少 TTFT。比如 Prefill 用 TP=4 在 4 张卡上拆分一个大矩阵乘。

**Decode 阶段**：是 memory-bound，单步计算量很小，瓶颈在显存带宽和 KV Cache 容量。此时 TP 增大反而会引入通信开销白白浪费时间。Decode 更适合**较小的 TP + 更多的 DP（data parallelism）** ——每个小组独立服务一批请求，用更多实例并行处理更多请求来摊薄 memory-bound 的开销。

在聚合模式下，一个实例的 TP/PP 配置是固定的，你不可能让同一组卡在 Prefill 时用 TP=4、Decode 时用 TP=2。但分离之后，两组卡可以各自用最优的并行策略。

**DeepSeek-V3 的实际部署就是典型案例**：Prefill 用较小的 EP + 较大的 DP，Decode 则用非常宽的 EP（≈256）+ 高 DP（≈8-16），两个阶段的并行策略完全不同。这在聚合部署下是做不到的。

---

### 3. 独立扩缩容 → 突破固定比例的桎梏

在聚合模式下，一个实例既做 Prefill 又做 Decode，系统的 Prefill 能力和 Decode 能力是**固定耦合**的。但实际负载中这两者的比例是剧烈变化的：

短 prompt + 长输出（如代码生成）→ Decode 压力大，Prefill 很轻  
长 prompt + 短输出（如文档摘要）→ Prefill 压力大，Decode 很轻  
突发流量 → Prefill 队列瞬间爆满

P/D 分离后，你可以动态调整 P:D 实例比例。DistServe 论文展示过，对于 ISL=512、OSL=64 的负载，最优比例是 2:1；而对于 ISL=2048、OSL=512 的负载，比例就完全不同。NVIDIA Dynamo 的 Planner 组件就是专门做这个事——持续监控负载特征，自动调整 P:D 比例。llm-d 在 Kubernetes 上也支持按角色独立 HPA。

在聚合模式下你只能"加卡"或"减卡"，每个卡承担的 P 和 D 负载比例是调度器隐式决定的，你很难精细控制。

---

### 4. 队列隔离 → 排队理论上的结构性优势

DistServe 论文用排队论做了一个很漂亮的分析。聚合模式下，Prefill 和 Decode 共享同一个服务队列，这是一个复杂的多类型请求混合排队系统，延迟很难建模和预测。

分离后，Prefill 相当于一个接近 **M/D/1 队列**（到达随机，服务时间相对确定），Decode 也是相对独立的队列。两个队列互不阻塞：Prefill 排队不会影响正在 Decode 的请求，Decode 队列满了也不会让新请求的 Prefill 排不上。这种结构性隔离让整个系统的延迟变得**可预测、可建模、可调优**。

---

### 5. 解锁更多系统级优化的可能性

一旦 Prefill 和 Decode 是独立服务，就打开了一系列新的优化空间：

**Prefill 实例可以做 Prefix Cache 共享**：多个请求的相同 system prompt 只需 Prefill 一次，KV Cache 缓存在 Prefill 节点供复用。Mooncake、LMCache 都在做这件事。

**Decode 实例可以做 KV Cache 卸载**：把不活跃的 KV Cache offload 到 CPU/SSD，腾出 GPU 显存服务更多并发请求。

**KV Cache 的传输和存储独立优化**：DeepSeek 建了 3FS（分布式文件系统），Mooncake 把 CPU/DRAM/SSD 都池化成 KV Cache 存储层，这些在聚合模式下都没有必要也没有空间做。

**为 RL 推理（如 RLHF/GRPO）提供更灵活的架构**：训练和推理可以共享 Decode 实例，Prefill 实例按需弹性伸缩。

---

### 6. 所以，代价是什么？

说了这么多好处，同构集群下 P/D 分离的代价也很明确：

**模型权重需要双份加载**：Prefill 实例和 Decode 实例各需要一份完整的模型权重，显存开销增大。这也是为什么聚合模式的吞吐量上限往往更高——它不用重复存模型。

**KV Cache 传输仍是开销**：即使在同一节点内用 NVLink（900GB/s on GB200），对于长 context 的请求 KV Cache 传输仍然需要时间。好在 GQA/MLA 大幅压缩了 KV 尺寸。

**集群利用率可能降低**：Decode GPU 的计算单元大部分时间是空闲的。这就是你说的核心问题——用一样的卡做 Decode 确实"浪费算力"。

---

## 总结：同构集群下 P/D 分离的关键收益排序

|优先级|收益|本质|
|---|---|---|
|**第一**|延迟可预测性 & 尾部 ITL 可控|干扰消除|
|**第二**|独立并行策略优化|不同阶段用不同 TP/EP/DP|
|**第三**|弹性扩缩容|P:D 比例动态调整|
|**第四**|解锁系统级优化|KV Cache 池化、Prefix Cache 共享等|
|**第五**|排队隔离 & 可建模性|排队论结构性优势|

所以结论是：**P/D 分离在同构集群下的核心价值不是"给 P 和 D 配不同硬件"，而是"让 P 和 D 不再互相拖后腿"。** 它本质上是用"模型权重多存一份 + KV Cache 传输一次"的代价，换来延迟可控性、调度灵活性和系统可组合性的巨大提升。对于以 goodput（满足延迟 SLO 的有效吞吐）为核心指标的在线服务，这笔交易非常划算。

至于"同构集群下 Decode GPU 算力浪费"的问题，业界也在积极探索解法——比如 Nexus 论文提出的 intra-GPU 分离（在同一张卡的 SM 层面划分 P 和 D 区域），以及 NVIDIA 新的 Rubin CPX 架构在硬件层面拥抱 P/D 分离的设计理念，都在试图从根本上解决这个效率问题。
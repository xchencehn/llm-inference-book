

# 并行策略与 Prefill/Decode 的关系

这是一个非常值得深入理解的问题。不同的并行策略在 Prefill 和 Decode 两个阶段表现出截然不同的效果，理解这一点对系统设计至关重要。

---

## 一、Tensor Parallelism（TP）—— 切权重矩阵

TP 的本质是把每一层的权重矩阵按列或按行切分到多个 GPU 上，每个 GPU 算一部分矩阵乘法，最后通过 AllReduce 通信汇总结果。

**在 Prefill 阶段**，你要算的是形如 (S,d)×(d,d)(S, d) \times (d, d)(S,d)×(d,d) 的大 GEMM。TP 把右边的权重矩阵按列切成 N 份，每个 GPU 算 (S,d)×(d,d/N)(S, d) \times (d, d/N)(S,d)×(d,d/N)，计算量大，Tensor Core 利用率高。AllReduce 的通信量相对于计算量来说占比很小，所以 TP 在 Prefill 阶段效率很好。

**在 Decode 阶段**，你要算的是 (1,d)×(d,d)(1, d) \times (d, d)(1,d)×(d,d)，本质上是向量乘矩阵。切成 N 份之后每个 GPU 算 (1,d)×(d,d/N)(1, d) \times (d, d/N)(1,d)×(d,d/N)，计算量本身就很小，但 AllReduce 的通信量并没有按比例缩小——每一步 decode 都要做一次 AllReduce。这导致通信占比变得很大，TP 的效率在 decode 阶段明显下降。

但 TP 在 decode 阶段仍然必要，原因有两个：第一，KV Cache 被分摊到多个 GPU 的显存上；第二，即便效率不高，把权重分摊后每个 GPU 需要从显存读取的权重量减少了，而 decode 正好是 memory-bound 的，所以减少读取量可以直接降低延迟。

**总结**：TP 在 Prefill 阶段是为了加速计算，在 Decode 阶段主要是为了分摊显存压力和减少访存量。

---

## 二、Pipeline Parallelism（PP）—— 切层

PP 的本质是把模型的 L 层分成若干段，每段放在不同的 GPU 上。数据像流水线一样从第一段流向最后一段。

**在 Prefill 阶段**，一个请求的 prompt 从 stage 0 进入，经过 stage 0 的那些层处理后，把激活（activation）发送给 stage 1，依此类推。如果只有一个请求，那在 stage 0 工作的时候 stage 1/2/3 都在空闲——这就是流水线气泡（pipeline bubble）。但如果有多个请求同时到达，就可以让 stage 0 处理完请求 A 后立刻处理请求 B，同时 stage 1 开始处理请求 A，从而填满流水线。Prefill 的计算量大，每个 stage 处理一个 microbatch 的时间较长，相对而言气泡占比可以接受。

**在 Decode 阶段**，问题变得严峻。每一个 decode step 每个 stage 的计算量非常小（只处理 1 个 token），但一个 token 必须串行经过所有 stage 才能得到输出。这意味着每个 decode step 的延迟是所有 stage 延迟之和加上 stage 之间的通信延迟。如果 PP 的 stage 数为 P，那么单个 token 的生成延迟大约放大 P 倍（相比不用 PP），再加上 P-1 次 stage 间通信。

更麻烦的是，decode 阶段多个请求虽然可以 batch 在一起，但它们共享同一个 pipeline，气泡问题依然存在。这就是为什么实际推理系统中 PP 的 stage 数通常不会设太大，一般 2-4 个 stage。

**总结**：PP 在 Prefill 阶段配合多请求可以较好地填满流水线；在 Decode 阶段会直接增大每个 token 的生成延迟，需要谨慎使用。

---

## 三、Sequence Parallelism（SP）—— 切序列

这里需要区分两种含义的 SP：

### 3a. Megatron 风格的 SP

这是 TP 的互补方案。在 TP 中，矩阵乘法部分被切分了，但 LayerNorm、Dropout 等操作是在完整的隐藏维度上做的，没有被并行化。Megatron SP 的做法是：在这些非 TP 操作处，把激活沿着序列维度 S 切分，每个 GPU 只处理 S/N 个 token 的 LayerNorm 等操作。这主要是为了节省激活内存，对 Prefill 阶段的长序列尤为重要，因为激活的大小与 S 成正比。在 Decode 阶段 S=1，这种 SP 基本不起作用。

### 3b. Context Parallelism / Ring Attention 风格的 SP

这种方案是真正把序列维度切开，每个 GPU 负责序列的一段。这对 Prefill 阶段处理超长序列非常关键。

假设序列长度 S=128K，4 个 GPU 各负责 32K 的子序列。问题在于 Attention 计算中，每个 token 的 Query 需要和所有它"能看到"的 Key 做点积。如果序列被切到不同 GPU 上，就需要通过 Ring 通信把 KV 块在 GPU 之间传递。Ring Attention 的做法是：每个 GPU 先用本地的 KV 算一部分 attention，然后把 KV 块沿 ring 发给下一个 GPU，重复这个过程直到每个 GPU 都见过所有需要的 KV，通信和计算可以 overlap。

在 Decode 阶段，如果历史 context 非常长（比如前面 prefill 了 128K token），那么 KV Cache 本身也是按序列维度分散在各 GPU 上的。Decode 的新 token 只有 1 个 Query，但需要和 128K 的 Key 做 attention。此时每个 GPU 只需要用本地存储的那部分 KV Cache 算局部 attention score，然后做一次 AllReduce 汇总。这实际上分摊了 KV Cache 的显存和访存压力。

**总结**：SP 在 Prefill 阶段解决的是超长序列的计算和内存瓶颈；在 Decode 阶段主要用于分摊超长上下文的 KV Cache。

---

## 四、直觉性的全局图景

你可以这样建立直觉：

Prefill 阶段的核心矛盾是**计算量太大**——S 个 token 的大矩阵乘法。TP 把每个矩阵乘法切小，PP 让不同层的计算流水线化，SP 把序列维度也切开。三者分别从"宽度""深度""长度"三个维度瓜分了这个庞大的计算任务。Prefill 是 compute-bound 的，只要切分后每个 GPU 上的计算粒度足够大（矩阵够大），Tensor Core 就能高效运转，并行效率高。

Decode 阶段的核心矛盾是**访存太重而计算太少**——每步只有 1 个 token，但要读取全部权重和 KV Cache。TP 分摊了权重和 KV Cache 的读取量，PP 对延迟不友好但能扩展模型规模，SP 在超长上下文时分摊 KV Cache。但无论怎么并行，每一步 decode 的通信开销相对于计算量来说都偏大，导致并行效率远低于 Prefill。

这也是为什么现在业界越来越多地采用 **Prefill/Decode 解耦（Disaggregated Serving）** 的架构——用不同的 GPU 集群分别处理 Prefill 和 Decode，对 Prefill 集群用较大的 TP/SP 度来追求吞吐，对 Decode 集群用较小的 TP 度来降低通信占比、提高单 token 延迟表现。这正是从上面这些分析中自然推导出的系统设计选择。
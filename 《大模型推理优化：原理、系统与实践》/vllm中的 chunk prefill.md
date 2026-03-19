

## 第一章：背景——LLM 推理的两阶段问题

### 1.1 Prefill 与 Decode 的本质差异

LLM 的推理过程天然分为两个阶段，它们的计算特性截然不同：

**Prefill 阶段**接收用户的完整 prompt（可能有数千甚至上万个 token），一次性计算所有 token 的 KV Cache 并输出第一个生成 token。这个过程是**计算密集型（Compute-bound）**的——大量的矩阵乘法让 GPU 算力跑满，衡量它的延迟指标是 TTFT（Time To First Token）。

**Decode 阶段**则是自回归地逐个生成 token，每次迭代只处理 1 个新 token。这个过程是**访存密集型（Memory-bound）**的——每次生成需要读取整个 KV Cache，但计算量极小，GPU 大量算力空闲。衡量它的延迟指标是 TPOT（Time Per Output Token）或 ITL（Inter-Token Latency）。

### 1.2 传统调度的困境

假设系统中同时有这样的请求：请求 A 是新到的长 prompt（8192 tokens），需要做 Prefill；请求 B、C、D 正在生成中，各自等待 Decode（每个只需处理 1 token）。

传统连续批处理有两种策略，都存在问题。

如果采用 **Prefill 优先**策略，请求 A 的 8192 token Prefill 会独占整个迭代，可能耗时 200ms。在此期间 B、C、D 的 Decode 完全被阻塞，用户体感就是"正在输出的文字突然卡住了"。

如果采用 **Decode 优先**策略，B、C、D 的 Decode 不受影响，但请求 A 必须等到所有 Decode 完成才能开始 Prefill，导致新用户的第一个字迟迟出不来，TTFT 大幅增长。

本质矛盾在于：传统方案中，Prefill 和 Decode 是互斥的，无法在同一次迭代中同时高效执行。

### 1.3 资源浪费的深层问题

Decode 阶段每个请求只处理 1 个 token，即使 batch 里有 100 个 Decode 请求，总共也就 100 个 token 的计算量。GPU 的算力在 Decode 阶段利用率可能只有 5%，剩余 95% 在空转。如果能把 Prefill 的计算"匀"到这些空闲算力上，系统整体吞吐量就能大幅提升。

---

## 第二章：Chunked Prefill 的核心设计

### 2.1 核心思想

Chunked Prefill 的思路可以用一句话概括：把长 Prefill 切成固定大小的 chunk，每次迭代只处理一个 chunk，剩余预算留给 Decode 请求，从而实现 Prefill 和 Decode 在同一个 iteration 内混合执行。

### 2.2 Token Budget 机制

系统设定一个每次迭代的 token 预算（由 `max_num_batched_tokens` 参数控制），例如 2048。每次迭代的调度按如下步骤分配预算：第一步，为所有正在 running 状态的 Decode 请求分配预算，每个消耗 1 token；第二步，用剩余预算为 Prefill 请求分配 chunk。例如 200 个 Decode 请求消耗 200 token 后，剩余 1848 个 token 的预算可以分配给一个 Prefill chunk，长 prompt 中未处理的部分留到后续迭代。

### 2.3 时间线对比

**没有 Chunked Prefill 的情况**下，一个 8192 token 的 Prefill 独占多个迭代，Decode 请求在此期间全部被阻塞，ITL 出现巨大毛刺。

**有了 Chunked Prefill 之后**，每个迭代中 Prefill 只处理一小块，Decode 请求每轮都能执行，ITL 保持平稳。代价是 Prefill 被拉长了多个迭代才能完成，TTFT 会略有增加。这本质上是用 TTFT 换 ITL 平稳性的 trade-off。

---

## 第三章：调度器（Scheduler）的完整逻辑

### 3.1 调度优先级

vLLM 的调度器遵循三级优先级。最高优先级给正在 Running 的 Decode 请求，它们已经在生成中不能中断，每个只消耗 1 token 的预算。其次是正在 Running 但 Prefill 未完成的请求（之前的 chunk 已部分处理），需要继续处理下一个 chunk。最低优先级给 Waiting 队列中新到的 Prefill 请求，只有剩余预算足够时才会接纳。

### 3.2 调度伪代码

```python
def schedule_one_iteration():
    budget = max_num_batched_tokens  # 例如 2048
    scheduled = []

    # 第一步：调度所有 Running 状态的 Decode 请求（不可被抢占）
    for req in running_queue:
        if req.is_decoding():
            budget -= 1
            scheduled.append((req, num_tokens=1))

    # 第二步：调度正在进行中的 Chunked Prefill（继续处理）
    for req in running_queue:
        if req.is_chunked_prefill_in_progress():
            chunk = min(req.remaining_prefill, budget)
            if chunk > 0:
                budget -= chunk
                scheduled.append((req, num_tokens=chunk))

    # 第三步：接纳新的 Prefill 请求
    for req in waiting_queue:
        if budget <= 0:
            break
        if not enough_kv_cache_blocks(req):
            break
        chunk = min(req.total_prompt_len, budget)
        budget -= chunk
        scheduled.append((req, num_tokens=chunk))
        move_to_running(req)

    return scheduled
```

### 3.3 完整调度示例

假设系统 budget = 2048，Running 中有请求 B（Decode，KVCache=500）和 请求 C（Decode，KVCache=300），Waiting 中有请求 A（Prefill，prompt=6000 tokens）和请求 D（Prefill，prompt=100 tokens）。

Iter 1 中，B 和 C 各消耗 1 token（剩余 2046），A 获得 chunk0 共 2046 tokens，A 已完成 2046/6000。Iter 2 同理，A 获得 chunk1 再完成 2046 tokens，累计 4092/6000。Iter 3 中，B 和 C 各消耗 1 token 后，A 只剩 1908 tokens 全部处理完，budget 还剩 138，D 的 100 tokens 完整 Prefill 也被塞入，A 和 D 都完成 Prefill 转入 Decode 状态。从 Iter 4 开始，A、B、C、D 四个请求全部在 Decode，而在整个过程中 B 和 C 的 Decode **从未被阻塞**。

---

## 第四章：Attention 的分块计算原理

### 4.1 标准 Full Prefill 的 Attention

以一个 8 token 的 prompt 为例（简化说明），一次性 Prefill 会构建完整的 Q、K、V，计算 8×8 的 causal attention score 矩阵（下三角可见，上三角被 mask）。

### 4.2 切块后的计算方式

将 8 token 切成两块，每块 4 tokens。

**Chunk 0（tokens 1~4）** 是第一块，前面没有历史 KV Cache，做标准的 4×4 Causal Self-Attention。计算完成后，k1~k4、v1~v4 存入 KV Cache。

**Chunk 1（tokens 5~8）** 是关键。Q 只有当前 chunk 的 4 个 token，但 K/V 必须包含所有历史加上当前的内容。Score 矩阵形状为 4×8，其中左半部分（对应 KV Cache 中的 k1~k4）全部可见（因为这些 token 的位置都在当前 Q 之前），右半部分（对应当前 chunk 内的 k5~k8）施加标准 causal mask。

```
Score 矩阵（Chunk 1, 4×8）：

        k1  k2  k3  k4 │ k5  k6  k7  k8
  q5 [  ✓   ✓   ✓   ✓  │  ✓   ✗   ✗   ✗  ]
  q6 [  ✓   ✓   ✓   ✓  │  ✓   ✓   ✗   ✗  ]
  q7 [  ✓   ✓   ✓   ✓  │  ✓   ✓   ✓   ✗  ]
  q8 [  ✓   ✓   ✓   ✓  │  ✓   ✓   ✓   ✓  ]
       ────────────────   ─────────────────
       KV Cache 部分       当前 chunk 内部
       全部可见（全 ✓）     Causal mask（下三角）
```

### 4.3 数学等价性

对于序列中任意位置 $i$ 的 attention 输出：

$$o_i = \frac{\sum_{j=1}^{i} \exp(q_i \cdot k_j / \sqrt{d}) \cdot v_j}{\sum_{j=1}^{i} \exp(q_i \cdot k_j / \sqrt{d})}$$

这个计算只依赖 $q_i$ 和 ${k_j, v_j}_{j \leq i}$。不管这些 K、V 是一次性算出来的，还是分批算出来存在 KV Cache 中再读出来的，只要数值相同，结果就严格等价。Chunked Prefill **不是近似**，它与 Full Prefill 在数学上完全相同。

---

## 第五章：混合 Batch 中的 Attention Mask

### 5.1 同一 Batch 里 Prefill 与 Decode 并存

一次 iteration 的 batch 可能包含：请求 A 的 Prefill chunk（2046 tokens，已有 KV Cache 2050 tokens），请求 B 的 Decode（1 token，KV Cache 512 tokens），请求 C 的 Decode（1 token，KV Cache 1024 tokens）。

每个请求的 attention 矩阵形状完全不同——A 是 (2046 × 4096)，B 是 (1 × 513)，C 是 (1 × 1025)。请求之间必须完全隔离，A 的 token 不能 attend to B 或 C 的 token。

### 5.2 实际实现不构造完整 Mask

vLLM 不会构造一个覆盖所有请求的巨大 mask 矩阵。它通过传递每个请求的元信息（`seq_lens`、`query_lens`、`block_tables`）给 attention kernel，由 kernel 内部根据这些信息为每个请求独立计算 attention。

---

## 第六章：计算执行的核心——哪些混合、哪些分开（重点）

这是理解 Chunked Prefill 实现细节的最关键部分。一个 Transformer 层的计算分为两大类，它们的混合策略完全不同。

### 6.1 两大类计算

**第一类是 Token-wise 计算**，不涉及 token 之间的交互，包括 QKV Projection（Linear）、Output Projection（Linear）、MLP/FFN（Linear×2）、LayerNorm/RMSNorm。每个 token 独立计算，输入输出形状都是 `(token_count, hidden_dim)`。

**第二类是 Token 间交互计算**，即 Attention。Q 需要和 K 做点积，涉及 token 之间的交互关系，而且不同请求之间必须隔离。

### 6.2 Token-wise 计算：完全混合，统一 GEMM

对于 Linear、MLP、Norm 这些操作，vLLM 把所有请求的所有 token **直接拼成一个大矩阵**，用一次 GEMM（General Matrix Multiplication）统一计算。

```
本次 iteration 调度了：
  请求 A (Prefill chunk):  2046 tokens → (2046, hidden_dim)
  请求 B (Decode):         1 token    → (1, hidden_dim)
  请求 C (Decode):         1 token    → (1, hidden_dim)

直接拼接：
  X = (2048, hidden_dim)

QKV Projection（一次 GEMM）：
  [Q, K, V] = X @ W_qkv
  → (2048, hidden_dim) × (hidden_dim, 3 × head_dim × num_heads)
  → 结果 (2048, 3 × head_dim × num_heads)

MLP 同理：
  gate = X @ W_gate       ← 一次 GEMM
  up   = X @ W_up         ← 一次 GEMM
  out  = SiLU(gate) * up @ W_down   ← 一次 GEMM
```

GPU 在执行这些 GEMM 时，**完全不关心哪个 token 属于 Prefill 哪个属于 Decode**，它只看到一个 (2048, hidden_dim) 的矩阵。这种拼接使得矩阵更大，GEMM 的计算强度更高，GPU 利用率更好——这正是 Chunked Prefill 能利用 Decode 阶段闲置算力的关键原因。

### 6.3 Attention 计算：逻辑分开，物理上一次 Kernel Launch

Attention 无法像 GEMM 一样简单拼接，原因有三个。第一，不同请求的 KV Cache 长度不同，attention score 矩阵的形状各异（A 是 2046×4096，B 是 1×513，C 是 1×1025）。第二，不同请求之间必须完全隔离，不能互相 attend。第三，Prefill chunk 内部需要 causal mask，而 Decode 和对 KV Cache 的 attention 不需要 mask（全部可见）。

vLLM 使用的 FlashAttention / FlashInfer kernel 的解决方案是：**发起一次 kernel launch，但 kernel 内部根据元信息为每个请求独立计算 attention。**

```
┌──────────────── 一次 Attention Kernel Launch ────────────────┐
│                                                               │
│  输入：                                                        │
│    Q_all = (2048, num_heads, head_dim)    ← 所有请求的 Q 拼接  │
│    KV Cache（PagedAttention 的 block table）                   │
│    元信息：                                                     │
│      query_start_loc = [0, 2046, 2047]   ← 每个请求 Q 的起始位置│
│      seq_lens = [4096, 513, 1025]        ← 每个请求的总上下文长度│
│      context_lens = [2050, 512, 1024]    ← 每个请求的 KV Cache 长│
│                                                               │
│  Kernel 内部（GPU 线程并行）：                                   │
│    线程组 1~M:    处理请求 A 的 attention (2046 × 4096)         │
│    线程组 M+1:    处理请求 B 的 attention (1 × 513)             │
│    线程组 M+2:    处理请求 C 的 attention (1 × 1025)            │
│    → 各请求的线程组独立并行，互不干扰                             │
│                                                               │
│  输出：                                                        │
│    attn_out = (2048, num_heads, head_dim)  ← 拼接好的结果       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

从 CUDA 的角度看，这是一次 kernel launch（一次 `<<<grid, block>>>` 调用），减少了 launch 开销。从计算逻辑看，每个请求的 attention 完全独立。这种设计称为 **ragged batch**（不规则 batch）——batch 中每个样本的序列长度不同，kernel 能正确处理。

### 6.4 FlashInfer 的具体 API

vLLM v1 引擎默认使用 FlashInfer 作为 attention backend，它提供了 `plan()` + `run()` 两步接口：

```python
import flashinfer

wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper()

# plan 阶段：预计算 kernel 调度方案
wrapper.plan(
    qo_indptr=[0, 2046, 2047, 2048],  # 每个请求 Q 的偏移量
    kv_indptr=[0, 8, 9, 10],          # KV page table 偏移
    kv_indices=page_indices,           # 各请求的 KV Cache page 地址
    kv_last_page_len=[...],           # 最后一页的有效长度
    num_qo_heads=32,
    num_kv_heads=8,                    # 支持 GQA
    head_dim=128,
)

# run 阶段：一次 kernel launch 执行所有请求的 attention
output = wrapper.run(
    q=all_queries,                     # (2048, 32, 128)
    paged_kv_cache=(k_cache, v_cache),
    causal=True,
)
# output: (2048, 32, 128)
```

`plan()` 预先计算哪些 GPU 线程处理哪个请求，`run()` 执行实际计算。调度方案可以被复用以减少开销。

### 6.5 一次完整 Iteration 的执行流程

```
┌────────────────────── 一次 Forward Pass ──────────────────────┐
│                                                                │
│  输入: 拼接后的 token IDs (2048,)                              │
│         ↓                                                      │
│  Embedding: (2048,) → (2048, hidden_dim)           ← 统一计算  │
│         ↓                                                      │
│  ┌──────── Transformer Layer × N ────────┐                     │
│  │                                        │                     │
│  │  RMSNorm:     (2048, h) → (2048, h)   │  ← 统一             │
│  │       ↓                                │                     │
│  │  QKV Linear:  (2048, h) → Q, K, V     │  ← 统一 GEMM        │
│  │       ↓                                │                     │
│  │  ★ Attention:                          │                     │
│  │    一次 kernel launch                  │                     │
│  │    内部按请求独立计算：                  │                     │
│  │      A: (2046 × 4096) prefill chunk    │                     │
│  │      B: (1 × 513) decode               │                     │
│  │      C: (1 × 1025) decode              │                     │
│  │    输出拼接: (2048, h)                 │                     │
│  │       ↓                                │                     │
│  │  Output Linear: (2048, h) → (2048, h)  │  ← 统一 GEMM       │
│  │       ↓                                │                     │
│  │  RMSNorm:     (2048, h) → (2048, h)   │  ← 统一             │
│  │       ↓                                │                     │
│  │  MLP/FFN:     (2048, h) → (2048, h)   │  ← 统一 GEMM        │
│  │                                        │                     │
│  └────────────────────────────────────────┘                     │
│         ↓                                                      │
│  LM Head: (2048, h) → (2048, vocab_size)           ← 统一 GEMM │
│         ↓                                                      │
│  Sampling:                                                     │
│    只对每个请求的最后一个 token 位置采样                          │
│      A → 第 2046 个位置（chunk 末尾） → 生成 token              │
│      B → 第 2047 个位置               → 生成 token              │
│      C → 第 2048 个位置               → 生成 token              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

整条流水线中，**只有 Attention 这一步**需要区分不同请求。其余所有计算（Embedding、Norm、所有 Linear/GEMM、Sampling 前的 LM Head）都是把所有 token 拼成一个大矩阵统一执行。

### 6.6 一句话总结计算逻辑

**GEMM 混在一起，Attention 逻辑上分开但物理上在一个 kernel 里并行执行。** Linear、MLP、Norm 等 token-wise 操作把所有请求的 token 拼成一个大矩阵统一计算；Attention 通过一次 kernel launch 发射，但 kernel 内部根据元信息为每个请求独立计算各自的 attention，请求之间完全隔离。

---

## 第七章：关键参数与调优

### 7.1 核心参数

`enable_chunked_prefill`（新版 vLLM 默认开启）控制是否启用 Chunked Prefill。`max_num_batched_tokens`（典型值 2048）控制每次迭代的 token 预算上限，是整个系统行为的核心旋钮。

### 7.2 调优权衡

这个参数设得大（如 8192），Prefill 切块少，TTFT 更短，但每个 iteration 耗时更长，Decode 的 ITL 波动更大，更接近传统无切块行为。设得小（如 512），每次 iteration 很快完成，Decode 的 ITL 非常平稳，但长 Prefill 需要很多个 iteration 才能完成，TTFT 会增加。

经验法则是：对于在线服务（延迟敏感），建议设为 2048 左右，在 TTFT 和 ITL 之间取得较好平衡；对于离线批处理（吞吐优先），可以设大一些或直接关闭 Chunked Prefill。

---

## 第八章：与其他技术的协同

### 8.1 与 Prefix Caching 的交互

如果多个请求共享相同的 system prompt，Prefix Caching 可以让这些 token 的 KV Cache 被复用而无需重算。Chunked Prefill 只需处理非共享部分，两者可以协同工作，进一步减少计算量。

---

## 第九章：局限性

Chunked Prefill 的主要局限包括三点。第一，TTFT 会增加——长 prompt 被切成多个 chunk 分多次处理，首个 token 的生成时间比一次性 Prefill 更长。第二，实现复杂度上升——调度器、attention kernel、KV Cache 管理都需要支持"部分 Prefill"的中间状态。第三，超短 Prefill 不受益——如果 prompt 本身就很短（比如 100 tokens），小于一个 chunk，Chunked Prefill 没有额外收益（但也没有额外开销）。

---

## 全景总结

```
  问题：长 Prefill 阻塞 Decode → ITL 毛刺 + GPU 资源浪费
                      ↓
  方案：将 Prefill 切成 chunk，与 Decode 混合调度
                      ↓
  调度：Token Budget 机制，Decode 优先 → Prefill 填充剩余预算
                      ↓
  计算：GEMM 统一执行 / Attention 单 kernel 内按请求独立计算
                      ↓
  收益：ITL 稳定 + 吞吐提升 + 显存更可控
                      ↓
  代价：TTFT 略增 + 实现复杂度提高
```

Chunked Prefill 本质上是一种**时分复用**策略：把 GPU 的每个时间片合理分配给 Prefill 和 Decode，让两种计算特性互补——Prefill 的计算密集填满 Decode 阶段闲置的算力，实现整体效率的最大化。

# 满血千B参数 MoE 模型推理集群：组网架构与并行策略全解析

以 DeepSeek-V3/R1（671B 参数，256 routed experts，每 token 激活 8 个）为核心案例，结合当前最先进的 NVIDIA GB200 NVL72 硬件，以及 DeepSeek、SGLang/LMSYS、NVIDIA Dynamo 等一线工程实践，全面拆解生产级推理集群的组网和并行切分策略。

---

## 一、核心架构理念：Prefill-Decode 分离 + 大规模 Expert Parallelism

千B级 MoE 模型的推理有两个根本性挑战。第一，模型参数总量巨大（DeepSeek-V3 为 671B），即使用 FP8 量化也需要 ~335GB 显存，远超单卡容量；256 个 expert 的权重必须分散存放。第二，Prefill 阶段是计算密集型（大量矩阵乘法），Decode 阶段是访存密集型（每步只生成一个 token，但需要加载 expert 权重），两者对硬件的需求截然不同，混在一起会互相拖累。

因此，一线大厂（DeepSeek、字节、阿里、NVIDIA 参考架构）普遍采用 **Prefill-Decode 分离（PD Disaggregation）** 的架构：Prefill 和 Decode 运行在不同的 GPU 池上，各自使用不同的并行策略，通过 RDMA 传输 KV Cache。

---

## 二、以 H800/H100 集群为例的实际部署（DeepSeek 官方方案）

DeepSeek 在 2025 年 2 月开源周公布了其推理系统的完整架构，这是目前最具参考价值的一线生产实践。

### 2.1 硬件与组网拓扑

DeepSeek 使用 H800 集群（受出口管制，NVLink 带宽为 400 GB/s，而非 H100 的 900 GB/s）。每个节点 8 卡 H800，节点内 NVLink 互联，节点间通过 InfiniBand 互联（推测为 400 Gbps）。在高峰期，V3+R1 推理服务合计使用 278 个节点（2224 张 H800），平均约 227 个节点（1816 张 GPU）。

### 2.2 并行策略详解

**Prefill 阶段**的部署单元为 4 个节点（32 张 GPU）：

Routed Expert 采用 EP32（256 个 expert 分布在 32 张 GPU 上，每张 GPU 持有 8 个 expert），加上 32 个冗余 expert（EPLB 复制的热门 expert），实际是 288 个 expert slot 分布在 32 张 GPU 上，每 GPU 约 9 个 routed expert 加 1 个 shared expert。MLA（Multi-head Latent Attention）和 Shared Expert 采用 DP32，即 32 张 GPU 各自独立处理不同请求的 attention。不使用 Tensor Parallelism（TP=1），因为在大 EP 规模下 TP 反而不利：Dense FFN 中间维度 18432 被 TP32 切分后每片只有 576，无法对齐 GPU 的 128 alignment boundary，计算效率很差。

Prefill 的核心思想是：用足够大的 EP 将 batch 做大（每 GPU 处理 16384 个 token），让 GroupGEMM 的算术密度上去，同时用 Two-Batch Overlap（TBO）将通信隐藏在计算背后——把一个大 batch 切成两个 micro-batch，当一个 micro-batch 在计算 MoE 时，另一个同时做 all-to-all dispatch/combine 通信。

**Decode 阶段**的部署单元为 18 个节点（144 张 GPU）：

Routed Expert 采用 EP144（256+32 冗余 = 288 个 expert slot 分布在 144 张 GPU 上，每 GPU 仅持有 2 个 routed expert 加 1 个 shared expert）。MLA/Shared Expert 同样 DP144。这是极致的"Wide Expert Parallelism"：每张 GPU 只放极少的 expert，带来三个关键好处。其一，每个 expert 的权重小，weight loading 更快，GroupGEMM 的 arithmetic intensity 更高。其二，每 GPU 留出大量显存给 KV Cache，支撑更大的 batch size（128-256 个并发请求）。其三，144 张 GPU 的聚合 HBM 带宽极大，整体吞吐量高。Decode 同样使用 TBO 的 5-stage pipeline 实现通信-计算重叠。

### 2.3 数据流路径

一个完整的请求流转如下。用户请求到达后，Prefill Load Balancer 将其分配到一个 Prefill 节点组（4 nodes/32 GPUs）。Prefill 完成后，KV Cache 通过 RDMA 传输到 Decode 节点组（18 nodes/144 GPUs）。Decode 节点组逐 token 生成，每步的 MoE 层需要跨 144 张 GPU 做 all-to-all dispatch（把 token 路由到持有被选中 expert 的 GPU）、expert 计算、all-to-all combine（把结果汇总回来）。

### 2.4 网络流量特征

MoE 的 all-to-all 通信模式与传统的 AllReduce 截然不同。AllReduce 是所有节点等量交换，流量模式规则；而 all-to-all dispatch 的流量取决于 expert 路由结果，是高度不规则的——某些 GPU 可能收到大量 token（热门 expert），某些几乎没有。这使得 EPLB（Expert Parallel Load Balancer）成为必需组件。DeepSeek 使用 EPLB 定期重新分配 expert，将热门 expert 复制多份分散到不同 GPU，冷门 expert 合并。

---

## 三、GB200 NVL72 时代的推理架构演进

在你前面描述的 NVL72 架构上部署 DeepSeek-R1 级模型，组网逻辑发生根本性变化。

### 3.1 Scale-up 域扩大的革命性影响

GB200 NVL72 在单机柜内通过 NVLink 5 将 72 张 B200 GPU 全互联，每 GPU 1.8 TB/s 双向带宽，聚合 130 TB/s。这意味着 **64 张 GPU 的 Wide EP decode 完全在 NVLink 域内完成 all-to-all 通信**，不需要走 InfiniBand。这对 MoE 推理是变革性的：

在 H100/H800 集群上，EP144 的 decode 需要 18 个节点跨 IB 做 all-to-all，IB 的带宽（每 GPU 约 50 GB/s）和延迟成为严重瓶颈。而在 NVL72 上，EP64（每 GPU 4 个 expert，64 GPU 就放满 256 个 expert）的全部 all-to-all 通信走 NVLink，带宽高出 36 倍，延迟低一个量级。NVIDIA 的仿真结果显示，NVL72 上 disaggregated serving + Wide EP 可实现 6 倍以上的吞吐提升（相比 co-located serving）。

### 3.2 NVL72 上的并行切分方案

根据 NVIDIA 官方博客和 TensorRT-LLM 的实践：

**Decode 阶段（核心受益者）**：使用 EP64，64 张 GPU 各持有 4 个 expert（256/64=4），全部在单机柜 NVLink 域内。剩余 8 张 GPU 可用于 Prefill 或冗余。每 GPU 因为只存 4 个 expert 权重，大量显存释放给 KV Cache，支撑更大 batch。NVIDIA 报告 EP32 相比 EP8 可实现 1.8 倍的每 GPU 吞吐提升，EP64 效果更优。

**Prefill 阶段**：可以灵活分配——在同一 NVL72 机柜内划分一部分 GPU 做 prefill（如 8-16 张 GPU），也可以专门用独立的 NVL72 机柜做 prefill。Prefill 对 NVLink 带宽的需求没有 decode 那么极端（因为 prefill 是 compute-bound），但 EP 仍然有益于增大 batch 和提高 GroupGEMM 效率。

### 3.3 多机柜 Scale-out

对于超大规模推理服务（同时服务百万用户），单个 NVL72 机柜是一个 **deployment unit**。多个 NVL72 机柜通过 ConnectX-7 NIC + InfiniBand/RoCE 网络互联，但这个跨机柜网络主要承载的是：Prefill 到 Decode 之间的 KV Cache 传输（RDMA）；Data Parallelism 层面的流量调度和负载均衡。而 **MoE 的 all-to-all 通信完全不走跨机柜网络**——这是 NVL72 架构的核心优势。

```
多机柜推理集群拓扑：

    ┌──────────────────────────────────────────────────────┐
    │                   Spine Switch Layer                  │
    │              (InfiniBand / RoCE 400-800G)            │
    └──┬──────────┬──────────┬──────────┬──────────┬───────┘
       │          │          │          │          │
   ┌───┴───┐  ┌──┴────┐  ┌──┴────┐  ┌──┴────┐  ┌──┴────┐
   │NVL72  │  │NVL72  │  │NVL72  │  │NVL72  │  │NVL72  │
   │Rack 1 │  │Rack 2 │  │Rack 3 │  │Rack 4 │  │Rack N │
   │72 GPU │  │72 GPU │  │72 GPU │  │72 GPU │  │72 GPU │
   │Decode │  │Decode │  │Decode │  │Prefill│  │Prefill│
   │Pool   │  │Pool   │  │Pool   │  │Pool   │  │Pool   │
   └───────┘  └───────┘  └───────┘  └───────┘  └───────┘
   
   每个 NVL72 机柜内部：
   - 72 GPU 通过 NVLink 全互联 (130 TB/s)
   - EP64 decode 或 EP32 prefill 完全在机柜内完成
   - 4× ConnectX-7 NIC per compute tray (scale-out)
   
   机柜间 InfiniBand 网络承载：
   - KV Cache transfer (Prefill → Decode, RDMA)
   - 请求调度、负载均衡
   - Data Parallelism 层面的协调
   - MoE all-to-all 通信 **不** 走此网络
```

### 3.4 NVIDIA Dynamo 编排框架

NVIDIA Dynamo 是专为这种架构设计的推理编排框架，负责：自动将 Prefill 和 Decode 分离到不同 GPU 池；根据实时负载做 Prefill/Decode 之间的动态配比调整（rate matching）——当输入序列长时增加 Prefill GPU，当并发请求多时增加 Decode GPU；KV Cache 路由——将请求调度到已经缓存了相关 KV Cache 的 Decode GPU，避免重复传输；通过 Kubernetes 进行弹性扩缩容。

---

## 四、为什么不用 Tensor Parallelism / Pipeline Parallelism？

在传统 dense 模型（如 Llama-70B）推理中，TP 是主流选择——8 卡 TP8 把模型切到 8 张 GPU，每张只存 1/8 的权重。但对于千B MoE 模型，一线实践几乎不用 TP 或 PP，原因非常深刻：

TP 的问题在于，MoE 模型的 FFN 层已经天然是稀疏的——256 个 expert 中只激活 8 个。如果用 TP 把每个 expert 切开，那每个 expert 的矩阵片段太小，计算效率极低（达不到 GPU 峰值算力）。而 EP 是把不同的完整 expert 放到不同 GPU 上，每个 expert 内部保持完整，计算效率高得多。

PP 的问题在于引入流水线气泡（bubble），对延迟敏感的推理场景不可接受——用户不能等第一个流水线阶段算完再算第二个阶段。

因此 **EP + DP** 成为 MoE 推理的标准组合：EP 负责分散 expert 权重和计算负载，DP 负责 attention 层的并行处理。

---

## 五、完整的并行策略总结表

```
┌──────────────┬────────────────────────────────┬───────────────────────────────────┐
│              │       H800/H100 集群            │         GB200 NVL72 集群           │
│              │    (DeepSeek 实际生产)           │      (NVIDIA 参考架构)             │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ Prefill 单元  │ 4 nodes × 8 GPU = 32 GPU       │ 8-16 GPU (同一 NVL72 机柜内       │
│              │ Routed EP32, MLA/Shared DP32   │  或专用 prefill 机柜)              │
│              │ TBO (Two-Batch Overlap)        │ EP32, DP, TBO                    │
│              │ 每 GPU ~9 routed + 1 shared    │                                   │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ Decode 单元   │ 18 nodes × 8 GPU = 144 GPU     │ 64 GPU (单 NVL72 机柜内)          │
│              │ Routed EP144, MLA/Shared DP144 │ EP64, DP, 全 NVLink 通信           │
│              │ 每 GPU ~2 routed + 1 shared    │ 每 GPU 4 routed experts           │
│              │ all-to-all 走 IB (瓶颈)         │ all-to-all 走 NVLink (130 TB/s)   │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ KV Transfer  │ RDMA over IB                   │ RDMA over IB/RoCE (跨机柜)        │
│              │                                │ 或 NVLink (机柜内 P→D)             │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ Scale-out    │ IB 400G Fat-tree               │ ConnectX-7 400G × 4/tray          │
│ 网络         │ 节点间 all-to-all 通信           │ 仅承载 KV 传输和调度，              │
│              │                                │ MoE 通信不出机柜                   │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ TP           │ 不使用 (TP=1)                   │ 不使用 (TP=1)                     │
│ PP           │ 不使用                          │ 不使用                             │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ 负载均衡      │ EPLB + Prefill/Decode 各自     │ EPLB (online 动态) +              │
│              │ Load Balancer                  │ Dynamo Planner 自动配比            │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ 精度         │ FP8 (MatMul/dispatch)          │ FP8/FP4 (Blackwell 支持 FP4)      │
│              │ BF16 (MLA core attention)      │                                   │
├──────────────┼────────────────────────────────┼───────────────────────────────────┤
│ 关键优化      │ TBO, DeepEP, DeepGEMM,         │ Wide-EP, NCCL 定制 EP kernel,     │
│              │ EPLB, 磁盘 KV Cache             │ MTP (Multi-Token Prediction),     │
│              │                                │ Dynamo disaggregated serving      │
└──────────────┴────────────────────────────────┴───────────────────────────────────┘
```

---

## 六、经济性数据：一线厂商的真实成本

DeepSeek 在 2025 年 2 月公布了一组极具参考价值的运营数据：

日均使用约 227 个 H800 节点（1816 张 GPU），按 $2/GPU/hour 计算，日成本约 $87,072。当日处理了 608B 输入 token（其中 56.3% 命中磁盘 KV Cache）和 168B 输出 token。如果全按 R1 定价计算，理论日收入 $562,027，**成本利润率 545%**。单节点（8 GPU）的吞吐为 prefill ~73.7k tokens/s、decode ~14.8k tokens/s。

SGLang 团队在 12 节点（96 张 H100）上复现了 DeepSeek 的架构，达到了 per-node 52.3k input tokens/s 和 22.3k output tokens/s，与 DeepSeek 官方数据相当接近（差距 ~6%）。按此吞吐计算，推理成本仅 $0.20/1M output tokens，约为 DeepSeek 官方 API 价格的 1/5。

---

## 七、关键洞察与总结

千B MoE 模型的推理集群设计，本质上是围绕 **all-to-all 通信模式** 做文章。不同于训练中以 AllReduce 为主的通信模式，MoE 推理的 expert dispatch/combine 是不规则的、数据依赖的 all-to-all 通信，这决定了组网的核心矛盾。

在 H100/H800 时代，这个矛盾的解法是用大规模 EP 跨越多节点，承受 IB 的带宽瓶颈，用 TBO 把通信延迟藏在计算后面。DeepSeek 用 18 节点 144 GPU 做 decode，本质是在说"我可以忍受 IB 的延迟，因为 batch 足够大、计算足够久，通信能被 overlap 掉"。

在 GB200 NVL72 时代，这个矛盾被硬件架构性地解决了——72 GPU NVLink 全互联让 EP64 的 all-to-all 在机柜内以 130 TB/s 完成，MoE 通信从"需要被 overlap 的瓶颈"变成了"几乎可以忽略的开销"。这就是为什么 NVIDIA 称 NVL72 是"为 MoE 推理而生"的架构。
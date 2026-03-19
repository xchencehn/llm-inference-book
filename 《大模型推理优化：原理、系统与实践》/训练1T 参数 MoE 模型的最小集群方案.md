

## 零、定义问题：什么是"满血"~1T MoE 模型？

在动手算之前，我们必须先把"目标模型"钉死。参考当前最先进的 MoE 模型实践（DeepSeek-V3 671B、Kimi K2 1T），我们定义一个合理的 ~1T 参数 MoE 架构：

```
模型参数卡片（参考 DeepSeek-V3 放大 + Kimi K2 实践）
─────────────────────────────────────────────
总参数量：            ~1T（1,000B）
每 token 激活参数：    ~50B
隐藏维度 (d_model)：   8,192
层数 (L)：             80 层（其中 ~78 层 MoE，2 层 dense）
注意力头数：           64 heads，GQA/MLA
每层 MoE 配置：
  ├── 共享专家 (shared)：   1 个
  ├── 路由专家 (routed)：   256 个
  ├── TopK 激活：           8 个
  └── 每个专家 FFN 中间维度：2,560
      (expert_dim ≈ d_model × 4 / num_active ≈ 调整到 2,560)
词表大小：             128K
序列长度：             8,192（预训练主阶段）
训练精度：             FP8 混合精度（权重 BF16 master copy + FP8 计算）
```

这个设计的参数量验证：每个路由专家的参数 ≈ 2 × d_model × expert_intermediate = 2 × 8192 × 2560 ≈ 42M。256 个路由专家 × 42M ≈ 10.75B/层。加上 shared expert（~160M）和 attention（~200M），每层 MoE 约 11.1B。80 层 × 11.1B ≈ 888B，加上 embedding（128K × 8192 × 2 ≈ 2B）和其他开销，总量约 **~900B–1T**，符合目标。

---

## 一、第一性原理之一：内存墙——模型能不能装下？

### 1.1 训练时的内存需求

训练需要存储的内容远不止权重本身。使用 FP8 混合精度训练（类 DeepSeek-V3 方案），每个参数的内存开销为：

```
每参数内存开销（FP8 混合精度 + Adam 优化器）
────────────────────────────────────────────
BF16 master 权重：                     2 bytes
FP8 前向/反向计算权重副本：             1 byte（可即时量化，不额外存储）
Adam 一阶矩 (m)：                     2 bytes（BF16）
Adam 二阶矩 (v)：                     2 bytes（BF16）
BF16 梯度：                            2 bytes
────────────────────────────────────────────
合计：~8 bytes/参数（使用 ZeRO-1/2 可分片优化器状态）
```

如果不做任何分片，1T 参数 × 8 bytes = **8 TB**。

但 MoE 模型有一个关键特性：**优化器状态只需为每个专家存储一份，不需要全局复制**。使用 ZeRO-1 Data Parallelism，优化器状态可以跨 DP 组分片。

此外还有 **激活值内存（Activation Memory）**。对于 8K 序列长度、micro-batch size = 1 的情况，每层的激活值约为：

```
激活值估算（每层，每 micro-batch）
────────────────────────────────────
attention 输入/输出：       seq_len × d_model × 2 × 2 bytes 
                          = 8192 × 8192 × 2 × 2 = 256 MB
MoE dispatch 缓冲区：     seq_len × d_model × topK × 2 bytes
                          = 8192 × 8192 × 8 × 2 = 1 GB
每层激活（重计算后）：      ~0.5–1 GB（选择性激活重计算）
────────────────────────────────────
80 层 × ~0.75 GB ≈ 60 GB
```

使用选择性激活重计算（Selective Activation Recomputation），可以把激活值内存压缩到 ~30–60 GB 每个 pipeline stage。

### 1.2 GB200 NVL72 的内存容量

```
GB200 NVL72 单机柜内存
────────────────────────────────
GPU HBM3e：72 GPU × 192 GB = 13,824 GB ≈ 13.5 TB
Grace LPDDR5X：36 CPU × 480 GB = 17,280 GB ≈ 17 TB
────────────────────────────────
HBM 总容量：13.5 TB（NVLink 统一地址空间）
```

**关键判断：1T 参数模型的纯权重（BF16）= 2 TB。8 TB 的训练总内存需求，单机柜 13.5 TB 的 HBM 在数值上是装得下的。** 但我们需要更细致地分析。

### 1.3 逐项装载验证

在 MoE 训练中，有一个核心事实：**所有 256 个专家的权重都必须常驻内存**，因为不同 token 会路由到不同专家。不存在"只加载激活的 8 个"的可能——那是推理优化，训练必须全部加载。

```
单机柜（72 GPU，13.5 TB HBM）内存分配
────────────────────────────────────────────
【权重（BF16 master copy）】
  共享部分（attention + embedding + shared expert）：
    ≈ 80层 × ~360M/层 + 2B embedding ≈ 31B → 62 GB
  路由专家：
    256 experts × 80层 × 42M/expert = 860B → 1,720 GB

  权重总计：~1,782 GB ≈ 1.74 TB

【优化器状态（Adam m + v，BF16 各 2B/param，ZeRO-1 分片）】
  如果 DP = N 个机柜，每机柜只存 1/N 的优化器状态
  单副本：1T × 4 bytes = 4 TB
  ZeRO-1 分片后每机柜：4 TB / N

  若 N = 2：2 TB / 机柜
  若 N = 1：4 TB / 机柜（单机柜撑不住！）

【梯度（BF16）】
  1T × 2 bytes = 2 TB
  但梯度可以即算即用、分层累积，不需要同时驻留全部
  每层梯度 ≈ 11.1B × 2 bytes = 22.2 GB
  Pipeline 并行下，最多同时存在几层的梯度

  实际梯度内存：~100–200 GB

【激活值】
  80 层分成 PP stages，选择性重计算后
  每 pipeline stage：~30–60 GB
  
────────────────────────────────────────────
```

**结论：单机柜（1 个 NVL72）无法独立完成 1T MoE 训练。** 核心瓶颈是优化器状态（4 TB）+ 权重（1.74 TB）= 5.74 TB，加上梯度和激活值，总需求约 **6.5–7 TB**。虽然低于 13.5 TB 的 HBM 物理容量，但考虑到内存碎片、CUDA 开销、通信缓冲区等实际消耗，13.5 TB 中可用的有效内存约 11–12 TB，**理论上单机柜可以勉强装下，但没有余量给更大的 batch size**。

然而，这还没考虑一个关键因素——**计算效率和训练速度**。

---

## 二、第一性原理之二：计算墙——要多少算力才能训练完？

### 2.1 训练总 FLOPs 估算

对于 MoE 模型，训练 FLOPs 的核心公式是基于**激活参数量**（非总参数量）：

```
每 token 训练 FLOPs ≈ 6 × 激活参数量
（前向 2×，反向 4×，总共 6×）

每 token FLOPs = 6 × 50B = 300 GFLOP/token
```

训练数据量：参考 DeepSeek-V3 的 14.8T token，一个 1T 模型应该训练至少 **15T tokens**（根据 Chinchilla scaling law 的 MoE 适配版，MoE 模型因为总参数大、激活参数小，通常需要更多数据）。

```
训练总 FLOPs = 300 GFLOP/token × 15T tokens
             = 300 × 10⁹ × 15 × 10¹²
             = 4.5 × 10²⁴ FLOP
             = 4.5 ZettaFLOP
```

### 2.2 GB200 NVL72 的计算能力

每颗 B200 GPU 的 FP8 Tensor Core 算力：

```
B200 单 GPU 规格
────────────────────────
FP8 Tensor Core (dense)：  ~4.5 PFLOPS（考虑到 sparsity 支持可达 9 PFLOPS）
FP8 Tensor Core (实际 MoE 训练利用率)：约 40-50% MFU
有效算力/GPU：4.5 × 0.45 ≈ 2.0 PFLOPS（保守估计）
────────────────────────

单机柜 72 GPU 有效算力：
72 × 2.0 PFLOPS = 144 PFLOPS
```

注：MFU（Model FLOPs Utilization）45% 对于 MoE 模型是合理的。NVIDIA 官方在 Hybrid-EP 博客中展示 DeepSeek-V3 在 Grace Blackwell 上达到 943 TFLOPS/GPU（MXFP8），对应约 20% 的峰值 FP8 利用率——但这是单纯的 MFU 定义不同。如果按照实际训练中测得的 ~900 TFLOPS/GPU 计算：

```
修正计算：
单 GPU 实际吞吐：~900 TFLOPS（MXFP8，来自 Hybrid-EP 实测）
单机柜 72 GPU：72 × 900 TFLOPS = 64.8 PFLOPS
```

### 2.3 训练时间估算

```
单机柜训练时间 = 总 FLOPs / 有效算力
             = 4.5 × 10²⁴ / (64.8 × 10¹⁵)
             = 6.94 × 10⁷ 秒
             = ~803 天
             ≈ 2.2 年
```

这显然太慢了。要把训练时间压缩到 **3 个月（~90 天）** 以内（行业可接受的时间窗口），需要：

```
所需机柜数 = 803 / 90 ≈ 9 个机柜
所需 GPU 数 = 9 × 72 = 648 GPU
```

但这是理想线性扩展的结果。考虑跨机柜通信开销导致的效率下降（scale-out 效率约 85-90%），实际需要：

```
修正后所需机柜数 ≈ 9 / 0.85 ≈ 11 个机柜
所需 GPU 数 ≈ 11 × 72 = 792 GPU
```

**如果目标是 2 个月完成训练：需要 ~16 个机柜，~1,152 GPU。**

---

## 三、第一性原理之三：通信墙——并行策略如何设计？

这是整个推导中最精妙的部分。GB200 NVL72 的双层网络拓扑（NVLink scale-up + InfiniBand/RoCE scale-out）从根本上决定了并行策略。

### 3.1 通信带宽层次结构

```
GB200 NVL72 通信带宽层次
────────────────────────────────────────────────────────
层级 1：GPU ↔ GPU（同一 NVL72 内）
  NVLink 5：1.8 TB/s 双向/GPU
  全机柜聚合：130 TB/s
  延迟：~1-2 μs
  
层级 2：GPU ↔ GPU（跨 NVL72 机柜）
  路径：GPU → NVLink-C2C → Grace LPDDR5X → PCIe Gen5 → ConnectX-7 → IB/RoCE
  每 GPU 对应 1 个 ConnectX-7 400Gbps NIC → ~50 GB/s
  每机柜总 scale-out 带宽：72 × 50 GB/s = 3,600 GB/s = 3.6 TB/s
  延迟：~5-10 μs
────────────────────────────────────────────────────────
带宽比：NVLink 内 / scale-out = 130 TB/s / 3.6 TB/s ≈ 36:1
```

这个 **36:1 的带宽比** 是设计一切并行策略的核心约束。**通信密集型操作必须留在 NVLink 域内，只有通信稀疏的操作才允许跨机柜。**

### 3.2 五维并行策略的设计

对于 ~1T MoE 模型在多机柜 GB200 NVL72 上的训练，我们需要同时使用五种并行维度：

```
并行维度             │ 缩写 │ 通信模式        │ 通信量级       │ 应放在哪里
───────────────────┼─────┼───────────────┼──────────────┼──────────
Expert Parallelism  │ EP  │ All-to-All     │ 极高（动态路由）│ NVLink 域内
Tensor Parallelism  │ TP  │ All-Reduce     │ 极高（每层2次）│ NVLink 域内
Pipeline Parallelism│ PP  │ Point-to-Point │ 中等（层间传递）│ 可跨域
Data Parallelism    │ DP  │ All-Reduce     │ 低（梯度同步） │ 跨域
Context Parallelism │ CP  │ Ring All-to-All│ 中-低          │ NVLink 域内
```

### 3.3 核心推导：NVLink 域内的并行分解

一个 NVL72 机柜有 72 GPU。这 72 GPU 的乘积分解方式决定了域内并行策略：

```
72 = EP × TP × PP_intra × DP_intra × CP

约束条件：
1. EP 必须在 NVLink 域内（all-to-all 通信量巨大）
2. TP 必须在 NVLink 域内（每层两次 all-reduce）
3. PP 可以跨域（点对点通信量小）
4. DP 的梯度同步可以跨域（通过 all-reduce，但 MoE 的梯度稀疏性降低了量）
```

**关键洞察——EP 是 MoE 训练的核心并行维度**：

DeepSeek-V3 在 H800 上使用 EP=64（跨 8 节点），因为 256 个专家需要分布到多个 GPU 上。在 GB200 NVL72 上，情况好得多：72 GPU 全部在 NVLink 域内。

让我们推导最优的域内分解。256 个专家需要分配到 EP 组中。核心权衡是：

- **EP 越大**：每 GPU 的专家数越少，显存压力越小，但 all-to-all 通信的参与方越多
- **EP 越小**：每 GPU 的专家数越多，通信参与方少，但显存压力大

```
最优 EP 推导：
────────────────────────────────────
256 个专家，72 GPU 在 NVLink 域

方案 A：EP = 72
  每 GPU 承载 256/72 ≈ 3.56 个专家 → 不整除，不可行

方案 B：EP = 64
  每 GPU 承载 256/64 = 4 个专家
  72 / 64 = 1.125 → 剩余 8 GPU
  这 8 GPU 可用于 TP=1 下的 DP=1.125 → 不整除

方案 C：EP = 36
  每 GPU 承载 256/36 ≈ 7.1 → 不整除

方案 D：EP = 32
  每 GPU 承载 256/32 = 8 个专家
  72 / 32 = 2.25 → 不整除

方案 E：EP = 8，TP = 1，PP_intra = 9
  72 = 8 × 9 × 1
  每 GPU 承载 256/8 = 32 个专家
  9-way PP 在域内

方案 F（推荐）：EP = 72
  使用 NVIDIA Wide-EP 技术，允许非整除分配
  256 个专家分配到 72 GPU：
  - 40 个 GPU 各承载 4 个专家（40 × 4 = 160）
  - 32 个 GPU 各承载 3 个专家（32 × 3 = 96）
  - 160 + 96 = 256 ✓
  配合 Expert Parallel Load Balancer (EPLB) 动态再平衡
```

但等一下——上面的分析忽略了 PP 和 TP。让我们更系统地思考。

### 3.4 系统化的并行策略设计

**步骤 1：确定是否需要 TP**

TP 主要用于单层参数太大、单 GPU 放不下的情况。让我们算一下：

```
单个 MoE 层的参数量（如果所有专家都在本 GPU）：
────────────────────────
attention 部分：
  Q/K/V/O 投影 ≈ 4 × d² = 4 × 8192² = 268M → 536 MB (BF16)
  
shared expert FFN：
  2 × d × 4d = 2 × 8192 × 32768 = 537M → 1.07 GB

如果 EP=72，每 GPU 上 3-4 个路由专家：
  4 × 42M = 168M → 336 MB

单 GPU 上一层的权重总量 ≈ 536 MB + 1.07 GB + 336 MB ≈ 1.94 GB
────────────────────────
```

192 GB HBM 装一层才 2 GB，显然不需要 TP 来拆分单层。**结论：TP = 1**（不需要 Tensor Parallelism）。这与 DeepSeek-V3 的做法一致——他们也刻意避免了 TP 以减少通信。

**步骤 2：确定 PP（Pipeline Parallelism）的 stage 数**

80 层需要分成 PP stages。PP stages 越多，pipeline bubble 越大（bubble 比例 ≈ (PP-1)/(PP-1+micro_batches)）。

域内 PP 的好处是 NVLink 点对点传输极快。跨域 PP 则走 scale-out 网络，但因为是点对点通信（不是 all-reduce），带宽需求相对可控。

```
每 PP stage 之间的通信量 = batch_size × seq_len × d_model × 2 bytes
假设 micro-batch = 2，seq_len = 8192，d_model = 8192：
= 2 × 8192 × 8192 × 2 = 256 MB

在 NVLink 上：256 MB / (1.8 TB/s) = 0.14 ms → 几乎可以忽略
在 IB 上：256 MB / (50 GB/s) = 5.1 ms → 可以接受
```

**步骤 3：确定 EP 的分配**

这是最关键的设计决策。NVLink 域内的 72 GPU 如何分配给 EP 和 PP？

```
域内分解：72 GPU = EP_intra × PP_intra

候选方案：
────────────────────────────────────────
方案 α：EP = 72，PP = 1（单域内不做 PP）
  - 每 GPU 3-4 个专家
  - 80 层全在每个 GPU 上（每层 3-4 个专家副本）
  - 问题：所有 80 层 × 3-4 个专家 × 42M = 80 × 4 × 42M × 2B = 26.9 GB 权重
  - 加上优化器状态（如果 ZeRO-1 DP 分片）
  - 单 GPU 内存估算：
    权重：26.9 GB
    优化器（ZeRO-1，假设 DP=N_cabinets）：26.9 × 4 / N = 107.5 / N GB
    梯度：26.9 GB
    激活值：80 层 × ~0.75 GB = 60 GB（全部层都在本 GPU 上！）
    总计：26.9 + 107.5/N + 26.9 + 60 = 114 + 107.5/N GB
    N=8 → 127 GB   ← 可以装入 192 GB
    N=4 → 141 GB   ← 可以装入 192 GB
    N=2 → 168 GB   ← 勉强
    
  - 优点：无 pipeline bubble，EP all-to-all 全在 NVLink 内
  - 缺点：每 GPU 要存 80 层的权重，激活值占用大

方案 β：EP = 36，PP = 2
  - 每 GPU 承载 256/36 ≈ 7.1 → 不整除
  
方案 γ：EP = 24，PP = 3
  - 256/24 ≈ 10.7 → 不整除

方案 δ：EP = 8，PP = 9
  - 每 GPU 承载 256/8 = 32 个专家
  - 80 层分成 9 个 PP stage，每 stage ~9 层
  - 单 GPU 内存：
    权重：9 × (32 × 42M + ~360M) × 2B = 9 × (1.344B + 0.36B) × 2 = 30.7 GB
    优化器：30.7 × 4 / N GB
    梯度：30.7 GB
    激活值：9 × 0.75 = 6.75 GB
    总计：68 + 123/N GB
    N=4 → 99 GB  ← 很宽裕
  - 优点：内存非常宽裕，PP stage 间通信少
  - 缺点：EP=8 意味着 all-to-all 在 8 GPU 内，72/8=9 个 EP 组独立
  
方案 ε（DeepSeek-V3 模式适配）：EP = 72，PP_cross = 跨域
  - 域内全部 72 GPU 做 EP，PP 在跨域完成
  - 80 层分成 PP stages，每个 stage = 一个完整的 NVL72 机柜
  - 需要 PP 个机柜
────────────────────────────────────────
```

### 3.5 最优方案的确定

让我们深入比较两个最有前途的方案：

**方案 A（全域 EP，推荐方案）：域内 EP=72，跨域 PP**

```
架构设计：
────────────────────────────────────────
域内（单个 NVL72，72 GPU）：
  EP = 72（全部 GPU 参与 Expert Parallelism）
  每 GPU 承载 ~3-4 个专家/层 × 80 层
  注意力层：全部 72 GPU 以 EP 模式处理（每 GPU 处理 1/72 的 token）
  
跨域：
  PP = K 个机柜（每个机柜处理 80/K 层）
  DP = M（数据并行，梯度同步跨域 all-reduce）
  
总机柜数 = PP × DP

关键：每个 PP stage = 一个完整 NVL72 机柜的 72 GPU
      stage 之间通过 scale-out 网络传递激活值
```

问题是：80 层如果 PP=1（单机柜），每 GPU 必须存储所有 80 层中属于自己的专家。我们已经验证过可行（方案 α）。但如果 PP > 1，可以进一步降低内存压力。

```
PP=2 的情况（每机柜 40 层）：
  单 GPU 权重 = 40 层 × (4 experts × 42M + attention 360M/72 GPU) × 2B
             ≈ 40 × (168M + 5M) × 2 
             = 13.8 GB
  优化器 = 13.8 × 4 / DP = 55.2 / DP GB
  梯度 = 13.8 GB
  激活 = 40 × ~0.4 GB = 16 GB（EP=72 下每 GPU 只处理 1/72 的 token）
  总计 = 44 + 55/DP GB
  DP=4 → 58 GB ← 极其宽裕！
```

等一下——**EP=72 模式下，每 GPU 处理的 token 数只有 total_batch 的 1/72**，这大幅降低了激活值内存！这是一个关键优势。

**方案 B（域内 EP×PP 分组）：EP=8，PP=9 域内**

```
域内分为 9 个 PP stage × 8 个 EP 组
每个 EP 组 8 GPU，跨域做 DP

单 GPU 权重 = 9层 × (32 experts × 42M + 360M) × 2B = 30.7 GB
（之前已算过）

问题：EP=8 意味着 all-to-all 在 8 GPU 之间
  但 256 个专家分布在 8 GPU 上，每 GPU 32 个专家
  每个 token 激活 8 个专家 → 平均每 GPU 1 个激活
  all-to-all 通信量 = batch × seq × d_model × topK × 2bytes
  
  同样的通信在 EP=72 时：
  每 GPU 3-4 个专家/层，每 token 激活 8 个 → 平均每 GPU 激活 0.11 个
  all-to-all 通信更稀疏但参与方更多
```

**最终选择方案 A 的理由：**

1. **NVIDIA Wide-EP 已验证 NVL72 全域 EP 的可行性**。NVIDIA 官方在 TensorRT-LLM 中已经实现了 EP=64 在 NVL72 上运行 DeepSeek-R1 的方案，并证明了 1.8x 吞吐提升。
2. **Hybrid-EP 通信库在 NVL72 上实测可达 NVLink 峰值带宽**（仅用 16 SMs）。
3. **EP=72 消除了域内 PP 的 pipeline bubble**——这是训练效率的重大提升。
4. **全域 EP 使得 MoE 路由最大化均匀**——72 GPU 的路由负载均衡远好于 8 GPU。

---

## 四、最终方案：最小集群配置

### 4.1 并行策略总览

```
┌─────────────────────────────────────────────────────────┐
│              ~1T MoE 模型最小训练集群方案                  │
│                                                         │
│  目标训练时间：~90 天                                     │
│  训练数据量：15T tokens                                   │
│  精度：MXFP8（MX-compatible FP8）                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  并行策略：                                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  EP = 72（Expert Parallel，NVLink 域内全域）       │  │
│  │  PP = 2 （Pipeline Parallel，跨 2 个机柜）        │  │
│  │  DP = 4 （Data Parallel，ZeRO-1，跨域）           │  │
│  │  TP = 1 （不需要 Tensor Parallel）                 │  │
│  │  CP = 1 （8K 序列不需要 Context Parallel）         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  总 GPU 数 = EP × PP × DP = 72 × 2 × 4 = 576 GPU      │
│  总机柜数 = PP × DP = 2 × 4 = 8 个 GB200 NVL72 机柜    │
│                                                         │
│  每个 NVL72 机柜是一个完整的 EP=72 域                     │
│  PP=2 意味着两个机柜组成一个 PP 流水线                    │
│  DP=4 意味着 4 条独立的 PP 流水线做数据并行                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 拓扑示意图

```
                          ┌─────────────────────────────────┐
                          │     InfiniBand/RoCE 骨干网络      │
                          │   (Quantum-2 IB 或 Spectrum-X)   │
                          └──┬───┬───┬───┬───┬───┬───┬───┬──┘
                             │   │   │   │   │   │   │   │
                      ┌──────┴───┴───┴───┴───┴───┴───┴───┴──────┐
                      │                                          │
          PP group 0 (层 0-39)              PP group 1 (层 40-79)
          ┌──────────────────┐              ┌──────────────────┐
          │                  │              │                  │
    DP=0  │  NVL72 Rack #0   │──PP link──→ │  NVL72 Rack #1   │  DP=0
          │  72 GPU, EP=72   │              │  72 GPU, EP=72   │
          │  层 0-39         │              │  层 40-79        │
          └──────────────────┘              └──────────────────┘
    DP=1  ┌──────────────────┐              ┌──────────────────┐
          │  NVL72 Rack #2   │──PP link──→ │  NVL72 Rack #3   │  DP=1
          │  72 GPU, EP=72   │              │  72 GPU, EP=72   │
          └──────────────────┘              └──────────────────┘
    DP=2  ┌──────────────────┐              ┌──────────────────┐
          │  NVL72 Rack #4   │──PP link──→ │  NVL72 Rack #5   │  DP=2
          │  72 GPU, EP=72   │              │  72 GPU, EP=72   │
          └──────────────────┘              └──────────────────┘
    DP=3  ┌──────────────────┐              ┌──────────────────┐
          │  NVL72 Rack #6   │──PP link──→ │  NVL72 Rack #7   │  DP=3
          │  72 GPU, EP=72   │              │  72 GPU, EP=72   │
          └──────────────────┘              └──────────────────┘
```

### 4.3 每 GPU 的内存使用明细

```
单 GPU 内存预算（PP=2, EP=72, DP=4）
────────────────────────────────────────────────
模型权重（BF16 master copy）：
  40 层 × (3.56 experts/GPU × 42M/expert + 5M_attn) × 2 bytes
  = 40 × (149.5M + 5M) × 2
  = 12.4 GB

优化器状态（ZeRO-1，DP=4 分片）：
  12.4 GB × 4 bytes/param ÷ 4
  = 12.4 GB

梯度缓冲区（BF16）：
  12.4 GB

激活值（选择性重计算）：
  40 层 × 每层 ~0.3 GB（EP=72 下每 GPU 处理 1/72 的 token）
  = 12 GB

EP All-to-All 通信缓冲区：
  ~10 GB

CUDA/NCCL 运行时开销：
  ~10 GB

框架 & 临时缓冲区：
  ~15 GB
────────────────────────────────────────────────
总计：~85 GB / 192 GB HBM
利用率：~44%（留有充足余量给更大 batch size）
```

有 107 GB 的余量！这意味着可以增大 micro-batch size 来提升计算效率，或者在长上下文扩展阶段（32K/128K）有足够空间。

### 4.4 通信量分析

**域内通信（NVLink，EP all-to-all）：**

```
每层 MoE 的 EP all-to-all 通信：
────────────────────────────────
dispatch 阶段：
  每 GPU 将自己处理的 token 发送给拥有对应专家的 GPU
  通信量 = micro_batch × seq_len × d_model × topK × dtype_size / EP
  
  假设全局 micro_batch = 72（每 GPU 1 个 sample）
  seq_len = 8192, d_model = 8192, topK = 8, BF16 = 2 bytes
  
  每 GPU dispatch 发送量 ≈ 1 × 8192 × 8192 × 8 × 2 / 72
  = ~15 MB / GPU / 层

combine 阶段：类似量级

总计每层 EP 通信：~30 MB / GPU
40 层 × 30 MB = 1.2 GB / GPU / micro-batch

在 1.8 TB/s NVLink 上：1.2 GB / 1.8 TB/s = 0.67 ms
─────────────────────────────────
结论：域内 EP 通信完全可以被计算掩盖（DualPipe 风格）
```

**跨域通信：**

```
1. PP 点对点（前向/反向激活传递）：
   每 micro-step：micro_batch × seq_len × d_model × 2 bytes
   = 1 × 8192 × 8192 × 2 = 128 MB
   但 EP=72 下每 GPU 只传自己负责的 token
   实际每 GPU：128 MB / 72 ≈ 1.8 MB
   在 400Gbps IB 上：1.8 MB / 50 GB/s = 0.036 ms → 几乎为零

2. DP 梯度同步（ZeRO-1 all-reduce，跨 DP=4）：
   每 GPU 的模型参数 ≈ 12.4 GB / 2 × 2 bytes = 12.4 GB 梯度
   All-reduce across 4 members：
   ring all-reduce 数据量 ≈ 2 × 12.4 GB × (4-1)/4 = 18.6 GB
   
   但这可以与 backward 计算 overlap！
   在 400Gbps IB 上：18.6 GB / 50 GB/s = 0.37 s
   一个完整 iteration 的计算时间远大于此，overlap 可行
   
3. MoE 梯度特殊性：
   只有被激活的专家有非零梯度
   每层 256 个专家中只有 8 个被当前 token 激活
   但 batch 足够大时，几乎所有专家都会被激活（至少一次）
   因此梯度是满的，但可以使用梯度压缩 / FP8 梯度通信
```

### 4.5 Pipeline Bubble 分析

```
DualPipe 调度（类 DeepSeek-V3）：
────────────────────────
PP = 2 stages
micro-batches per step (m) = 设为 24（可调）

经典 1F1B bubble 比例 = (PP - 1) / (PP - 1 + m) = 1/25 = 4%

DualPipe 可以进一步将 bubble 降到接近 0%
（通过双向流水线 + 计算-通信重叠）

PP=2 的 bubble 开销几乎可以忽略
```

### 4.6 训练吞吐估算

```
单 GPU 有效算力：~900 TFLOPS (MXFP8, Hybrid-EP)
576 GPU 总有效算力：518.4 PFLOPS

训练效率折损：
  PP bubble：~4% → ×0.96
  DP 通信 overlap 残余：~3% → ×0.97
  其他开销（load balancing, routing）：~5% → ×0.95

有效吞吐：518.4 × 0.96 × 0.97 × 0.95 = 457 PFLOPS

训练时间 = 4.5 × 10²⁴ / (457 × 10¹⁵)
        = 9.85 × 10⁶ s
        = 114 天 ≈ 3.8 个月
```

如果 3.8 个月可接受，8 个机柜（576 GPU）就是最小配置。如果要压缩到 3 个月以内，把 DP 增加到 6：

```
替代方案：DP = 6
总机柜 = 2 × 6 = 12 个 NVL72
总 GPU = 864
训练时间 ≈ 114 × (576/864) = 76 天 ≈ 2.5 个月
```

---

## 五、Scale-Out 网络组网方案

### 5.1 网络拓扑

```
8 个 NVL72 机柜的 scale-out 网络：
────────────────────────────────────────────

每机柜 72 个 ConnectX-7 NIC × 400Gbps = 28.8 Tbps / 机柜
8 机柜总 NIC 数 = 576 个
总 scale-out 端口带宽 = 576 × 400Gbps = 230.4 Tbps

推荐网络拓扑：Rail-Optimized Fat-Tree

              ┌─────────────────────────┐
              │  Spine Layer             │
              │  Quantum-2 QM9700       │
              │  (400G InfiniBand)       │
              │  若干台交换机             │
              └────┬────┬────┬────┬─────┘
                   │    │    │    │
              ┌────┴────┴────┴────┴─────┐
              │  Leaf Layer              │
              │  每机柜 2-3 台叶交换机    │
              │  每台 64 端口 × 400G     │
              └─────────────────────────┘

Rail-Optimized 设计：
  - 每个 NVL72 机柜内的 GPU 按 "rail" 编号
  - GPU[i] 在所有机柜中连接到同一组叶交换机
  - 这保证 DP all-reduce 中的对等 GPU 有最短路径
  
  Rail 0: Rack0-GPU0, Rack1-GPU0, ..., Rack7-GPU0 → Leaf Switch 0
  Rail 1: Rack0-GPU1, Rack1-GPU1, ..., Rack7-GPU1 → Leaf Switch 1
  ...
  Rail 71: Rack0-GPU71, ..., Rack7-GPU71 → Leaf Switch 71
  
  需要 72 组 leaf switch（可合并为更少的物理交换机）
  每组 leaf switch 连接 8 个 NIC (来自 8 个机柜)
  DP all-reduce 在每组 leaf switch 内部完成，不需要过 spine
```

实际上，由于 DP=4（不是 8），PP=2 的通信模式是点对点：

```
实际通信分组：
────────────────
PP 通信（stage 0 → stage 1）：
  Rack0 ↔ Rack1（PP group 0）
  Rack2 ↔ Rack3（PP group 1）
  Rack4 ↔ Rack5（PP group 2）
  Rack6 ↔ Rack7（PP group 3）

DP 通信（all-reduce across 4 PP groups）：
  {Rack0, Rack2, Rack4, Rack6}（PP stage 0 的 4 个 DP 副本）
  {Rack1, Rack3, Rack5, Rack7}（PP stage 1 的 4 个 DP 副本）
```

### 5.2 IB 交换机需求

```
Quantum-2 QM9700：64 端口 × 400Gbps NDR

Leaf 层：
  每个 NVL72 机柜需要 72 个 400G 上行端口
  72 / 64 = 1.125 → 每机柜至少 2 台 QM9700 作为 leaf
  8 机柜 × 2 台 = 16 台 leaf 交换机
  每台 leaf：~36 端口接本机柜 NIC，~28 端口上行到 spine

Spine 层：
  16 台 leaf × 28 上行 = 448 端口需要 spine 承接
  spine 交换机数 = 448 / 64 ≈ 7 台 QM9700

总计：~23 台 Quantum-2 QM9700 InfiniBand 交换机

如果使用 NVIDIA Quantum-2 的 SHARP in-network compute，
可以在交换机内完成部分 all-reduce，进一步降低 DP 通信延迟
```

---

## 六、完整方案总结

```
┌══════════════════════════════════════════════════════════════════════┐
║            ~1T MoE 模型最小训练集群 —— 完整方案                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  ▎模型规格                                                          ║
║    总参数：~1T    激活参数：~50B    专家数：256+1 shared             ║
║    层数：80      隐藏维度：8192    序列长度：8K                      ║
║    精度：MXFP8 混合精度                                             ║
║                                                                    ║
║  ▎硬件配置                                                          ║
║    计算：8 × GB200 NVL72 机柜                                       ║
║          = 576 GPU (B200)                                          ║
║          = 288 Grace CPU                                           ║
║          = 576 × 192 GB = 110 TB HBM3e                            ║
║          = 576 × 400Gbps = 230 Tbps scale-out                     ║
║    网络：~23 台 Quantum-2 QM9700 400G IB 交换机                    ║
║          Rail-optimized fat-tree 拓扑                               ║
║    功耗：8 × 120 kW = ~960 kW + 网络/冷却 ≈ 1.2 MW                ║
║                                                                    ║
║  ▎并行策略                                                          ║
║    ┌────────────────────────────────────────────────────────┐      ║
║    │  Expert Parallel (EP) = 72    ← NVLink 域内全域        │      ║
║    │  Pipeline Parallel (PP) = 2   ← 跨 2 个机柜           │      ║
║    │  Data Parallel (DP) = 4       ← 跨 4 个 PP 流水线     │      ║
║    │  Tensor Parallel (TP) = 1     ← 不需要                │      ║
║    │  Context Parallel (CP) = 1    ← 8K 序列不需要          │      ║
║    │                                                        │      ║
║    │  GPU 总数 = 72 × 2 × 4 = 576 ✓                       │      ║
║    └────────────────────────────────────────────────────────┘      ║
║                                                                    ║
║  ▎通信方案                                                          ║
║    域内（NVLink 5, 130 TB/s）：                                     ║
║      EP all-to-all dispatch/combine（Hybrid-EP 库）                 ║
║      DualPipe 计算-通信重叠 → 通信几乎完全隐藏                       ║
║    跨域（IB 400G, 50 GB/s/GPU）：                                   ║
║      PP 点对点：激活值传递，~1.8 MB/GPU/step → 延迟 <0.1ms         ║
║      DP all-reduce：梯度同步 ~18.6 GB/GPU，与 backward overlap      ║
║      SHARP in-network reduce 加速 all-reduce                       ║
║                                                                    ║
║  ▎关键软件栈                                                        ║
║    框架：NVIDIA Megatron-Core                                       ║
║    EP 通信：Hybrid-EP (NVLink TMA + IBGDA)                         ║
║    调度器：DualPipe pipeline scheduler                              ║
║    精度：MXFP8 + BF16 master weights                               ║
║    优化器：ZeRO-1 distributed Adam                                  ║
║    负载均衡：Online EPLB (Expert Parallel Load Balancer)            ║
║                                                                    ║
║  ▎预计性能                                                          ║
║    单 GPU 实际吞吐：~900 TFLOPS (MXFP8)                            ║
║    集群有效算力：~457 PFLOPS                                        ║
║    训练数据：15T tokens                                             ║
║    训练总 FLOPs：~4.5 ZettaFLOP                                    ║
║    预计训练时间：~114 天（3.8 个月）                                 ║
║                                                                    ║
║  ▎成本估算                                                          ║
║    GPU 小时：576 GPU × 114 天 × 24 小时 = 1,576K GPU-hours         ║
║    按 $3/GPU-hour 租赁价：~$4.7M                                   ║
║    硬件购置（8 NVL72 + 网络）：~$25-30M                             ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 七、为什么这个方案是"最小"的？

让我们验证为什么不能再减少机柜数：

**能否用 4 个机柜（288 GPU）？**

```
4 机柜方案：PP=2, DP=2
训练时间：114 × (576/288) = 228 天 ≈ 7.6 个月 → 太慢
优化器内存：每 GPU 分到 DP=2 → 24.8 GB 优化器 → 可以装下
结论：技术上可行，但训练时间不可接受
```

**能否用 2 个机柜（144 GPU）？**

```
2 机柜方案：PP=2, DP=1
训练时间：228 × 2 = 456 天 ≈ 15 个月 → 完全不可接受
优化器内存：DP=1，每 GPU 49.6 GB 优化器 → 加上其他约 110 GB，可以装下
结论：内存可行但时间完全不可接受
```

**能否只用 1 个机柜（72 GPU）？**

```
1 机柜方案：EP=72, PP=1, DP=1
优化器内存：每 GPU 需存 1/1 的优化器状态
  单 GPU 参数 = ~13.9B → 优化器 = 55.6 GB
  总内存 = 26.9(权重) + 55.6(优化器) + 26.9(梯度) + 60(激活) + 20(缓冲)
         = 189 GB → 几乎等于 192 GB HBM，没有余量！
训练时间：114 × 8 = 912 天 ≈ 2.5 年 → 完全不可接受
结论：内存极限，时间不可接受
```

因此，**8 个 GB200 NVL72 机柜（576 GPU）是在 3-4 个月合理时间窗口内训练 ~1T MoE 模型的最小配置**。如果时间允许放宽到 6 个月，4 个机柜（288 GPU）在技术上也是可行的最小配置。

---

## 八、架构精妙之处的回顾

回到你开篇描述的三阶段架构演进，这个方案完美体现了 GB200 NVL72 的设计哲学：

```
通信类型         │ 走什么路径              │ 延迟    │ 带宽
────────────────┼────────────────────────┼────────┼──────────
EP all-to-all   │ NVLink 5（域内）        │ ~1 μs  │ 130 TB/s 聚合
PP 激活传递     │ C2C→LPDDR5X→PCIe→IB    │ ~10 μs │ 50 GB/s/GPU
DP 梯度同步     │ C2C→LPDDR5X→PCIe→IB    │ ~10 μs │ 50 GB/s/GPU
                │ + SHARP in-network     │        │
```

**EP（最重通信）完全在 NVLink 域内**——这正是 NVL72 "72 GPU 大域"设计的根本价值。DeepSeek-V3 在 H800 上被迫让 EP all-to-all 走 IB 网络（因为 DGX H100 只有 8 GPU 在 NVSwitch 内），通信开销占比超过 50%。在 GB200 NVL72 上，Hybrid-EP 实现了近零通信开销的 EP，这是 14% 训练加速的核心来源。

**PP 和 DP（轻通信）走 scale-out 网络**——PP 点对点传递量极小（MB 级），DP 梯度同步可以完全 overlap，scale-out 的 PCIe + IB 瓶颈被巧妙规避。

这就是 NVIDIA"把 NVLink 域做大、让重通信不出域"策略的最佳实践。
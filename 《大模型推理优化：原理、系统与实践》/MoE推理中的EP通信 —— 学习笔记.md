
## 一、基本概念：EP（Expert Parallelism）是什么

MoE（Mixture of Experts）模型中，专家数量庞大（如DeepSeek-V3有256个路由专家），单张GPU根本装不下所有专家的权重。EP的做法是把专家分散到多张GPU上，每张卡只持有一部分专家。

```
以 DeepSeek-V3、8张GPU为例：

GPU 0: 专家 0-31
GPU 1: 专家 32-63
GPU 2: 专家 64-95
...
GPU 7: 专家 224-255
```

这就引出了一个核心问题：token被路由到的专家不在本地GPU上，就需要通信。这份笔记就是围绕这个通信展开的。

---

## 二、Router（Gate网络）在哪里计算？

Router本质上是一个很小的线性层：

```
gate_score = softmax(hidden_state × W_gate)

W_gate 形状: [hidden_dim, num_experts] = [7168, 256] ≈ 7MB (FP16)
```

因为参数极小，**每张卡上都持有一份完整的Router权重副本**，各自在本地独立计算路由。不需要集中到某张卡上做，否则反而引入不必要的通信瓶颈。

---

## 三、标准EP通信流程

教科书式的EP流程，每生成一个token要经历：

```
Step 1: 每张卡本地算 Router → 得知自己的token要去哪些专家（哪些GPU）
Step 2: All-to-All Dispatch  → 把token的hidden_state发送到目标专家所在GPU
Step 3: 各GPU用本地专家计算
Step 4: All-to-All Combine   → 把专家输出发回token来源GPU
```

All-to-All是一种"精确投递"的集合通信：每张卡只把数据发给需要它的那张卡，每张卡也只接收自己需要的数据。

---

## 四、Decode阶段的关键洞察：可以绕过All-to-All

### 4.1 为什么可以绕过？

Decode阶段每步只处理很少的token（比如bs=64就是64个token）。与此同时，Attention层通常采用TP（张量并行），TP的AllReduce结束后，**每张卡上天然已经持有全部64个token的完整hidden_state**。

```
TP AllReduce后的状态：

GPU 0: 拥有 token 0-63 的完整 hidden_state  ✅
GPU 1: 拥有 token 0-63 的完整 hidden_state  ✅
...
GPU 7: 拥有 token 0-63 的完整 hidden_state  ✅
```

既然每张卡已经有了全部token，那进入MoE时，每张卡可以直接：

```
1. 本地算 gate → 知道全部64个token各自去哪些专家
2. 从中挑出路由到本地专家的那些token
3. 本地专家计算
4. 最后做一次 AllReduce 合并各卡的专家输出
```

**全程不需要All-to-All，也不需要AllGather。** 通信模式变成了简单的AllReduce。

### 4.2 数据量验证

```
bs=64, hidden_dim=7168, FP16

64个token的hidden_state: 64 × 7168 × 2B = 896KB ≈ 不到1MB

每张卡都存一份完整副本：毫无显存压力
AllReduce通信量：< 1MB，延迟极低
```

### 4.3 对比两种方案

```
标准 All-to-All 方案:
  → 每张卡只发目标卡需要的token（精确投递）
  → 通信模式不规则，调度开销相对较大

TP天然全量 + 本地挑选方案:
  → 不需要额外通信来分发token
  → 每张卡本地挑选，最后AllReduce聚合
  → 通信模式规则，硬件优化好，延迟更低
```

Decode阶段数据量极小，All-to-All的"精确投递"省下的那点带宽毫无意义，反而是其不规则的调度开销成了负担。

---

## 五、Prefill阶段：为什么必须用All-to-All

### 5.1 数据量爆炸

```
Prefill: bs=64, seq_len=4096 → 共 262144 个token

262144 × 7168 × 2B ≈ 3.5GB
```

### 5.2 Attention层并行策略变了

Prefill阶段token数量巨大，纯TP的AllReduce通信量已经很重（3.5GB），且显存压力大。因此Prefill阶段通常引入SP（Sequence Parallelism）或按batch/序列维度切分。

```
SP切分后：

GPU 0: 只持有 token 0-32767 的 hidden_state     (1/8)
GPU 1: 只持有 token 32768-65535 的 hidden_state  (1/8)
...
```

每张卡不再持有全部token，"本地挑选"方案的前提就不成立了。

### 5.3 就算强行让每张卡持有全量呢？

```
每张卡存全部 262144 个token的hidden_state → 3.5GB × 8张卡
→ 显存浪费严重
→ 前面Attention的AllReduce本身就是3.5GB，通信开销巨大

代价远超All-to-All方案
```

所以Prefill阶段必须使用All-to-All精确投递，让每个token只发送到需要它的GPU。

---

## 六、Decode vs Prefill 对比总结

```
                      Decode                Prefill
─────────────────────────────────────────────────────────
token数量             少 (~64)              多 (~262144)
每卡持有全量hidden    ✅ 天然持有(TP)        ❌ 需切分(SP)
全量hidden显存开销    < 1MB，忽略不计        ~3.5GB，不可忽视
MoE通信策略           本地挑选 + AllReduce   All-to-All
通信复杂度            低，规则              高，不规则
```

---

## 七、工程启示

正是因为Decode和Prefill的通信特性差异巨大，现代MoE推理系统普遍采用**Prefill-Decode分离架构（Disaggregation）**：把Prefill和Decode拆到不同的机器集群上，各自采用最适合的并行策略和通信模式。Prefill集群侧重大吞吐、用All-to-All；Decode集群侧重低延迟、用更简单的通信模式。这也是DeepEP中区分"正常模式"与"低延迟模式"的底层原因。
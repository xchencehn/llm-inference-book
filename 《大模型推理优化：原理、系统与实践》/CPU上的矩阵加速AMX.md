

# AMX 到底是不是芯片上真实的物理结构？

## 一句话回答：是的，100% 是真实存在的物理晶体管电路

AMX 不是什么"软件模拟"或"虚拟概念"，它是实打实地刻在硅片上、由数十亿晶体管中的一部分构成的**专用硬件模块**。Intel 自己的官方说法是：

> “Intel AMX is a **dedicated hardware block** found on the Intel Xeon Scalable processor core.”

下面我从硬件到软件，一层一层地给你拆开讲。

---

## 1. 芯片上到底长什么样？

### 核心内部的结构

每一个 Sapphire Rapids 的 P-core（基于 Golden Cove 微架构）内部，大致可以分为这几大物理区域：前端（取指/解码）、调度器/重排序缓冲区、执行单元（ALU、FPU/FMA、Load/Store），以及缓存（L1/L2）。AMX 的 TMUL（Tile Matrix Multiply Unit）就是**紧挨着 FMA 单元旁边的一块独立的硬件区域**。

具体来说，TMUL 包含两个物理组成部分：

**第一部分：Tile 寄存器文件（TILEDATA）。** 这是 8 个二维矩阵寄存器，每个最大 16 行 × 64 字节 = 1 KiB，整个寄存器文件总共 8 KiB。这不是复用普通的通用寄存器或 AVX-512 的 ZMM 寄存器——这是一组全新的、物理上独立的 SRAM 存储单元。你可以把它想象成芯片核心里多出了一小块 8 KiB 的高速存储"格子"。

**第二部分：TMUL 运算阵列。** 这是一个由**融合乘加（FMA）单元组成的网格（grid）**，本质上类似于 Google TPU 的脉动阵列（systolic array）或 NVIDIA 的 Tensor Core。它的工作就是接收两个 tile 寄存器的数据，在一条指令内完成整个子矩阵的乘累加运算：C[M][N] += A[M][K] × B[K][N]。

在 Sapphire Rapids 上，这个阵列每个时钟周期可以完成 **2048 次 INT8 操作** 或 **1024 次 BF16 操作**（一条 TMUL 指令需要 16 个周期完成，但可以流水线化）。以 BF16 为例，单核理论峰值达到约 **3482 GFLOPS**——这是同一颗核心上 AVX-512 的 **16 倍**。

### 占多大面积？

Intel 没有公开 TMUL 单元的精确面积数据，但从架构分析来看，它不算很大。现代 CPU 核心中，缓存（L1/L2）才是面积大户。TMUL 的 8 KiB 寄存器文件加上运算阵列，大概占每个核心面积的很小一部分。但即使面积不大，它提供的矩阵乘法吞吐量却是传统 FMA 单元的一个数量级以上——这就是专用硬件的威力：你用晶体管做专一的事情，效率极高。

---

## 2. 它跟普通执行单元的关系

可以这样理解 CPU 核心里的执行单元层次：

**通用 ALU**（加减法、逻辑运算、移位等）→ 什么都能算，但每次只处理一个标量值。

**SIMD/FMA 单元**（AVX-512 等）→ 一次处理一个 512-bit 向量（比如 16 个 FP32），相当于一条流水线。在 Sapphire Rapids 上有两个 512-bit FMA 单元（分布在 port 0/1 和 port 5）。

**AMX TMUL 单元** → 一次处理整个二维子矩阵。它不是"更宽的 SIMD"，而是一种全新的计算范式。它有自己的寄存器文件（tile registers），自己的数据通路，自己的运算阵列。你可以把它理解为 CPU 核心里嵌入的一个"微型矩阵协处理器"。

Bob Valentine（AMX 的主要架构师）在接受采访时的说法是：

> “The best way to think of AMX is that it’s a **matrix math overlay** for the AVX-512 vector math units. We can think of it like a **TensorCore type unit for the CPU**.”

这里"overlay"的意思是，AMX 指令会跟普通 x86 指令同步执行，共享同一个指令流和内存一致性模型，但运算本身是在专用硬件上完成的。

---

## 3. 支持 AMX 的芯片有哪些

|代号|发布时间|AMX 子集支持|
|---|---|---|
|**Sapphire Rapids**（第 4 代 Xeon Scalable）|2023.1|AMX-TILE, AMX-INT8, AMX-BF16|
|**Emerald Rapids**（第 5 代 Xeon Scalable）|2023.12|同上|
|**Granite Rapids**（Xeon 6, P-core）|2024.9|+AMX-FP16|
|**Granite Rapids-D**（Xeon 6 D）|2024+|+AMX-COMPLEX（支持 FP16 复数矩阵）|
|**Diamond Rapids**（未来）|预计 2025+|+AMX-FP8, AMX-TF32, AMX-TRANSPOSE 等|

注意：只有 **P-core**（性能核）有 AMX。E-core（能效核）没有。消费级 Alder Lake/Raptor Lake 虽然也用了 Golden Cove 微架构，但**没有 AMX**——Intel 只在服务器/工作站 Xeon 上启用了它。

---

## 4. 为什么我们编程"没感觉"？

这是你提的最好的问题。答案涉及多个层面：

### 层面一：抽象层太厚了

现代软件的层次结构大致是：

```
你的 Python/Java/Go 代码
    ↓
框架/库 (PyTorch, TensorFlow, NumPy, OpenBLAS, oneDNN...)
    ↓
编译器 (GCC, LLVM/Clang, JIT)
    ↓
操作系统内核 (Linux 5.16+ 才支持 AMX 的上下文切换)
    ↓
CPU 微架构 (解码器 → 调度器 → 执行单元/TMUL)
    ↓
晶体管
```

大多数程序员工作在最上面一两层。你写 `torch.matmul(A, B)` 的时候，PyTorch 会调用 oneDNN 或 MKL，而 oneDNN 内部有针对 AMX 优化的代码路径——它会先检测 CPU 是否支持 `CPUID.AMX-TILE`，如果支持就走 AMX 路径。**整个过程对你完全透明**。

### 层面二：AMX 指令本身非常低层且非常特殊

AMX 总共只有约 **12 条指令**。使用流程极其"仪式感"：

1. 先调用 `syscall(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` 向操作系统申请权限
2. 用 `ldtilecfg` 配置 tile 的行数和列宽
3. 用 `tileloadd` 把数据从内存加载到 tile 寄存器（还需要手动做矩阵 B 的特殊内存布局重排）
4. 用 `tdpbf16ps` 或 `tdpbusd` 等指令做矩阵乘累加
5. 用 `tilestored` 把结果写回内存
6. 用 `tilerelease` 释放 tile 状态

这跟你平时写的 `a + b` 完全是两个世界。99.9% 的程序员永远不需要、也不应该直接碰这些指令。

### 层面三：并不是所有计算都需要矩阵乘法

AMX 只做一件事：**矩阵乘累加**（Matrix Multiply-Accumulate）。它不能做加法、不能做条件跳转、不能做字符串处理、不能做任何通用计算。对于绝大多数应用程序（Web 服务、数据库、文件处理、GUI 程序），程序的瓶颈从来不是矩阵乘法，而是分支预测、缓存命中率、内存带宽、I/O 延迟等等。

AMX 的唯一甜蜜点是：**深度学习的推理和训练**——因为神经网络的核心运算就是巨量的矩阵乘法。

### 层面四：需要操作系统配合

Linux 内核直到 **5.16 版本**才加入 AMX 支持。因为 AMX 的 tile 寄存器文件高达 8 KiB，在线程上下文切换时需要保存/恢复这些状态。这比保存普通的 64-bit 通用寄存器（几百字节）开销大得多。操作系统必须显式分配额外的 XSAVE 区域来处理这件事。如果内核不支持，即使硬件有 AMX，你也用不了。

---

## 5. 一个类比帮你理解

想象一下你家厨房（CPU 核心）：

**通用 ALU** = 菜刀。什么都能切，但一次只切一刀。

**AVX-512 FMA** = 切蛋器。一刀下去把鸡蛋切成 16 片（16 个 FP32 并行）。但要切一整盘鸡蛋，你得一个一个放进去。

**AMX TMUL** = 厨房里专门嵌入了一台**全自动饺子机**。你把面和馅倒进去，它一口气给你批量出几百个饺子。但它只能包饺子——你想用它炒菜？不行。

大多数人在厨房里炒菜、煮汤、烤面包，根本不需要这台饺子机。但如果你是开饺子馆的（跑大模型推理），这台机器就是命脉。

而你之所以"没感觉"，是因为你在用美团外卖下单（PyTorch），后厨（oneDNN/MKL）自动帮你调度了这台饺子机，你连厨房门都不用进。

---

## 6. KTransformers 如何利用它

回到你最开始提的 KTransformers 场景：MoE（Mixture of Experts）模型中，大部分专家参数是稀疏激活的。把这些不常用的专家权重放在 CPU 内存中，用 AMX 来做它们的矩阵乘法，而把 Attention 等热点计算放在 GPU 上——这就是 CPU-GPU 异构协同的精髓。

AMX 在这里的价值在于：单核 BF16 矩阵乘法峰值约 3.5 TFLOPS，一颗 56 核的 Xeon 理论峰值可达约 **195 TFLOPS BF16**。虽然远不及 H100 GPU 的千 TFLOPS 级别，但对于那些"偶尔被激活的专家层"来说，已经足够了——而且 CPU 内存（DDR5 可达数百 GB 甚至 TB 级）比 GPU 显存（HBM 通常 80-192 GB）便宜得多、容量大得多。

这就是 AMX 存在的真正意义：**不是替代 GPU，而是让 CPU 在矩阵计算上不再是彻底的配角**。
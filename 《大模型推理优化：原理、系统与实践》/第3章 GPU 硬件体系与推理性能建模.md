
# 第3章 GPU 硬件体系与推理性能建模



大模型推理的每一次优化，最终都要落实到具体的硬件上执行。一个推理工程师如果不理解GPU的计算架构、内存层次和互联拓扑，就如同一个赛车手不了解自己的发动机——他或许能跑，但永远无法跑到极限。

本章将从GPU的微架构出发，逐层剖析影响推理性能的硬件要素。我们首先深入GPU内部，理解流式多处理器（SM）、线程束（Warp）和张量核心（Tensor Core）的工作原理；然后分析GPU多层内存体系的带宽与容量特征；接着建立内存带宽与计算吞吐之间的平衡关系模型；再扩展到多GPU互联拓扑，理解分布式推理的通信基础。由于本书涵盖的KTransformers引擎采用CPU-GPU异构计算范式，我们还将系统介绍CPU端的高性能计算能力，特别是AVX-512和AMX指令集在大模型推理中的角色。最后，我们简要介绍TPU、Groq LPU、Cerebras WSE等新兴推理硬件，为读者提供更广阔的技术视野。

---

## 3.1 GPU 计算架构：SM、Warp、Tensor Core

### 3.1.1 从SIMT到大模型推理：GPU并行计算的基本范式

GPU之所以能在大模型推理中占据核心地位，根本原因在于其大规模并行计算的架构设计。与CPU追求低延迟、复杂控制流的设计哲学不同，GPU追求的是高吞吐、大规模数据并行。理解这种差异，是理解所有推理优化的起点。

NVIDIA GPU采用**单指令多线程**（Single Instruction, Multiple Threads, SIMT）执行模型。在这一模型下，成千上万个轻量级线程同时执行相同的指令序列，但各自操作不同的数据。这与大模型推理的计算特征天然契合：无论是矩阵乘法中对不同行列的并行计算，还是注意力机制中对不同头（head）的并行处理，都可以自然地映射为大量并行线程。

### 3.1.2 流式多处理器（Streaming Multiprocessor, SM）

SM是GPU的基本计算单元，每个GPU芯片由数十到上百个SM组成。以NVIDIA近几代数据中心级GPU为例，A100（Ampere架构）拥有108个SM，H100（Hopper架构）拥有132个SM，而B200（Blackwell架构）进一步增加到160个以上。

每个SM内部包含以下关键组件：

**CUDA核心（FP32/FP16/INT8计算单元）。** 这是最基本的浮点和整数运算单元。每个SM通常包含数十到上百个CUDA核心，分布在多个处理块（processing block）中。以H100为例，每个SM包含128个FP32 CUDA核心，分布在4个处理块中，每个处理块拥有32个FP32核心。

**Tensor Core（张量核心）。** 这是NVIDIA从Volta架构（2017年）开始引入的专用矩阵运算单元，也是当前大模型推理中最重要的计算引擎。Tensor Core的核心能力是在单个时钟周期内完成一个小矩阵的乘累加（Matrix Multiply-Accumulate, MMA）运算。我们将在本节后续详细讨论Tensor Core。

**Warp调度器。** 每个SM配备多个Warp调度器（通常为4个），负责在每个时钟周期选择就绪的Warp并发射指令。多个Warp调度器使得SM能够在一个周期内同时调度来自不同Warp的指令，从而隐藏访存延迟。

**寄存器文件（Register File）。** 每个SM拥有一个大容量的寄存器文件，供所有活跃线程使用。以H100为例，每个SM拥有256 KB的寄存器文件。寄存器是GPU中速度最快的存储层级，单周期即可访问。

**共享内存与L1 Cache。** 每个SM拥有一块可配置的片上存储，可在共享内存（Shared Memory）和L1 Data Cache之间灵活划分。以H100为例，这块片上存储的总容量为228 KB。共享内存是SM内所有线程可直接寻址的快速暂存区，是实现高效Tiling算法（如FlashAttention）的关键资源。

**特殊功能单元（SFU）。** 用于执行超越函数，如正弦、余弦、指数、倒数等。在推理过程中，Softmax中的指数运算、SiLU/GELU激活函数等都依赖SFU。

从推理优化的视角看，SM的资源配置决定了若干关键的性能上限。SM数量直接决定了GPU的总并行度：更多的SM意味着可以同时执行更多的线程块（Thread Block），从而支持更大的批处理和更高的计算吞吐。每个SM的寄存器和共享内存容量则限制了每个线程块的规模和每个SM上能同时驻留的线程块数量——这一指标被称为**占用率**（Occupancy），它直接影响GPU隐藏访存延迟的能力。

### 3.1.3 线程束（Warp）：GPU执行的基本粒度

Warp是NVIDIA GPU中实际调度和执行的最小单位。一个Warp由32个线程组成，这32个线程在同一时刻执行同一条指令（SIMT）。理解Warp对于推理性能分析至关重要，因为它决定了GPU计算的效率边界。

**Warp发散（Warp Divergence）。** 当一个Warp内的32个线程遇到条件分支（if-else），且不同线程需要走不同的分支时，GPU必须串行执行两个分支路径——先让需要执行if分支的线程执行，其余线程空等；然后反过来执行else分支。这被称为Warp发散，会导致执行效率降低。在推理场景中，MoE模型的专家路由就可能引发Warp发散：同一Warp中的不同Token可能被路由到不同的专家，导致计算效率下降。这也是为什么vLLM和SGLang在处理MoE模型时需要对Token进行分组排序（grouping/sorting），使得路由到同一专家的Token尽量分配到同一Warp内。

**Warp级原语（Warp-Level Primitives）。** 现代CUDA编程提供了丰富的Warp级协作原语，如`__shfl_sync`（Warp内数据交换）、`__ballot_sync`（Warp级投票）、`__reduce_add_sync`（Warp级归约）等。这些原语允许同一Warp内的线程无需通过共享内存即可直接交换数据，延迟极低。在高性能推理Kernel中，这些原语被广泛用于实现高效的归约操作（如Softmax中的行最大值和行求和）以及数据重排。

**Warp调度与延迟隐藏。** GPU通过在多个Warp之间快速切换来隐藏访存延迟。当一个Warp因为等待内存读取而暂停时，Warp调度器会立即切换到另一个就绪的Warp继续执行。这种**延迟隐藏**（Latency Hiding）机制是GPU实现高吞吐的核心手段。为了有效隐藏延迟，需要每个SM上有足够多的活跃Warp，这就是为什么Occupancy对于访存密集型Kernel（如Decode阶段的Attention Kernel）尤为重要。

### 3.1.4 Tensor Core：大模型推理的计算引擎

Tensor Core是当前GPU上执行大模型推理计算的核心硬件单元。从Volta（第一代）到Hopper（第四代）再到Blackwell（第五代），Tensor Core经历了持续的演进，每一代都带来了对推理更友好的特性。

**基本工作原理。** Tensor Core执行的是矩阵乘累加（MMA）操作：D = A × B + C，其中A、B、C、D都是小矩阵。Tensor Core在单个时钟周期内完成这一操作，远快于用CUDA核心逐元素计算的方式。

**各代Tensor Core的关键参数。** 下面我们以数据中心GPU为主线，梳理各代Tensor Core对推理的影响：

Volta/Turing（第一/二代）Tensor Core支持FP16 × FP16 → FP32的混合精度MMA，操作粒度为4×4×4矩阵。这一代Tensor Core已经足以支撑FP16推理，但尚不支持更低精度的计算。

Ampere（第三代，A100）Tensor Core将操作粒度扩展到支持更灵活的矩阵尺寸，并首次引入了对BF16、TF32和INT8数据类型的原生支持。对推理而言最重要的是INT8支持，因为它直接加速了INT8量化推理。A100的INT8 Tensor Core算力达到624 TOPS（每秒万亿次整数运算），是其FP16算力（312 TFLOPS）的两倍。

Hopper（第四代，H100）Tensor Core引入了两项革命性特性。其一是原生FP8支持，包括E4M3和E5M2两种格式，FP8算力高达约1979 TFLOPS，相比FP16再翻倍。FP8量化在保持接近FP16精度的同时，将计算吞吐翻倍并减半显存占用，已成为当前生产环境推理的主流选择。其二是**Transformer Engine**（TE），这是NVIDIA在硬件层面对Transformer推理的直接优化——TE内置了动态量化逻辑，能在运行时自动将FP16/BF16数据转换为FP8进行计算，并管理精度缩放因子，从而使得用户几乎无需修改代码即可获得FP8加速。vLLM和SGLang都已集成了Transformer Engine的支持。

Blackwell（第五代，B200/GB200）进一步引入了FP4支持，理论算力再次翻倍，同时支持更大的MMA操作粒度。此外，Blackwell引入的第二代Transformer Engine支持了基于微缩放（Microscaling, MX）格式的FP4/FP6量化，为极低比特推理提供了硬件级支持。

**Tensor Core的使用条件。** Tensor Core并非在所有情况下都能被充分利用。为了高效使用Tensor Core，矩阵维度需要是特定数值（通常是8或16）的倍数，数据需要以特定的内存布局排列。在推理的Decode阶段，当Batch Size为1时，矩阵乘法退化为矩阵-向量乘法（GEMV），矩阵的一个维度为1，此时Tensor Core的利用率会大幅下降——这正是Decode阶段成为访存瓶颈而非计算瓶颈的硬件根因之一。增大Batch Size（如通过连续批处理汇聚更多Decode请求）可以将GEMV"攒"回GEMM，从而重新发挥Tensor Core的算力优势。

**WMMA与MMA指令。** 在CUDA编程层面，Tensor Core通过两种接口暴露给程序员。WMMA（Warp Matrix Multiply-Accumulate）是Warp级别的API，一个Warp的32个线程协作完成一次矩阵乘累加操作，编程相对简洁。MMA（PTX层级的矩阵指令）则提供了更细粒度的控制，允许程序员直接指定寄存器到Tensor Core的数据映射，能够榨取更高的性能，但编程复杂度也显著增加。高性能推理Kernel（如FlashAttention、Marlin量化Kernel）通常使用PTX级别的MMA指令来获得极致性能。

### 3.1.5 GPU计算资源与推理工作负载的映射

理解了SM、Warp和Tensor Core的硬件特性后，我们来分析推理的两个阶段如何映射到这些硬件资源上。

**Prefill阶段。** Prefill处理的是包含大量Token的输入序列，主要计算是大规模矩阵乘法（GEMM）。这些GEMM操作的矩阵维度较大（M维度等于输入Token数，可达数千），能够充分利用所有SM上的Tensor Core，实现接近峰值的计算吞吐。Prefill阶段通常是**计算密集型**（Compute-Bound）的，瓶颈在于Tensor Core的算力上限。

**Decode阶段。** Decode每次仅生成一个Token（或在批处理下生成Batch Size个Token），权重矩阵乘法退化为小矩阵乘法甚至矩阵-向量乘法。此时Tensor Core的利用率较低，大量时间花在从HBM读取模型权重上。Decode阶段通常是**访存密集型**（Memory-Bound）的，瓶颈在于HBM的带宽。

这一分析直接解释了为什么不同的优化技术针对不同的阶段：FlashAttention主要优化Prefill阶段的Attention计算效率，而模型量化（减少权重的HBM读取量）和批处理优化（增大有效Batch Size以提高Tensor Core利用率）主要优化Decode阶段的吞吐。

---

## 3.2 GPU 内存层次：HBM、L2 Cache、Shared Memory、Register File

GPU内存层次的设计遵循一个经典的计算机体系结构原则：越靠近计算单元的存储，容量越小、速度越快。理解这一层次结构对推理优化的意义在于：推理引擎的大量优化本质上就是尽可能让数据在更快的存储层级上被计算，减少对慢速存储的访问。

### 3.2.1 HBM（高带宽显存）

HBM（High Bandwidth Memory）是GPU的主存储器，也是推理中模型参数和KV Cache的主要存放地。HBM通过3D堆叠和硅中介层（Silicon Interposer）技术实现了远超传统GDDR显存的带宽。

各代数据中心GPU的HBM规格对比如下表所示：

|GPU型号|HBM类型|容量|带宽|
|---|---|---|---|
|A100 SXM|HBM2e|80 GB|2.0 TB/s|
|H100 SXM|HBM3|80 GB|3.35 TB/s|
|H200 SXM|HBM3e|141 GB|4.8 TB/s|
|B200 SXM|HBM3e|192 GB|8.0 TB/s|

对于大模型推理而言，HBM的两个参数都极为关键。

**容量决定了能承载的模型规模和并发能力。** 一个FP16精度的70B参数模型需要约140 GB显存仅用于存放参数，这已经超过了单张H100的80 GB容量，必须使用张量并行或模型量化来解决。KV Cache的显存占用随着并发请求数和序列长度线性增长：以LLaMA-70B为例，GQA配置下每个Token的KV Cache约占0.625 KB（FP16），一个8192 Token的请求需要约5 MB的KV Cache，1000个并发请求就需要约5 GB。在长上下文场景（如128K Token）下，单个请求的KV Cache就可能达到80 MB。HBM容量不足是限制推理吞吐的最常见瓶颈之一——这正是vLLM的PagedAttention和各种KV Cache压缩技术的核心动机。

**带宽决定了Decode阶段的吞吐上限。** 在Decode阶段，每生成一个Token都需要从HBM中读取完整的模型参数（在无量化情况下）。一个FP16的70B模型参数约140 GB，在H100上以3.35 TB/s的带宽读取需要约42 ms，这意味着单请求Decode的理论速度上限约为每秒24 Token。这就是为什么Decode阶段被称为"访存密集型"——不是因为计算简单，而是因为数据搬运成了瓶颈。量化到INT8可以将参数量减半至70 GB，理论Decode速度提升到约48 Token/s；量化到INT4则进一步减至35 GB，理论可达约96 Token/s。这正是量化技术在推理中的核心价值来源。

### 3.2.2 L2 Cache

L2 Cache是片上（On-Chip）的全局共享缓存，位于HBM和SM之间。A100拥有40 MB的L2 Cache，H100增加到50 MB，B200进一步增大。L2 Cache的带宽远高于HBM——虽然NVIDIA通常不公开精确的L2带宽数字，但实测表明其带宽通常是HBM带宽的3到5倍。

在推理中，L2 Cache的作用体现在以下几个方面。

**热点数据缓存。** Embedding表、LayerNorm的参数（gamma/beta）等体积小但频繁访问的数据，一旦被加载到L2 Cache中就可以被快速重用，避免反复访问HBM。

**跨SM的数据共享。** 当多个SM需要访问同一块数据时（例如在张量并行中，不同SM可能需要读取同一层的偏置向量），L2 Cache可以作为共享缓冲，避免每个SM都去HBM取一次。

**KV Cache的部分缓存。** 对于当前活跃的请求，其最近生成的KV Cache有较高概率仍驻留在L2 Cache中，后续Token的Attention计算可以从L2直接命中，减少HBM访问。

然而，L2 Cache的容量相对于推理工作负载的数据量来说仍然很小。一个70B FP16模型的参数为140 GB，即使是50 MB的L2也只能缓存其0.036%。因此，L2 Cache的命中率高度依赖于数据访问模式的局部性——这也是为什么算子融合（Operator Fusion）如此重要：将多个连续的操作融合为一个Kernel，使得中间结果不必回写HBM再重新读取，而是在片上（L2或Shared Memory）直接传递。

### 3.2.3 Shared Memory（共享内存）

Shared Memory是SM内部的程序员可控的快速暂存区。它与L1 Data Cache共享同一块物理SRAM，用户可以通过编程配置二者的容量比例。以H100为例，每个SM拥有228 KB的可配置SRAM。

Shared Memory在推理Kernel中扮演着核心角色，特别是在FlashAttention的实现中。FlashAttention的核心思想是将大规模的Attention矩阵分块（Tiling），每次只将一小块Q、K、V数据从HBM加载到Shared Memory中进行计算，计算完成后将结果写回，再加载下一块。通过这种方式，FlashAttention将Attention的显存占用从O(N²)降低到O(N)，同时利用Shared Memory的高带宽实现了计算加速。

Shared Memory的带宽极高，A100和H100的理论Shared Memory带宽可达约19 TB/s（每个SM约175 GB/s × SM数量），远超HBM带宽。但其容量限制了每次能处理的数据块大小。在FlashAttention中，Block Size的选择就直接受Shared Memory容量制约：更大的Block Size可以提高Tensor Core利用率和数据重用率，但需要更多Shared Memory来存放Q、K、V的分块数据以及Softmax的中间结果。FlashAttention-2和FlashAttention-3在这一点上做了大量的工程权衡。

此外，Shared Memory还被用于实现高效的Warp间通信（同一Thread Block内的不同Warp共享Shared Memory）、归约操作的中间结果暂存、以及量化Kernel中的查找表（Lookup Table）存储等。

### 3.2.4 Register File（寄存器文件）

寄存器是GPU中速度最快的存储层级，单周期即可读写。每个SM拥有大容量的寄存器文件（H100为256 KB），供该SM上所有活跃线程使用。每个线程可使用的寄存器数量有上限（通常为255个32位寄存器），超过这个限制就会发生**寄存器溢出**（Register Spilling），溢出的数据会被放到Local Memory中（实质上位于HBM），这会导致严重的性能下降。

在推理Kernel的设计中，寄存器的使用需要精心管理。Tensor Core的MMA操作要求输入和输出矩阵片段存放在寄存器中；FlashAttention中Softmax的行最大值和行求和等在线归约状态也保存在寄存器中。如果一个Kernel使用过多的寄存器，不仅可能导致溢出，还会减少每个SM上能同时驻留的线程块数量（因为寄存器文件被更多地分配给了单个线程），从而降低Occupancy，削弱延迟隐藏的效果。这就是高性能Kernel优化中经典的**寄存器压力**（Register Pressure）问题。

### 3.2.5 内存层次总览与推理优化的映射

将上述各层次汇总，我们可以画出GPU内存层次的全景图：

|存储层级|典型容量（H100）|典型带宽|延迟|可编程性|
|---|---|---|---|---|
|Register File|256 KB/SM|~20 TB/s|~1 cycle|编译器/程序员|
|Shared Memory / L1|228 KB/SM|~19 TB/s|~30 cycles|程序员显式管理|
|L2 Cache|50 MB|~12 TB/s|~200 cycles|硬件自动管理|
|HBM|80 GB|3.35 TB/s|~400 cycles|程序员显式管理|

需要说明的是，上表中的带宽和延迟数字是近似值，实际值取决于访问模式、数据对齐等因素。但这些数字清晰地展示了一个数量级的差异：从HBM到Register，带宽提升约6倍，延迟降低约400倍。

推理优化中几乎所有的"算子优化"和"内存优化"，本质上都可以归结为一个目标：**让数据尽可能在靠近计算单元的存储层级上被处理，减少对HBM的访问次数。** FlashAttention通过Tiling将数据搬运到Shared Memory；算子融合避免中间结果回写HBM；KV Cache的分页管理通过紧凑存储减少内存碎片；量化将模型参数压缩到更小的体积，减少HBM读取量。所有这些技术，归根结底都是在与GPU内存层次的物理约束做斗争。

---

## 3.3 GPU 内存带宽与计算吞吐的平衡关系

在第2章中，我们介绍了Roofline模型的基本概念。本节将把Roofline模型的分析落实到具体的GPU硬件参数上，建立从硬件规格到推理性能上限的定量关系。

### 3.3.1 算术强度（Arithmetic Intensity）回顾

算术强度（Arithmetic Intensity, AI）定义为一个计算任务中每从内存读取/写入一个字节所执行的浮点运算数，单位为FLOPs/Byte：

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

这个简单的比值决定了一个计算任务的性能瓶颈是计算还是访存。当算术强度低于某个临界值时，性能受限于内存带宽（Memory-Bound）；当算术强度高于该临界值时，性能受限于计算吞吐（Compute-Bound）。这个临界值被称为**平衡点**（Balance Point）或**脊点**（Ridge Point），等于GPU的峰值算力除以峰值内存带宽：

$$\text{Ridge Point} = \frac{\text{Peak FLOPS}}{\text{Peak Memory Bandwidth}}$$

### 3.3.2 各代GPU的Roofline参数

以FP16精度为例（这是大模型推理最常用的精度之一），各代GPU的关键参数和对应的脊点如下：

**A100 SXM：** FP16 Tensor Core算力为312 TFLOPS，HBM2e带宽为2.0 TB/s。脊点 = 312 / 2.0 = 156 FLOPs/Byte。

**H100 SXM：** FP16 Tensor Core算力为989 TFLOPS（使用稀疏性时可达1979 TFLOPS，此处以稠密算力为准），HBM3带宽为3.35 TB/s。脊点 = 989 / 3.35 ≈ 295 FLOPs/Byte。

**H200 SXM：** FP16算力与H100相同（989 TFLOPS），但HBM3e带宽增至4.8 TB/s。脊点 = 989 / 4.8 ≈ 206 FLOPs/Byte。注意H200相比H100脊点反而降低了，这意味着在H200上更多的工作负载可以进入Compute-Bound区域——换句话说，H200的更高带宽使得原本被Memory-Bound的场景（如Decode）获得了实质性的加速。

如果考虑FP8精度，H100的脊点进一步升高（算力翻倍而带宽不变），Decode阶段更加深入Memory-Bound区域。这似乎是一个悖论：精度越低，Tensor Core算力越高，但Decode反而更Memory-Bound。原因在于FP8在减少计算量的同时也减少了数据体积，但由于Decode阶段的算术强度本身就很低（约为1-2 FLOPs/Byte的量级），即使翻倍也远远低于FP8的脊点。

### 3.3.3 推理两阶段在Roofline中的定位

**Prefill阶段的算术强度分析。** Prefill阶段的主要计算是对输入序列的大规模GEMM。考虑一个形状为 (M, K) × (K, N) 的矩阵乘法，总FLOPs为 2MKN（乘和累加各计一次），读取的数据量为 (MK + KN) × bytes_per_element（暂不考虑输出写回）。算术强度近似为：

$$\text{AI}_{\text{Prefill}} \approx \frac{2MKN}{(MK + KN) \times \text{bytes_per_element}}$$

当M（输入Token数）较大时，例如M = 2048，K = N = 4096（典型的FFN维度），FP16精度下：

$$\text{AI}_{\text{Prefill}} \approx \frac{2 \times 2048 \times 4096 \times 4096}{(2048 \times 4096 + 4096 \times 4096) \times 2} \approx \frac{6.87 \times 10^{10}}{4.97 \times 10^{7}} \approx 1382 \text{ FLOPs/Byte}$$

这远超H100的脊点（295 FLOPs/Byte），因此Prefill阶段处于Compute-Bound区域。Prefill的性能主要取决于Tensor Core的算力，优化方向是提高Tensor Core的利用率。

**Decode阶段的算术强度分析。** Decode阶段每次处理Batch Size为B的请求，权重矩阵乘法变为 (B, K) × (K, N)。当B较小时（例如B = 1，即单请求推理），FP16精度下：

$$\text{AI}_{\text{Decode}} \approx \frac{2 \times 1 \times K \times N}{(1 \times K + K \times N) \times 2} \approx \frac{2KN}{2KN} = 1 \text{ FLOPs/Byte}$$

当 B << N 时，权重矩阵的读取量（KN × bytes）远大于输入的读取量（BK × bytes），算术强度约等于B。即使B = 32，算术强度也仅为约32 FLOPs/Byte，仍远低于H100 FP16的脊点295。这意味着**在实际推理中，Decode阶段几乎总是Memory-Bound的。**

这一定量分析揭示了一个核心优化原则：**对于Decode阶段，提高性能的最有效手段不是增加更多的计算单元，而是要么减少需要从HBM读取的数据量（通过量化），要么增大Batch Size使算术强度提升进入Compute-Bound区域（通过更好的批处理调度），要么直接使用带宽更高的硬件（如HBM3e）。** vLLM的连续批处理和PagedAttention正是服务于增大有效Batch Size的目标；各种量化技术服务于减少数据读取量的目标。

### 3.3.4 Batch Size的"甜蜜区间"

Batch Size是Decode阶段算术强度的直接调节器。随着Batch Size增大，算术强度线性增长，性能也随之提升——但只在Memory-Bound区域内如此。一旦Batch Size足够大，算术强度超过脊点，性能就进入Compute-Bound区域，此后继续增大Batch Size将不再带来显著加速，反而会增加延迟。

此外，更大的Batch Size意味着需要同时维护更多请求的KV Cache，占用更多HBM容量。当KV Cache占用过多HBM时，留给模型参数和激活值的空间就不足，可能触发KV Cache的换出（Eviction）或请求抢占（Preemption），反而降低性能。这就形成了一个权衡：Batch Size存在一个"甜蜜区间"，在这个区间内，Tensor Core利用率足够高，KV Cache的显存压力还在可控范围内，系统吞吐达到最优。

推理引擎的调度器需要动态地寻找这个甜蜜区间。vLLM通过`max_num_batched_tokens`和`max_num_seqs`等参数来约束Batch Size；SGLang的调度器则通过运行时反馈动态调节。第8章和第9章将详细讨论这些调度策略。

---

## 3.4 多 GPU 互联拓扑：NVLink、NVSwitch、PCIe、InfiniBand

当模型规模超过单张GPU的容量限制时，必须将模型分布到多个GPU上进行推理。此时，GPU之间的通信带宽和延迟成为决定推理性能的关键因素。不同的互联技术提供了不同的带宽和拓扑特征，直接影响了分布式推理中并行策略的选择。

### 3.4.1 NVLink：GPU之间的高速直连

NVLink是NVIDIA开发的GPU间高速互联技术，提供了远超PCIe的带宽。

**NVLink的发展历程。** 从第一代（Pascal架构，单向20 GB/s × 4 link）到第四代（Hopper架构，H100），NVLink的带宽持续提升。H100搭载的NVLink 4.0提供18个Link，总双向带宽达到900 GB/s。B200搭载的NVLink 5.0进一步提升到1800 GB/s。

**NVLink的拓扑限制。** NVLink是点对点连接，每张GPU的NVLink接口数量有限。在没有NVSwitch的情况下，NVLink只能直接连接少数几张GPU（通常2到4张形成全连接）。对于需要8张GPU全连接的场景，NVLink alone不够用，需要NVSwitch。

### 3.4.2 NVSwitch：GPU全连接的桥梁

NVSwitch是NVIDIA的GPU互联交换机芯片，它将多张GPU通过NVLink全连接起来。

**节点内NVSwitch。** 在DGX H100服务器中，4个NVSwitch芯片将8张H100 GPU全连接，任意两张GPU之间都能以NVLink 4.0的全速（900 GB/s双向）通信。这对于张量并行（Tensor Parallelism）至关重要，因为张量并行要求每一层的计算后都执行All-Reduce通信，通信频率极高，必须有高带宽低延迟的互联支持。在8-GPU张量并行中，All-Reduce操作的有效带宽约为NVLink带宽的倍数关系，具体取决于All-Reduce算法（Ring、Tree等）。

**跨节点NVLink（NVLink Network / NVLink Switch）。** Blackwell架构引入了第五代NVSwitch，支持跨节点的NVLink互联，构建所谓的"NVLink域"（NVLink Domain）。在DGX GB200 NVL72系统中，72张GPU通过NVLink Switch全连接，形成一个超大的NVLink域，跨节点GPU通信的带宽与节点内相当。这对于超大模型（如DeepSeek-V3的671B参数MoE模型）的推理至关重要，因为这些模型可能需要跨多个节点分布，跨节点的高带宽NVLink可以显著降低Expert Parallelism中All-to-All通信的延迟。

### 3.4.3 PCIe：通用但带宽有限

PCIe（Peripheral Component Interconnect Express）是连接CPU与GPU、以及不同PCIe设备之间的标准总线接口。

**PCIe的带宽。** PCIe 4.0 x16提供约32 GB/s的单向带宽（64 GB/s双向）；PCIe 5.0 x16将其翻倍至约64 GB/s单向（128 GB/s双向）。与NVLink相比，即使是最新的PCIe 5.0也仅为NVLink 4.0的约七分之一。

**PCIe在推理中的角色。** PCIe主要用于CPU-GPU之间的数据传输以及没有NVLink连接的GPU之间的通信。在KTransformers的异构推理方案中，CPU完成MoE专家的计算后，需要通过PCIe将结果传回GPU。PCIe的有限带宽成为CPU-GPU协同推理的潜在瓶颈，这也是为什么KTransformers强调流水线化和异步传输来隐藏PCIe传输延迟。此外，在多GPU但无NVLink的消费级平台上（如多张RTX 4090通过PCIe桥接），张量并行的通信开销会显著增大，通常需要改用Pipeline Parallelism来减少通信频率。

### 3.4.4 InfiniBand / RoCE：跨节点高速网络

当推理需要扩展到多台服务器（多节点）时，节点之间的网络互联成为关键。

**InfiniBand（IB）。** InfiniBand是数据中心高性能计算的主流网络技术。NVIDIA的ConnectX-7网卡支持NDR 400Gb/s（约50 GB/s）的单端口带宽。DGX H100服务器通常配备8个ConnectX-7网卡（每张GPU配一个），总出口带宽达到400 GB/s（双向）。InfiniBand支持RDMA（Remote Direct Memory Access），允许GPU通过GPUDirect RDMA直接访问远程节点GPU的显存，绕过CPU参与，从而降低延迟。

**RoCE（RDMA over Converged Ethernet）。** RoCE在标准以太网上实现RDMA功能，成本低于InfiniBand但在大规模集群中的可靠性和拥塞控制能力稍弱。在一些推理部署场景中，100GbE或400GbE以太网 + RoCE是一种更经济的选择。

**跨节点通信对推理的影响。** 在Pipeline Parallelism中，节点间只需要传递层间激活值，通信量相对较小，InfiniBand或高速以太网通常足够。但在Expert Parallelism中，All-to-All通信的数据量可能很大（每个Token都需要发送到其路由的专家所在的节点），此时网络带宽可能成为瓶颈。DeepSeek-V3在推理中采用的EP（Expert Parallelism）方案就需要高效的跨节点All-to-All通信支持。vLLM的NIXL（Network Interface eXchange Layer）就是为解决高效跨节点传输问题而设计的抽象层，它封装了InfiniBand RDMA、GPUDirect等底层通信机制，为推理引擎提供统一的高性能传输接口。

### 3.4.5 互联拓扑对并行策略选择的影响

不同的互联带宽直接决定了并行策略的适用性。这里我们给出一个简化的决策框架：

**张量并行（TP）** 要求每一层都进行通信（All-Reduce），通信频率极高，对带宽和延迟都非常敏感。TP通常只在NVLink全连接的GPU之间使用。在DGX H100中，8张GPU通过NVSwitch全连接，TP的典型规模是2、4或8。跨节点做TP（通过InfiniBand通信）通常效率很低，除非使用Blackwell的NVLink Switch实现跨节点NVLink。

**流水线并行（PP）** 在层间切分模型，节点之间只需传递一次激活值（在没有微批调度的情况下），通信频率远低于TP。PP适用于跨节点或PCIe连接的GPU之间。但PP会引入"流水线气泡"（Pipeline Bubble），即某些阶段空闲等待的时间，在推理场景中需要通过调度优化来最小化。

**专家并行（EP）** 将MoE模型的不同专家分布在不同GPU上，需要All-to-All通信来路由Token。EP的通信量取决于激活的专家数量和Token数量，通常比TP的通信更"突发"（bursty）。EP通常需要NVLink或高带宽InfiniBand支持。

**数据并行（DP）** 将不同的请求分配到不同的模型副本上，各副本之间无需通信（推理时不需要梯度同步）。DP是最"通信友好"的策略，适用于任何互联条件，但需要每个副本都能容纳完整模型。

实际的大模型推理部署通常组合使用多种并行策略。例如，一个典型的DeepSeek-V3部署可能在节点内使用TP+EP（8张GPU通过NVSwitch），跨节点使用DP或PP。第13章将详细讨论这些并行策略的组合方式。

---

## 3.5 CPU 计算能力：AVX-512、AMX 指令集与大模型推理

在以GPU为中心的推理架构讨论中，CPU常常被忽视。然而，KTransformers引擎的出现表明，现代高端CPU的计算能力不容小觑，特别是在MoE模型的推理中，CPU可以承担大量的专家计算工作。本节将系统介绍与大模型推理相关的CPU计算能力。

### 3.5.1 现代服务器CPU的计算架构

现代x86服务器CPU（如Intel Xeon Scalable系列和AMD EPYC系列）已经发展出强大的并行计算能力。与GPU的大规模SIMT并行不同，CPU提供的是多核心 + 宽SIMD（Single Instruction, Multiple Data）的并行模式。

一颗典型的高端服务器CPU（如Intel Xeon w9-3595X，Granite Rapids架构）拥有多达数十个物理核心，每个核心配备独立的L1/L2缓存，多个核心共享大容量的L3缓存（可达数百MB）。每个核心支持宽SIMD指令集（AVX-512或更新的AMX），可以在单个时钟周期内对大量数据元素执行相同的运算。

### 3.5.2 AVX-512：512位SIMD向量计算

AVX-512（Advanced Vector Extensions 512-bit）是Intel从Skylake-SP服务器处理器开始引入的SIMD指令集扩展，将SIMD寄存器宽度从256位（AVX2）扩展到512位。

**基本能力。** 512位的SIMD寄存器可以同时容纳16个FP32元素或32个FP16/BF16元素，一条AVX-512指令可以同时对这些元素执行相同的运算（加法、乘法、乘累加等）。这意味着，相比标量计算，AVX-512在理论上可以提供16倍（FP32）或32倍（FP16）的吞吐提升。

**VNNI扩展。** AVX-512 VNNI（Vector Neural Network Instructions）进一步增加了对INT8/INT16乘累加的优化指令，可以在单个时钟周期内完成多组INT8乘法并累加到INT32，非常适合量化神经网络的推理。

**在推理中的应用。** 对于量化模型（如INT4/INT8量化的权重），CPU可以使用AVX-512 VNNI指令高效地执行矩阵-向量乘法。在KTransformers中，当MoE模型的专家权重以量化格式（如GGUF Q4_K_M）存储在CPU内存中时，CPU使用AVX-512指令执行专家的前向计算，这比不使用SIMD的标量计算快一到两个数量级。

**频率降档问题。** 值得注意的是，AVX-512指令的执行通常会导致CPU降频（称为"AVX-512 Frequency Throttling"），因为宽SIMD操作的功耗较高。不同的CPU型号降频程度不同。在评估CPU推理性能时，需要考虑实际的AVX-512运行频率而非标称频率。

### 3.5.3 AMX（Advanced Matrix Extensions）：CPU上的矩阵加速器

AMX是Intel从Sapphire Rapids（第四代Xeon Scalable）处理器开始引入的矩阵计算加速扩展，可以说是CPU上的"Tensor Core"。AMX代表了CPU计算能力的质的飞跃，也是KTransformers选择将MoE专家计算放在CPU上的关键技术基础。

**TMM（Tile Matrix Multiply）单元。** AMX引入了8个新的"Tile"寄存器，每个Tile寄存器大小为1 KB，可以存储一个16×64字节的矩阵。AMX的TMUL（Tile Matrix Multiply Unit）可以在单个指令周期内完成两个Tile寄存器中矩阵的乘累加操作。

**支持的数据类型。** AMX支持BF16和INT8两种数据类型。对于BF16，TMUL执行16×32 × 32×16 → 16×16的矩阵乘累加，每个周期产生256个BF16乘累加结果。对于INT8，由于数据更紧凑，每个周期可以产生更多的乘累加结果。

**AMX的计算吞吐。** 以Intel Xeon w9-3595X为例（Granite Rapids架构），单核AMX BF16吞吐可达数TFLOPS级别，整颗CPU（数十核）的AMX BF16吞吐可达数十TFLOPS。虽然这与GPU的数百到数千TFLOPS相比仍有数量级差距，但对于MoE模型中仅涉及少量激活专家的计算而言已经足够——特别是在Decode阶段，每个Token只激活2个专家（以DeepSeek-V3为例），每个专家的计算量有限，CPU通过AMX完全可以在合理的时间内完成。

**KTransformers的AMX利用。** KTransformers的核心思路是将MoE模型中Attention部分（需要访问KV Cache，对延迟敏感）放在GPU上执行，而将FFN中的专家计算卸载到CPU上，利用AMX指令加速。由于MoE模型的专家参数量巨大（DeepSeek-V3有256个专家，但每次只激活8个），将所有专家参数放在GPU显存中极不经济。CPU内存的容量通常为数百GB到数TB，足以容纳全部专家参数。AMX使得CPU能以可接受的速度完成这些计算，而不必支付昂贵的多GPU显存成本。

### 3.5.4 CPU内存系统：容量、带宽与NUMA

CPU内存系统的特征与GPU有显著不同，理解这些差异对于分析异构推理的性能至关重要。

**容量优势。** 单台服务器的DDR5内存容量通常为256 GB到2 TB甚至更高，远超单张GPU的HBM容量。这是CPU-GPU异构推理的核心动机之一：用廉价而充裕的CPU内存存储大量模型参数（特别是MoE专家的权重），用昂贵但高速的GPU显存存储需要高带宽访问的数据（如Attention层的权重和KV Cache）。

**带宽劣势。** DDR5内存的带宽远低于HBM。以8通道DDR5-5600为例，理论带宽约为358 GB/s（8 × 44.8 GB/s），仅为H100 HBM带宽（3.35 TB/s）的约十分之一。这意味着CPU端的计算同样可能面临访存瓶颈，特别是当专家权重以较高精度存储时。KTransformers通过使用低比特量化格式（如INT4）存储专家权重来缓解这一问题——INT4量化将每个参数压缩到4位，使得同样的内存带宽可以"喂入"更多的参数。

**NUMA（Non-Uniform Memory Access）。** 现代双路或多路服务器的CPU采用NUMA架构，每个CPU Socket有自己直连的本地内存，访问远端Socket的内存延迟更高、带宽更低。在KTransformers的实际部署中，需要确保专家参数存储在执行计算的CPU核心所在Socket的本地内存中，否则跨NUMA节点的内存访问会显著降低性能。这种NUMA感知的内存分配是KTransformers性能优化的重要工程细节。

**CPU缓存体系。** 现代服务器CPU拥有多级缓存体系：L1 Cache（通常32-48 KB指令缓存 + 32-48 KB数据缓存，每核心），L2 Cache（通常1-2 MB，每核心），L3 Cache（共享，总容量可达数百MB）。KTransformers在设计专家计算的Kernel时，会精心安排数据访问模式以最大化缓存命中率。例如，在执行量化权重的矩阵-向量乘法时，按缓存行（Cache Line）大小对齐数据访问，确保连续访问落在同一缓存行内，可以显著提升有效内存带宽。

### 3.5.5 AMD CPU的对应能力

虽然本节以Intel为主线（因为KTransformers最初针对Intel AMX优化），但AMD EPYC系列CPU也具备强大的计算能力。AMD通过AVX-512指令集（从Zen 4架构开始全面支持）提供SIMD计算能力，其AVX-512实现在某些场景下甚至比Intel更高效（因为AMD的Zen 4/5架构不会因AVX-512而降频）。在矩阵加速方面，AMD目前尚未推出与Intel AMX直接对应的指令集扩展，但其强大的AVX-512 VNNI支持和高核心数设计使得AMD CPU同样是异构推理的有力候选硬件。

---

## 3.6 异构计算平台：CPU-GPU 协同的硬件基础

前面几节分别介绍了GPU和CPU的计算与存储特征。本节将它们放在一起，分析CPU-GPU协同推理的硬件基础，这也是理解KTransformers架构设计的关键背景。

### 3.6.1 CPU-GPU协同的总线互联

CPU与GPU之间的数据传输主要通过PCIe总线完成。如3.4.3节所述，PCIe 5.0 x16提供约64 GB/s的单向带宽。在异构推理中，CPU完成专家计算后需要将结果（通常是一个维度为[Batch, Hidden]的张量）传回GPU，GPU完成Attention计算后可能需要将中间结果传给CPU。这些传输都经由PCIe。

**PCIe带宽对异构推理的制约。** 以DeepSeek-V3为例，每个Token经过一个MoE层后的输出维度为7168（hidden dimension），FP16下占14 KB。如果Batch Size为1，每层的CPU→GPU传输量仅为14 KB，即使在PCIe 4.0下也只需不到1微秒。但如果考虑到每个MoE层的传输都有启动延迟（通常为数微秒），以及CPU和GPU之间的同步开销，PCIe传输可能成为流水线效率的制约因素。KTransformers通过异步传输和计算-传输重叠来缓解这一问题。

**CXL（Compute Express Link）的前景。** CXL是一种新兴的互联标准，基于PCIe物理层但提供了缓存一致性（Cache Coherence）和内存共享（Memory Sharing）语义。CXL 3.0理论上允许CPU和GPU共享统一的内存地址空间，消除显式的数据拷贝。虽然CXL在大模型推理中的实际应用尚处于早期，但它可能在未来显著改变CPU-GPU异构推理的编程模型和性能特征。

### 3.6.2 KTransformers视角下的异构计算分工

KTransformers的核心设计思想是：根据不同计算任务的特征，将它们分配到最适合的硬件上执行。这一分工遵循以下原则：

**GPU负责延迟敏感、带宽需求高的计算。** Attention层的计算涉及大量KV Cache的读取，需要HBM的高带宽支持。同时，Attention层的QKV投影和输出投影是Dense矩阵乘法，适合GPU的Tensor Core。此外，模型中所有的Dense层（如Gate网络、Embedding、LM Head等）参数量相对较小，可以完全放在GPU显存中。

**CPU负责参数量大但计算量可控的专家计算。** MoE模型的专家FFN层参数量巨大（DeepSeek-V3的256个专家的总参数量远超Attention层），但每次推理只激活少数专家（如8个），单次计算量有限。这些专家权重以量化格式存储在CPU内存中，CPU通过AMX指令完成计算。

**流水线化执行。** CPU和GPU的计算可以流水线化执行：当GPU在计算第L层的Attention时，CPU可以同时计算第L-1层的专家输出（如果数据依赖允许），或者预取第L+1层专家的权重到缓存。这种计算-传输的重叠（Overlap）是KTransformers实现高效异构推理的关键技术。

### 3.6.3 异构平台的硬件选型考量

对于希望使用KTransformers进行本地推理的用户，硬件选型需要综合考虑以下因素。

**CPU的选择。** AMX支持是首要条件（Intel Sapphire Rapids及以后）。核心数越多、AMX吞吐越高，专家计算速度越快。大容量L3缓存有助于提高数据局部性。DDR5内存通道数和频率直接影响可用内存带宽。

**内存容量。** 需要足以容纳所有专家权重（量化后）。以DeepSeek-V3 671B模型为例，INT4量化后的总参数量约为335 GB。加上操作系统和其他开销，建议至少384 GB DDR5内存。

**GPU的选择。** 由于GPU只需处理Attention层和少量Dense层，对显存容量的要求大幅降低。KTransformers的典型配置使用单张24 GB消费级GPU（如RTX 4090）即可承载DeepSeek-V3的非专家参数。GPU的HBM带宽仍然重要，因为Attention的KV Cache读取是GPU端的主要瓶颈。

**PCIe连接。** 确保GPU以PCIe 4.0/5.0 x16全速连接，避免PCIe降速成为瓶颈。

这种异构配置的成本远低于纯GPU方案。部署DeepSeek-V3 671B模型的纯GPU方案需要至少8张H100（80 GB × 8 = 640 GB显存），硬件成本高达数十万美元。而KTransformers的异构方案仅需一颗高端Intel Xeon CPU加一张消费级GPU，总成本可控在数万美元量级。代价是吞吐量显著低于纯GPU方案，主要适用于单用户或低并发场景。这一成本-性能权衡将在第11章和第20章中详细分析。

---

## 3.7 新兴推理硬件简介：TPU、Groq LPU、Cerebras WSE、专用加速器

虽然NVIDIA GPU目前在大模型推理市场占据主导地位，但多种新兴硬件正试图从不同角度挑战这一格局。理解这些硬件的设计哲学，不仅有助于把握行业趋势，也能加深我们对推理计算本质需求的理解。

### 3.7.1 Google TPU（Tensor Processing Unit）

TPU是Google自主设计的AI专用加速器，专为Tensor运算优化。TPU系列已发展到第六代（TPU v6e/Trillium，2024年发布），被广泛用于Google内部的模型训练和推理服务。

**架构特点。** TPU的核心计算单元是Matrix Multiply Unit（MXU），这是一个大规模的脉动阵列（Systolic Array），用于高效执行矩阵乘法。与GPU的Tensor Core类似，MXU是TPU的算力核心，但其规模更大（典型配置为128×128或256×256），单次矩阵乘法的吞吐更高。TPU还配备了大容量的片上SRAM（称为HBM+VMEM架构），用于缓存中间结果和KV Cache。

**互联与系统。** TPU通过ICI（Inter-Chip Interconnect）高速互联，多个TPU芯片组成Pod（通常为数百到数千个芯片），形成一个超大规模的计算集群。这种紧密互联的架构使得TPU在大规模分布式推理中具有通信效率优势。

**对推理的意义。** TPU的大规模脉动阵列设计在Prefill阶段的大矩阵乘法中效率极高。大容量片上SRAM有助于缓存KV Cache，减少HBM访问。然而，TPU的生态封闭性（主要通过Google Cloud TPU使用，编程框架以JAX/TensorFlow为主）限制了其在开源推理引擎中的应用。目前vLLM和SGLang都提供了实验性的TPU支持，但成熟度远不如GPU后端。

### 3.7.2 Groq LPU（Language Processing Unit）

Groq的LPU采用了与GPU和TPU截然不同的设计哲学，代表了一种激进的推理优化思路。

**架构特点。** Groq LPU最显著的特征是**确定性执行**（Deterministic Execution）和**大容量SRAM**。传统GPU的执行模型是动态调度的——Warp调度器在运行时决定哪个Warp获得执行机会，Cache的命中与否也在运行时才能确定。Groq LPU则采用了TSP（Tensor Streaming Processor）架构，计算的调度在编译时静态确定，执行过程完全确定无缓存。每个LPU芯片配备约230 MB的SRAM（没有HBM），所有数据访问的延迟在编译时就已知。

**对推理的意义。** LPU的确定性执行消除了GPU中常见的动态调度开销和Cache Miss惩罚，在延迟上有显著优势。Groq在其GroqCloud推理服务中展示了极低的Token延迟（某些场景下每Token仅数毫秒）。但LPU也面临明显的限制：片上SRAM容量有限（230 MB），无法容纳大型模型的全部参数，需要通过多芯片分布来扩展，这增加了芯片间通信的压力。此外，没有HBM意味着批处理大量请求时的KV Cache存储是一个挑战。

**Groq的设计哲学揭示的洞察。** Groq LPU的出现证明了一个观点：GPU并非推理的唯一最优硬件形态。推理的计算模式相对固定（自回归生成的每一步结构几乎相同），不需要GPU那样复杂的动态调度机制。如果能将整个推理过程的数据流在编译时完全规划好，就可以消除大量的运行时开销。这一思路也影响了GPU上的推理优化——CUDA Graph（消除Kernel Launch开销）本质上也是在将动态调度变为静态调度。

### 3.7.3 Cerebras WSE（Wafer-Scale Engine）

Cerebras的WSE是目前世界上面积最大的芯片（晶圆级芯片），WSE-3的面积达到整个300mm硅晶圆，集成了90万个AI核心和44 GB的片上SRAM。

**架构特点。** WSE的核心理念是"消除片外内存访问"。通过将巨量的SRAM集成在芯片上，WSE试图让整个模型（或至少其热点参数）完全驻留在片上，完全避免HBM/DRAM访问的带宽和延迟瓶颈。WSE的片上总带宽可达数百PB/s级别，远超任何外部内存系统。

**对推理的意义。** 如果模型参数能完全放入WSE的片上SRAM中（44 GB足以容纳一个INT8量化的约25B-35B参数模型），推理将完全不受内存带宽限制——包括Decode阶段。这意味着Decode的速度将完全由计算能力决定，理论上可以实现极高的Token生成速率。Cerebras在其推理服务中展示了超过1000 Token/s的生成速度。

**局限性。** 44 GB的片上SRAM对于当前最大的模型（如DeepSeek-V3 671B）仍然不够。此外，WSE的编程模型和软件生态与主流的CUDA生态差异很大，移植现有的推理引擎代码工作量巨大。成本也是一个重要考量：单个WSE系统（CS-3）的价格远高于同等数量的GPU服务器。

### 3.7.4 其他专用推理加速器

除了上述三种已经进入商业化部署的硬件外，还有多种处于不同发展阶段的推理加速器值得关注。

**NVIDIA Grace-Hopper（GH200）超级芯片。** Grace-Hopper将NVIDIA的Grace ARM CPU和Hopper GPU通过NVLink-C2C高速连接（900 GB/s双向），两者之间的内存可以统一寻址。Grace CPU提供高达480 GB的LPDDR5x内存（带宽约500 GB/s），与H100的80 GB HBM形成互补。这种架构天然适合KTransformers类型的异构推理——CPU内存中的专家参数可以通过NVLink-C2C高速传输到GPU，比PCIe快一个数量级。

**Intel Gaudi加速器。** Intel的Gaudi系列（Gaudi 2/3）是面向AI训练和推理的专用加速器。Gaudi 3配备约128 GB的HBM2e显存和集成的RDMA网络接口，定位为H100的竞争对手。vLLM已经提供了对Gaudi的后端支持。

**AMD Instinct系列。** AMD MI300X配备192 GB HBM3显存（比H100的80 GB大2.4倍）和5.3 TB/s的带宽，在显存容量上有显著优势。对于显存受限的大模型推理场景（如长上下文推理中KV Cache占用大量显存），MI300X的大容量HBM是一个有吸引力的选择。vLLM和SGLang均提供了ROCm后端的支持。

**专用推理ASIC。** 一些初创公司和研究机构正在设计专门面向大模型推理的ASIC，如本书介绍的KLLM就提出了一种基于K-Means量化的硬件-软件协同设计方案。这类专用加速器通常针对推理的特定计算模式（如索引化矩阵乘法、稀疏注意力）设计专用的计算单元和片上存储架构，在推理的能效比上有可能超越通用GPU。但其面临的挑战是软件生态的建设——GPU之所以占据主导地位，很大程度上是因为CUDA生态的成熟和丰富。

### 3.7.5 硬件多样性对推理引擎设计的影响

推理硬件的多样化趋势对推理引擎的架构设计提出了新的要求。一个优秀的推理引擎需要具备**硬件抽象能力**——将上层的调度、批处理、KV Cache管理等逻辑与底层的硬件执行解耦，使得同一套上层逻辑可以运行在不同的硬件后端上。

vLLM和SGLang都在朝这个方向努力。vLLM通过Platform和Worker的抽象支持CUDA GPU、AMD ROCm、Intel Gaudi、TPU等多种后端。SGLang同样提供了多硬件后端的支持。KTransformers则通过Kernel注入框架实现了算子级别的可替换性——用户可以为不同的硬件编写不同的Kernel实现，并通过配置文件注入到推理流水线中。

从更长远的视角看，推理硬件的演进将持续影响推理优化的技术路径。如果未来的硬件能提供足够的片上存储来缓存大部分KV Cache（如Cerebras WSE的方向），那么当前围绕KV Cache显存管理的大量优化（PagedAttention、KV Cache压缩等）的重要性可能会降低。如果硬件原生支持稀疏计算（如专门为MoE设计的稀疏矩阵乘法单元），那么MoE推理的效率可能大幅提升。如果硬件支持更低精度的计算（如FP4、INT2甚至二值化），那么极端量化的推理方案（如KLLM的K-Means量化）可能获得更大的性能提升空间。硬件与软件的协同演进，将是推理优化领域最重要的长期趋势之一。

---

## 本章小结

本章从硬件视角全面分析了影响大模型推理性能的关键因素。我们首先深入GPU内部，理解了SM、Warp和Tensor Core的工作原理，以及它们如何映射到推理的两个阶段。然后分析了GPU多层内存体系——从HBM到Register的每一层都对推理优化有具体的影响。通过Roofline模型的定量分析，我们建立了GPU内存带宽与计算吞吐之间的平衡关系，解释了为什么Prefill是Compute-Bound而Decode是Memory-Bound，以及这一差异如何指导优化策略的选择。

在多GPU互联部分，我们分析了NVLink、NVSwitch、PCIe和InfiniBand的带宽特征及其对并行策略选择的影响。在CPU部分，我们详细介绍了AVX-512和AMX指令集，解释了它们如何使CPU成为MoE专家计算的可行平台——这是KTransformers异构推理方案的硬件基础。最后，我们概览了TPU、Groq LPU、Cerebras WSE等新兴推理硬件，分析了它们各自的设计哲学和对推理的启示。

理解硬件是理解优化的前提。接下来的章节中，我们将在此硬件知识基础上，逐一深入讨论注意力优化（第4章）、KV Cache管理（第5章）、模型量化（第6章）等核心推理优化技术，读者将看到每一项优化技术是如何精确地针对本章所揭示的硬件瓶颈而设计的。
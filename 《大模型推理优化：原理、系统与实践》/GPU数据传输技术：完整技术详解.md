# GPU数据传输技术：完整技术详解

## 一、GPU内部硬件架构

### 1.1 GPU Die的物理组成

一颗GPU die上集成了以下主要功能单元：

**SM（Streaming Multiprocessor）** 是GPU的基本计算单元。以H100为例，一颗die上有132个SM。每个SM内部包含：CUDA Core（整数和浮点运算单元）、Tensor Core（矩阵乘加速单元）、寄存器文件（每SM 256KB）、L1 Cache / Shared Memory（每SM 228KB，可配置比例分配）、Load/Store Unit（负责向内存子系统发出读写请求）、Warp Scheduler（调度线程束执行）。SM是唯一能执行用户编写的CUDA kernel代码的地方。所有计算都发生在SM内部的寄存器和L1/Shared Memory中。

**L2 Cache** 是所有SM共享的片上缓存。H100有50MB，B200进一步加大。L2在物理上被分成多个slice，分布在die的不同区域。L2的关键作用是：缓存从HBM读取的数据，减少重复访问HBM的次数；作为所有内存请求的汇聚点——无论SM要访问本地HBM、对端GPU的HBM、还是主机内存，请求都首先经过L2。

**Crossbar / NoC（Network on Chip）** 是片上互联网络。它将SM、L2 Cache的各个slice、PCIe接口、NVLink接口、Copy Engine等所有功能单元连接在一起。当一个SM要向NVLink接口发送数据时，数据就是通过这个Crossbar路由到NVLink端口的。Crossbar的内部带宽远高于任何外部接口，因此在正常工作负载下它不会成为瓶颈。

**PCIe接口** 是GPU与CPU、网卡等设备通信的通道。H100支持PCIe Gen5 x16，单向带宽约64 GB/s。

**NVLink接口** 是GPU与GPU之间的高速互联通道。H100有18个NVLink 4.0端口，总双向带宽900 GB/s。B200有18个NVLink 5.0端口，总双向带宽1800 GB/s。每个NVLink端口在物理上是独立的SerDes通道组，由die边缘的PHY（物理层电路）驱动。

**HBM（High Bandwidth Memory）** 是GPU的主存储器，位于die外部但通过硅中介层（silicon interposer）与die紧密互联。H100配备80GB HBM3，带宽3.35 TB/s。B200配备192GB HBM3e，带宽8 TB/s。HBM由多个独立的memory stack组成（H100为5个stack），每个stack有自己的独立内存控制器连接到die上。

### 1.2 这些单元之间的连接关系

理解数据传输的关键在于理解这些单元之间的拓扑连接。SM不直接连接HBM，也不直接连接NVLink。所有数据流都经过Crossbar进行路由。具体的连接拓扑是：SM通过Crossbar连接到L2 Cache的各个slice；L2 Cache的各个slice通过各自的内存控制器连接到对应的HBM stack；L2 Cache的各个slice同时通过Crossbar连接到PCIe接口和NVLink接口。

因此，任何从SM出发的内存访问请求，无论目标是本地HBM还是远端GPU的HBM，都遵循以下路径：SM → Crossbar → L2 Cache → Crossbar → 目标接口（HBM控制器 / NVLink端口 / PCIe端口）。

---

## 二、三种数据搬运机制

GPU上存在三种不同的硬件机制可以发起和执行数据传输。它们在发起者、执行者、适用场景、性能特征上都有本质区别。

### 2.1 Copy Engine（CE）

Copy Engine是GPU die上集成的专用DMA（Direct Memory Access）硬件引擎。H100有3个独立的CE。CE的唯一功能就是在两个内存地址之间搬运数据，它不能做任何计算。

**工作流程如下。** 首先，CPU上运行的程序调用CUDA Runtime API，例如`cudaMemcpyAsync(dst, src, size, kind, stream)`。这个调用不会立即执行数据搬运，而是将一个"搬运命令"放入指定stream对应的GPU命令队列中。当GPU的命令处理器从队列中取出这条命令时，它将命令派发给一个空闲的Copy Engine。Copy Engine收到命令后，开始自主工作：它根据源地址和目标地址的物理位置，通过相应的接口（HBM控制器、NVLink端口、PCIe端口）逐块读取源数据并写入目标地址。整个过程中，SM完全不参与，可以同时执行计算kernel。

**数据路径取决于源和目标的物理位置。** 如果是本地HBM到本地HBM（显存内拷贝），路径是：HBM → 内存控制器 → L2 → Crossbar → L2 → 内存控制器 → HBM。如果是本地HBM到对端GPU的HBM（通过NVLink），路径是：本地HBM → 本地内存控制器 → 本地L2 → 本地Crossbar → 本地NVLink端口 → NVLink物理链路 → 对端NVLink端口 → 对端Crossbar → 对端L2 → 对端内存控制器 → 对端HBM。如果是本地HBM到主机内存（通过PCIe），路径是：HBM → 内存控制器 → L2 → Crossbar → PCIe接口 → PCIe总线 → CPU内存控制器 → DRAM。

**性能特征方面，** CE执行大块连续数据传输时效率很高，能达到NVLink理论带宽的约81%（B200实测约726 GB/s对900 GB/s理论单向带宽）。但CE的劣势在于它只能搬运数据，不能在搬运过程中做任何计算（如加法、归约），且CE的数量有限（H100只有3个），无法并行执行大量独立的小传输。CE的启动延迟也较高，需要经过CPU下发命令、GPU命令处理器调度等步骤，不适合频繁的小数据传输。

### 2.2 SM的Load/Store指令

SM上运行的CUDA kernel中的每条内存读写指令（在PTX层面对应`ld.global`和`st.global`）本身就是一次数据传输。这不是一种专门的"数据传输模式"，而是GPU最基本的内存访问方式。

**工作流程如下。** kernel中的线程执行一条load指令，例如读取地址0x7F0000001000处的数据。GPU的MMU（内存管理单元）将这个虚拟地址翻译成物理地址。物理地址中编码了目标的物理位置——是本地HBM的哪个bank，还是对端GPU的NVLink端口编号。如果是本地HBM，请求经过L2 Cache（可能命中缓存直接返回，也可能miss后从HBM读取）。如果是对端GPU的地址，请求经过本地L2（不会缓存对端数据），通过Crossbar路由到NVLink端口，发送到对端GPU，对端GPU的NVLink端口接收后通过对端Crossbar路由到对端L2，从对端HBM读取数据，原路返回。

**地址映射的实现方式有两种。** 第一种是CUDA Unified Virtual Addressing（UVA），CUDA runtime自动将所有GPU的显存和主机内存映射到一个统一的虚拟地址空间，GPU的MMU在硬件层面知道哪个地址范围对应哪个物理设备。第二种是CUDA IPC（Inter-Process Communication），通过`cudaIpcGetMemHandle`导出一块GPU显存的句柄，另一个进程（可能控制另一个GPU）通过`cudaIpcOpenMemHandle`将这块显存映射到自己GPU的虚拟地址空间，此后就可以像访问本地内存一样访问它。

**关键的硬件行为是，** Stanford Hazy Research的实测确认：当SM通过NVLink访问对端GPU的数据时，对端数据在读取后不会被缓存到本地GPU的L2 Cache中。这意味着如果kernel多次读取同一个对端地址，每次都必须重新经过NVLink传输。这是一个硬件设计决策——如果缓存对端数据，就需要复杂的缓存一致性协议来处理对端GPU修改数据后的失效问题，NVIDIA选择了简单但一致性语义更清晰的方案。

**性能特征方面，** SM通过load/store指令访问对端GPU时，NVLink带宽利用率约60%。利用率不如CE高的原因是：SM发出load指令后必须等待数据返回才能使用，期间该线程被挂起（虽然warp scheduler会切换到其他warp执行，但总有一个上限）；要达到高带宽需要大量SM同时发出大量并发的内存请求，且访问模式必须是coalesced的（一个warp内32个线程访问连续地址，合并成一次宽内存事务）。SM的load/store方式的独特优势是可以在搬运数据的同一个kernel中对数据做计算，例如读取对端数据后立即求和写入本地，这就是vLLM Custom AllReduce能将通信和计算融合在一个kernel中的基础。

### 2.3 TMA（Tensor Memory Accelerator）

TMA是NVIDIA从Hopper架构（H100）开始引入的专用硬件单元。它的设计目标是高效搬运多维tensor数据，同时减少SM在数据搬运上的开销。

**TMA的硬件位置在SM内部，** 但它是一个独立于CUDA Core和Tensor Core的功能单元。每个SM有自己的TMA单元。

**工作流程如下。** 程序员创建一个Tensor Map Descriptor，其中描述了源tensor的基地址、维度、步长、数据类型、swizzle模式等信息。kernel中的一个线程（只需一个，不需要整个warp）执行TMA专用PTX指令，例如`cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes`。TMA硬件接收到这条指令后，根据descriptor自动计算出需要读取的所有内存地址，发出所有内存请求，并将数据写入目标位置（通常是Shared Memory）。整个搬运过程是异步的，发出TMA指令的线程和同一SM上的其他线程可以继续执行其他计算指令。搬运完成通过mbarrier（内存屏障）通知。

**TMA的关键创新在于，** 传统的SM load/store方式搬运一个32×32的矩阵块需要1024个线程各自发出一条load指令，这1024条指令都要占用SM的指令发射带宽和寄存器。而TMA方式只需要一个线程发出一条指令，TMA硬件自动处理所有地址计算和内存请求，剩余所有线程和SM资源可以做计算。

**TMA与NVLink的结合使用** 是Stanford Hazy Research的一个重要发现。TMA指令中的源地址不限于本地HBM——如果源地址指向对端GPU的显存（通过IPC映射），TMA会自动通过NVLink发起远程读取。更进一步，TMA可以与NVSwitch的multicast地址配合使用，实现一对多的广播传输。实测数据显示：TMA方式访问NVLink对端数据的带宽利用率约74%，高于普通SM load/store的60%；只需要8到16个SM发出TMA指令就能接近打满NVLink带宽，而普通load/store方式需要占用大量SM才能达到类似带宽。这意味着可以用少量SM负责通信，绝大多数SM专注于计算，实现真正细粒度的通信-计算overlap。

---

## 三、三条GPU间数据传输路径的详细解剖

### 3.1 路径一：NVLink直传（同一机器内GPU到GPU）

**物理链路结构。** NVLink的每个端口（称为一个link）由多对高速差分信号线组成。每对差分线称为一个lane。NVLink 4.0（H100）的每个link包含2个sub-link（一收一发），每个sub-link由多个lane组成，每个lane的信号速率为53.125 Gbps（PAM4编码，物理速率26.5625 GBaud）。18个link的总双向带宽为900 GB/s。NVLink 5.0（B200）速率翻倍，18个link总双向带宽1800 GB/s。

**信号传输的物理过程。** 数据从HBM的存储单元中被读出，以并行数字信号的形式到达die边缘的NVLink PHY电路。PHY中的SerDes（Serializer/Deserializer）将并行数据转换为高速串行信号，使用PAM4编码（每个符号承载2 bit）。信号经过驱动器放大后，通过焊球（bump）从die传到封装基板，再通过封装引脚传到PCB或高速铜缆/光缆。在NVLink 4.0中，GPU之间可以通过PCB trace直接互联（同一PCB上），也可以通过NVSwitch互联（铜缆或PCB连接）。在NVLink 5.0中引入了光互联选项，用光收发器将电信号转换为光信号，通过光纤传输后再转回电信号。对端GPU的NVLink PHY接收信号后，SerDes将串行数据还原为并行数据，通过die上的NVLink接口逻辑进入Crossbar网络。

**协议层面，** NVLink使用NVIDIA私有的协议栈。链路层负责流量控制（基于credit的流控）、错误检测和纠正（CRC校验+重传）。事务层负责封装读写请求和响应。不同于PCIe的树形拓扑，NVLink支持点对点和通过NVSwitch的多对多拓扑。

**NVSwitch的作用。** NVSwitch是一个独立的ASIC芯片，功能是NVLink的交换机。在DGX H100中，4个NVSwitch芯片将8个H100 GPU全连接互联。每个GPU的18个NVLink端口分散连接到不同的NVSwitch上。当GPU0要给GPU5发数据时，数据从GPU0的某些NVLink端口出发，到达某个NVSwitch，NVSwitch的交叉开关将数据路由到连接GPU5的端口，数据到达GPU5。NVSwitch内部没有HBM或DRAM等持久存储，它只有用于缓冲在途数据包的SRAM buffer。数据在NVSwitch中的停留时间仅为路由延迟级别（纳秒量级），不会"存放"在NVSwitch中。

**NVSwitch的in-network compute能力。** 虽然NVSwitch没有持久存储，但从第三代NVSwitch（用于DGX H100及之后）开始，NVSwitch内部集成了简单的计算单元，可以在数据流经交换机时执行归约操作（如加法）。这是通过multicast地址和`multimem.ld_reduce`/`multimem.red`指令实现的。具体机制是：NVIDIA在NVLink地址空间中定义了一组特殊的multicast地址。当多个GPU同时向同一个multicast地址写入数据时，NVSwitch接收到来自多个GPU的数据包后，在内部完成逐元素加法，将结果存入内部buffer。当GPU从这个multicast地址读取时，NVSwitch返回归约后的结果。这使得AllReduce操作可以在一次写入加一次读取中完成，而不需要传统ring或tree算法的多轮通信。

### 3.2 路径二：GPUDirect RDMA（跨节点GPU到GPU）

**基本原理。** GPUDirect RDMA的核心思想是让网卡（NIC）能直接访问GPU的显存，绕过CPU内存作为中间缓冲。

**实现这一目标需要解决的关键问题是地址映射。** GPU的HBM有自己的物理地址空间，这个地址空间对CPU和其他PCIe设备来说不是直接可见的。为了让网卡能访问GPU显存，需要将GPU显存的一部分物理地址映射到PCIe的地址空间中（称为PCIe BAR空间）。NVIDIA的GPU驱动通过配置GPU的PCIe BAR寄存器来实现这一点。Linux内核中的`nvidia-peermem`模块负责将GPU显存注册到InfiniBand子系统的memory region中，使得IB verbs API可以操作GPU显存地址。

**发送方向的完整数据路径如下。** 应用程序（如NCCL库）调用IB verbs的`ibv_post_send`，传入一个scatter-gather list，其中buffer的地址指向GPU显存。网卡的DMA引擎收到这个发送请求后，根据buffer的物理地址（已通过memory registration翻译为DMA地址），发起PCIe读事务（Memory Read TLP）。这个PCIe读事务通过PCIe交换结构到达GPU。GPU的PCIe接口逻辑接收到这个读请求后，将其转化为内部的内存读操作：通过Crossbar路由到L2 Cache，再到相应的HBM控制器，从HBM读出数据。读出的数据通过相反的路径返回：HBM → 内存控制器 → L2 → Crossbar → PCIe接口 → PCIe总线 → 网卡。网卡收到数据后，将其封装成InfiniBand或RoCE数据包（添加IB传输层头部、网络层头部、链路层头部），从网络端口发送出去。

**接收方向的完整数据路径如下。** 对端网卡从网络上收到数据包后，剥离协议头部，提取出有效载荷数据。网卡的DMA引擎根据预先注册的接收buffer地址（指向对端GPU的显存），发起PCIe写事务（Memory Write TLP）。这个写事务通过PCIe交换结构到达对端GPU。GPU的PCIe接口将写请求通过Crossbar路由到L2，再到HBM控制器，数据被写入GPU显存。

**关键特征是，** 在整个过程中，CPU不参与任何数据拷贝。CPU的作用仅限于初始化阶段（配置地址映射、创建QP、注册MR等），以及发送/接收的信令（发出ibv_post_send/recv命令）。数据平面完全由网卡DMA引擎和GPU PCIe接口硬件完成。

**性能瓶颈在PCIe。** 即使网卡的网络端口带宽很高（如ConnectX-7支持400 Gbps），PCIe Gen5 x16的单向带宽只有约64 GB/s（约512 Gbps）。因此，单个GPU通过单个PCIe链路能达到的网络吞吐量受限于PCIe带宽。在DGX H100架构中，NVIDIA通过给每个GPU配备一个专用的ConnectX-7网卡（共8个），并将每个网卡通过独立的PCIe x16链路连接到对应的GPU，来最大化总聚合网络带宽（8 × 400 Gbps = 3.2 Tbps）。

### 3.3 路径三：传统CPU中转方式

在没有GPUDirect RDMA的环境中，数据传输需要CPU内存作为中转站。路径是：GPU显存（通过GPU的Copy Engine经PCIe）→ CPU的DRAM（通过CPU的DMA经PCIe或QPI/UPI）→ 网卡发送缓冲区 → 网络 → 对端网卡接收缓冲区 → 对端CPU DRAM → 对端GPU显存。这条路径比GPUDirect RDMA多了两次PCIe传输（GPU↔CPU内存）和两次内存拷贝，延迟更高，CPU内存带宽也可能成为瓶颈。NCCL在检测到GPUDirect RDMA不可用时会退回到这种方式。

---

## 四、vLLM Custom AllReduce的硬件层面完整分析

### 4.1 初始化阶段

每个参与AllReduce的GPU进程分配一块GPU显存作为通信buffer。每个进程调用`cudaIpcGetMemHandle`获得这块buffer的IPC句柄——这个句柄编码了这块内存的物理位置信息，可以通过进程间通信（如shared memory或socket）传递给其他进程。每个进程收到其他所有GPU的IPC句柄后，调用`cudaIpcOpenMemHandle`将对端的buffer映射到自己GPU的虚拟地址空间。在这一步中，GPU驱动会配置本GPU的页表（Page Table），将对端GPU的物理地址映射到本GPU的虚拟地址范围。页表中的条目会标记这些地址的物理位置是"对端GPU，通过NVLink的第N个端口可达"。此后，本GPU的SM执行load/store指令访问这些虚拟地址时，MMU通过查页表就知道要将请求路由到NVLink端口。

### 4.2 执行阶段（以TP=4，one-shot模式为例）

**第一步：本地写入。** 每个GPU的kernel将自己需要归约的数据写入自己的buffer。这是普通的本地HBM写入：SM → L2 Cache → HBM控制器 → HBM。

**第二步：内存屏障。** 执行`__threadfence_system()`，这条指令确保前一步的写入对所有GPU（包括通过NVLink观察的对端GPU）可见。在硬件层面，这条指令会等待所有pending的store操作从L2 Cache flush到HBM，并且NVLink的一致性机制确认对端可以看到最新数据。

**第三步：同步屏障。** 所有GPU通过一个轻量级的flag-based屏障确认都已完成写入。具体实现是每个GPU原子地设置一个共享的flag值，然后轮询等待所有GPU的flag都就绪。flag本身也存储在GPU显存中，通过NVLink IPC映射实现跨GPU访问。

**第四步：融合读取+计算+写入。** 这是核心步骤。每个GPU的kernel同时执行以下操作：读取本地buffer中的数据（本地HBM读取），读取其他3个GPU的buffer中的数据（通过NVLink远程读取），将4份数据逐元素相加，将结果写入本地的输出tensor（本地HBM写入）。在硬件层面，一个GPU的SM在执行这一步时，SM发出4条load指令（1条本地+3条远程）。本地load走SM → Crossbar → L2 → HBM控制器 → HBM → 返回数据。3条远程load各自走SM → Crossbar → L2（不缓存，直接转发）→ Crossbar → NVLink端口 → NVLink链路 → 对端NVLink端口 → 对端Crossbar → 对端L2 → 对端HBM → 原路返回数据。SM收到4份数据后在寄存器中完成加法，然后发出1条store指令将结果写入本地HBM。

**第四步中所有4个GPU同时执行这一操作，** 因此NVLink上同时存在双向的大量数据流。NVSwitch的全交叉开关特性保证了所有GPU对之间的传输可以同时以全带宽进行，不存在带宽冲突。

### 4.3 为什么这比NCCL的Ring AllReduce更快（小数据量时）

NCCL的Ring AllReduce将数据分成N份（N为GPU数），在ring上传递N-1次reduce-scatter再传递N-1次allgather，总共2(N-1)步。每一步都是一个独立的kernel launch，有启动延迟（约2-5μs的host-to-device命令延迟 + kernel调度延迟）。此外，NCCL使用Copy Engine进行数据传输，CE的启动开销也需要考虑。

vLLM的Custom AllReduce只有一个kernel launch，所有通信和计算都在这个kernel内部完成。对于LLM推理时典型的小数据量（几KB到几百KB），数据传输的时间远小于kernel launch和同步的固定开销。因此，减少kernel launch次数从2(N-1)次到1次带来的延迟节省是决定性的。

但对于大数据量（几十MB以上），数据传输时间主导总延迟，此时Ring AllReduce的流水线特性（每个GPU在任一时刻只需要发送1/N的数据）和CE的高带宽利用率反而更优。vLLM的Custom AllReduce在数据量超过阈值时会退回到NCCL。

---

## 五、DMA的本质

贯穿上述所有路径的核心技术是DMA（Direct Memory Access）。DMA的本质很简单：一个专用的硬件电路（DMA引擎/DMA控制器）能够在两个内存地址之间自主传输数据，不需要通用处理器（CPU或GPU的SM）逐字节参与。

DMA引擎的工作方式是：接收一组参数——源地址、目标地址、传输长度（以及可能的步长、维度等信息）；自主发起内存总线事务（读事务和写事务），按顺序或按设定模式在源和目标之间搬运数据；传输完成后通过中断或状态寄存器通知发起者。

在GPU数据传输中涉及的DMA引擎有：GPU的Copy Engine是DMA引擎——搬运GPU显存与主机内存或其他GPU显存之间的数据；网卡的DMA引擎——搬运网卡与CPU内存或GPU显存之间的数据（GPUDirect RDMA时直接搬运GPU显存）；CPU芯片组中也有DMA控制器，但在现代GPU通信路径中较少直接参与。

DMA与SM load/store的本质区别在于：DMA是"设置好参数后自动执行"的批量操作，适合大块连续数据搬运；SM load/store是"每条指令搬运一个cache line"的细粒度操作，适合需要与计算交织的场景。TMA可以看作两者的融合——它具有DMA的"设置一次自动搬运一大块"的特性，但它集成在SM内部，可以与SM的计算指令并行执行，并且支持异步完成通知。

---

## 六、信号层面的数据传输

数据在各种互联上的物理传输形式也值得了解。HBM与GPU die之间使用硅中介层上的微凸块互联，信号是并行低速差分信号（相对于SerDes来说），每个HBM stack有1024 bit宽的数据总线，频率约为数GHz级别，通过宽总线×适中频率达到高带宽。NVLink使用高速SerDes串行链路。每个lane使用PAM4编码（4电平脉冲幅度调制），每个符号传输2 bit。NVLink 4.0每lane 53.125 Gbps，NVLink 5.0每lane 106.25 Gbps。信号在铜缆或PCB trace上以电磁波形式传播，速度约为光速的60-70%（取决于介质的介电常数）。接收端使用均衡器（equalizer）补偿信号在传输中的衰减和失真，再由CDR（Clock and Data Recovery）电路恢复时钟和数据。PCIe同样使用SerDes串行链路，Gen5每lane 32 GT/s（NRZ编码，每符号1 bit），x16链路总共约64 GB/s单向。InfiniBand / Ethernet网络链路在机柜内使用铜缆（DAC或ACC），较长距离使用光纤。400G InfiniBand使用4个100G lane，每lane PAM4编码。

在所有这些互联上，数据都是以协议数据包（packet）的形式传输的。每个packet包含头部（路由信息、序号、类型）、有效载荷（实际数据）、尾部（CRC校验码）。硬件链路层负责数据包的组装、发送、接收、校验、错误重传。上层（事务层）看到的是可靠的字节流或可靠的读/写事务。
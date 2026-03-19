
## 第一阶段：DGX H100 —— NIC 挂在 PCIe 上，经典 GPUDirect RDMA

这是你原文描述的架构，也是最好理解的：

```
         节点A                                    节点B
    ┌─────────────┐                          ┌─────────────┐
    │  GPU (HBM)  │                          │  GPU (HBM)  │
    │      │      │                          │      │      │
    │   PCIe x16  │                          │   PCIe x16  │
    │      │      │                          │      │      │
    │  ConnectX-7 │══════ InfiniBand ══════> │  ConnectX-7 │
    └─────────────┘                          └─────────────┘
```

NIC 和 GPU 在同一个 PCIe root complex 下，NIC 的 DMA 引擎可以直接读写 GPU 的 BAR 空间（即 HBM），这就是经典的 GPUDirect RDMA。瓶颈是 PCIe Gen5 x16 的 ~64 GB/s。

---

## 第二阶段：GB200 NVL72 —— 一个意外的"退步"

GB200 的架构发生了根本性变化：GPU 不再挂在 PCIe 上，而是通过 **NVLink-C2C** 连接到 Grace CPU。

```
GB200 Superchip 内部：

    Grace CPU (Arm)
    ├── LPDDR5X 480GB（CPU本地内存）
    ├── PCIe Gen5 ──── ConnectX-7 NIC（scale-out 网卡）
    ├── PCIe Gen5 ──── BlueField-3 DPU（存储/管理）
    │
    └── NVLink-C2C (900 GB/s 双向) ──── B200 GPU × 2
                                          └── HBM3e 192GB × 2
```

注意关键变化：**GPU 没有 PCIe endpoint 了**。GPU 与 CPU 之间是 NVLink-C2C，而 NIC 挂在 Grace CPU 的 PCIe 总线上。

这意味着：**经典的 GPUDirect RDMA 在这个拓扑上不工作**。因为 GPUDirect RDMA 要求 NIC 和 GPU 共享同一个 PCIe 根复合体，NIC 的 DMA 引擎直接向 GPU HBM 发起 PCIe Memory Write TLP。但现在 GPU 根本不是 PCIe 设备，NIC 的 DMA 引擎"看不到" HBM。

### 替代方案：NIC → LPDDR5X → NVLink-C2C → HBM

实际的数据路径变成了：

```
跨节点发送（本地 GPU → 远端 GPU）：

  本地 GPU HBM 
       │ NVLink-C2C (900 GB/s)
       ▼
  本地 Grace LPDDR5X（中转缓冲）
       │ PCIe Gen5 DMA
       ▼
  本地 ConnectX-7 NIC 
       │ InfiniBand/RoCE
       ▼
  远端 ConnectX-7 NIC
       │ PCIe Gen5 DMA
       ▼
  远端 Grace LPDDR5X（中转缓冲）
       │ NVLink-C2C (900 GB/s)
       ▼
  远端 GPU HBM
```

虽然多了 LPDDR5X 这一跳中转，但有几个重要的优点：

**第一**，整个过程仍然不需要 CPU 做 memcpy，数据搬运全部由 DMA 引擎和 NVLink-C2C 硬件完成。GPU 可以直接通过 NVLink-C2C 的一致性协议读写 LPDDR5X，不需要 CPU 介入。

**第二**，NVLink-C2C 的带宽是 900 GB/s，远高于 PCIe Gen5 的 64 GB/s，所以 GPU ↔ LPDDR5X 这一段不是瓶颈。真正的瓶颈还是 NIC 端的 PCIe Gen5（~64 GB/s）。

**第三**，NVLink-C2C 的延迟极低（芯片间互联），且支持一致性协议（coherent），GPU 可以像访问本地内存一样访问 LPDDR5X。

### 那为什么要这样设计？

因为 GB200 NVL72 的核心理念是：**把 72 个 GPU 通过 NVLink 构成一个巨大的统一内存域**。

```
GB200 NVL72 机柜内部拓扑：

    ┌─────────────────────────────────────────┐
    │          9 × NVLink Switch Tray          │
    │    (每个 tray 2 颗 NVLink Switch 芯片)    │
    │     全交叉互联，130 TB/s 聚合带宽         │
    └──┬───┬───┬───┬───┬───┬───┬───┬───┬──────┘
       │   │   │   │   │   │   │  ...  │
      GPU0 GPU1 GPU2 GPU3 ............GPU71
       │   │   │   │              │   │
      NVLink-C2C                 NVLink-C2C
       │   │   │   │              │   │
     Grace0 Grace1  ...        Grace35
       │   │   │   │              │   │
      PCIe  PCIe  PCIe           PCIe  PCIe
       │   │   │   │              │   │
      NIC  NIC  NIC NIC          NIC  NIC
    (ConnectX-7, 用于 scale-out 跨机柜通信)
```

**节点内**（72 GPU 之间）：数据走 NVLink，900 GB/s/GPU，完全不经过 PCIe 和 NIC，延迟极低。72 个 GPU 的 HBM 构成 ~13.8 TB 的统一地址空间，任何 GPU 可以直接访问任何其他 GPU 的 HBM。

**跨机柜**（scale-out）：数据必须经过 NIC，走上面描述的 HBM → C2C → LPDDR5X → PCIe → NIC 路径。每个 compute tray 有 4 个 ConnectX-7 NIC（4 × 400 Gbps = 1.6 Tbps），提供跨机柜通信。

这种设计的哲学是：**用超大的 NVLink 域最大化节点内通信效率，让大部分通信（比如 tensor parallelism、pipeline parallelism 的大量通信）留在 NVLink 域内，只有 data parallelism 的 gradient 同步等相对稀疏的通信才走跨机柜的 NIC**。

---

## 第三阶段：Vera Rubin NVL72 —— 真正的"NIC 接入 NVLink 域"方向

2026 年 1 月 NVIDIA 在 CES 上发布了 Vera Rubin 平台，这是你问的"未来方向"的关键。

根据 NVIDIA 官方技术博客，Vera Rubin NVL72 的关键架构变化包括：

### 六颗新芯片协同设计

Vera Rubin 平台包含六颗全新芯片：Vera CPU、Rubin GPU、NVLink 6 Switch、ConnectX-9 SuperNIC、BlueField-4 DPU、Spectrum-6 以太网交换机。这六颗芯片是作为一个统一系统从头协同设计的。

### NVLink 6：带宽翻倍

每颗 Rubin GPU 的 NVLink 6 带宽达到 **3.6 TB/s 双向**（Blackwell 是 1.8 TB/s），NVLink-C2C 带宽达到 **1.8 TB/s**（Blackwell 是 900 GB/s）。9 个 NVLink 6 switch tray 在机柜内提供 **260 TB/s** 的聚合带宽。

### ConnectX-9 SuperNIC：每颗 GPU 1.6 Tb/s scale-out 带宽

```
Vera Rubin NVL72 Compute Tray 内部：

    ┌──────────────────────────────────────────────────┐
    │            Vera Rubin Compute Tray                │
    │                                                  │
    │  ┌─────────────┐          ┌─────────────┐       │
    │  │  Rubin GPU  │          │  Rubin GPU  │       │
    │  │  288GB HBM4 │          │  288GB HBM4 │       │
    │  │  22TB/s BW  │          │  22TB/s BW  │       │
    │  └──────┬──────┘          └──────┬──────┘       │
    │         │ NVLink-C2C 1.8TB/s      │ NVLink-C2C   │
    │  ┌──────┴──────┐          ┌──────┴──────┐       │
    │  │  Vera CPU   │          │  Vera CPU   │       │
    │  │ 88 Olympus  │          │ 88 Olympus  │       │
    │  │ 1.5TB LPDDR │          │ 1.5TB LPDDR │       │
    │  └──┬───┬──────┘          └──────┬───┬──┘       │
    │     │   │                        │   │          │
    │   PCIe  PCIe                   PCIe  PCIe       │
    │     │   │                        │   │          │
    │  ┌──┴─┐┌┴──┐                 ┌──┴─┐┌┴──┐       │
    │  │CX-9││BF4│                 │CX-9││BF4│       │
    │  │Quad││DPU│                 │Quad││DPU│       │
    │  └────┘└───┘                 └────┘└───┘       │
    │                                                  │
    │  ←──── NVLink 6 Spine Connectors ────→          │
    └──────────────────────────────────────────────────┘
```

每个 compute tray 有 **quad ConnectX-9 SuperNIC 板卡**，每块板卡提供 4 × 800 Gbps = 3.2 Tbps。NVIDIA 的说法是每颗 Rubin GPU 对应 **1.6 Tb/s** 的 scale-out 网络带宽。

### 关键问题：NIC 是否真正接入了 NVLink 域？

根据目前公开的信息，在 GB200 和 Vera Rubin NVL72 中，**ConnectX NIC 仍然是通过 PCIe 连接到 Vera/Grace CPU 的**，而不是直接挂在 NVLink fabric 上。数据路径仍然是：

```
GPU HBM → NVLink-C2C → CPU LPDDR5X → PCIe → NIC → 网络
```

但这里有一个非常关键的区别：**在这个架构中，PCIe 瓶颈的严重性被大幅降低了**，原因如下：

**第一，PCIe 升级到 Gen6**。Vera Rubin 支持 PCIe Gen6，单向带宽 ~128 GB/s（x16），比 Gen5 翻倍。

**第二，scale-up 域的极大扩展吸收了大部分通信**。72 GPU 在 NVLink 域内以 3.6 TB/s/GPU 通信。对于 Mixture-of-Experts (MoE) 模型的 expert parallelism、tensor parallelism 等通信密集型操作，几乎全部在 NVLink 域内完成。只有 data parallelism 的 gradient 同步等相对带宽需求较低的通信走 scale-out 网络。

**第三，NVLink 6 Switch 内置 SHARP in-network compute**。NVLink 6 交换机集成了 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol），可以在网络交换机内部直接执行 all-reduce 等集合通信操作。这将 all-reduce 通信量减少最多 50%，tensor parallelism 执行时间改善最多 20%。这意味着需要走出 NVLink 域的数据量进一步减少。

### 真正让 NIC 接入 NVLink 域的可能路径

虽然当前 NIC 仍通过 PCIe 连接，但 NVIDIA 的架构演进方向非常清晰。有几种可能的实现方式：

**方式一：NIC 芯片集成 NVLink 接口**。未来的 ConnectX 或其继任者直接在硅片上集成 NVLink PHY，成为 NVLink fabric 的一个端点。这样 NIC 的 DMA 引擎可以通过 NVLink 直接访问 GPU HBM，完全绕过 PCIe 和 CPU 内存。

**方式二：NIC 通过 NVLink-C2C 连接到 GPU**。类似于 Grace CPU 通过 NVLink-C2C 连接到 GPU 的方式，NIC 也可以通过 C2C 互联直接与 GPU 对接。

**方式三：NVSwitch 本身具备网络接口能力**。NVLink Switch 芯片已经集成了 SHARP in-network compute，下一步可能直接集成网络收发能力，让 NVSwitch 既是 GPU 间的交换机，又是跨机柜的网络接口。

---

## 总结：架构演进的全景图

```
演进阶段        │  GPU↔NIC 连接     │ GPU↔NIC 带宽   │ 瓶颈在哪里
──────────────┼──────────────────┼───────────────┼──────────────
DGX A100/H100 │ 共享 PCIe 总线    │ ~64 GB/s       │ PCIe Gen5
              │ (经典 GDMA)       │               │
──────────────┼──────────────────┼───────────────┼──────────────
GB200 NVL72   │ GPU←C2C→CPU←PCIe→NIC │ 受限于 PCIe  │ PCIe Gen5
(Blackwell)   │ (C2C 900GB/s,     │ (~64 GB/s)    │ (但NVLink域
              │  PCIe ~64 GB/s)   │               │  内不需要NIC)
──────────────┼──────────────────┼───────────────┼──────────────
Vera Rubin    │ GPU←C2C→CPU←PCIe→NIC │ 受限于 PCIe  │ PCIe Gen6
NVL72         │ (C2C 1.8TB/s,     │ (~128 GB/s)   │ (NVLink域内
              │  PCIe ~128 GB/s)  │               │  通信更充分)
──────────────┼──────────────────┼───────────────┼──────────────
未来           │ GPU←NVLink→NIC    │ TB/s 级别     │ 网络本身
              │ (NIC 直接在       │               │ (PCIe 从数据
              │  NVLink 域内)     │               │  路径上消失)
```

核心思想是：NVIDIA 并没有一步到位地把 NIC 直接塞进 NVLink 域（这需要 NIC 芯片重新设计硅片集成 NVLink PHY，工程量巨大），而是采取了一条务实的演进路线——先把 NVLink scale-up 域做大（从 8 GPU → 72 GPU → 576 GPU），让绝大多数通信密集型操作不需要走 NIC，再逐步提升 scale-out 路径的带宽（PCIe Gen5 → Gen6，NVLink-C2C 900 GB/s → 1.8 TB/s），最终在时机成熟时让 NIC 原生接入 NVLink 域。


以下基于 CANN 社区版 9.0 / 商用版 8.2 官方文档整理，覆盖你能接触到的所有接口及其语义。官方文档入口统一在：[https://www.hiascend.com/document](https://www.hiascend.com/document)

---

## 第一层：框架层（PyTorch torch.distributed + HCCL 后端）

这一层你写 Python 代码，框架自动调用 HCCL。初始化时指定 `backend="hccl"`。

### 1.1 初始化与销毁

`torch.distributed.init_process_group(backend="hccl", ...)` — 初始化分布式进程组，建立 HCCL 通信域。必须在所有集合通信调用之前执行。

`torch.distributed.destroy_process_group()` — 销毁进程组，释放 HCCL 通信域资源。

`torch.distributed.is_hccl_available()` — 检查当前环境是否支持 HCCL 后端。返回布尔值。

`torch.distributed.is_initialized()` — 检查分布式进程组是否已初始化。

### 1.2 集合通信

`torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)` — 对 group 内所有 rank 的 tensor 执行归约操作（默认求和），结果写回每个 rank 的 tensor。这是分布式训练中最核心、调用频率最高的接口，梯度同步几乎全靠它。

`torch.distributed.broadcast(tensor, src, group=None, async_op=False)` — 将 src rank 的 tensor 广播到 group 内所有其他 rank。常用于参数初始化同步。

`torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)` — 将 group 内每个 rank 的 tensor 收集到所有 rank 的 tensor_list 中。tensor_list 长度等于 group size。

`torch.distributed.reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)` — 先对 input_list 做归约，然后将结果分片散发到各 rank。等价于 reduce + scatter。

`torch.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False)` — 归约后结果只发送到 dst rank。其他 rank 的 tensor 不更新。

`torch.distributed.all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)` — 全交换。每个 rank 向每个其他 rank 发送不同的数据。MoE（混合专家）模型中 expert 路由的核心操作。

`torch.distributed.scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)` — src rank 将 scatter_list 中的分片分别发送到各 rank。

`torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)` — 所有 rank 将 tensor 发送到 dst rank 的 gather_list 中。

`torch.distributed.barrier(group=None, async_op=False)` — 同步屏障，阻塞直到 group 内所有 rank 都到达此点。

### 1.3 点对点通信

`torch.distributed.send(tensor, dst, group=None, tag=0)` — 阻塞发送 tensor 到 dst rank。

`torch.distributed.recv(tensor, src=None, group=None, tag=0)` — 阻塞接收 tensor，来自 src rank。

`torch.distributed.isend(tensor, dst, group=None, tag=0)` — 非阻塞发送，返回 Work 对象，可调用 `.wait()` 等待完成。

`torch.distributed.irecv(tensor, src=None, group=None, tag=0)` — 非阻塞接收。

### 1.4 辅助查询

`torch.distributed.get_rank(group=None)` — 获取当前进程在 group 中的 rank 编号。

`torch.distributed.get_world_size(group=None)` — 获取 group 内总 rank 数。

`torch.distributed.get_backend(group=None)` — 返回当前后端名称字符串（“hccl”）。

`torch.distributed.new_group(ranks=None, ...)` — 创建子通信组，用于模型并行、pipeline 并行中的分组通信。

### 1.5 归约操作类型

`torch.distributed.ReduceOp.SUM` — 求和。最常用，梯度同步标配。

`torch.distributed.ReduceOp.PRODUCT` — 求积。

`torch.distributed.ReduceOp.MAX` — 取最大值。

`torch.distributed.ReduceOp.MIN` — 取最小值。

**HCCL 后端在昇腾上的限制**：不支持 `ReduceOp.BAND`、`ReduceOp.BOR`、`ReduceOp.BXOR`（位运算类），这些是 Gloo/NCCL 特有的。

### 1.6 典型使用模式

```python
import torch
import torch.distributed as dist
import torch_npu  # 昇腾 PyTorch 扩展

# 初始化
dist.init_process_group(backend="hccl", init_method="env://")
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)

# 训练循环中的梯度同步
loss.backward()
for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad /= dist.get_world_size()
optimizer.step()

# 清理
dist.destroy_process_group()
```

官方参考：[https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/Distributed接口列表.md](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/Distributed%E6%8E%A5%E5%8F%A3%E5%88%97%E8%A1%A8.md)

---

## 第二层：HCCL C API 层

直接调用 C 接口，获得对通信域、集合通信算子、点对点通信、零拷贝、拓扑查询的完全控制。头文件 `hccl/hccl.h`。以下基于 CANN 9.0 / 8.2 官方文档。

### 2.1 数据类型定义

**HcclResult** — 所有 HCCL 接口的返回值枚举。`HCCL_SUCCESS(0)`=成功，`HCCL_E_PARA(1)`=参数错误，`HCCL_E_PTR(2)`=空指针，`HCCL_E_MEMORY(3)`=内存错误，`HCCL_E_INTERNAL(4)`=内部错误，`HCCL_E_NOT_SUPPORT(5)`=不支持，`HCCL_E_NOT_FOUND(6)`=资源未找到，`HCCL_E_UNAVAIL(7)`=资源不可用，`HCCL_E_SYSCALL(8)`=系统调用错误，`HCCL_E_TIMEOUT(9)`=超时，`HCCL_E_TCP_CONNECT(11)`=TCP连接失败，`HCCL_E_ROCE_CONNECT(12)`=RoCE连接失败，`HCCL_E_TCP_TRANSFER(13)`=TCP传输失败，`HCCL_E_ROCE_TRANSFER(14)`=RoCE传输失败，`HCCL_E_RUNTIME(15)`=Runtime调用失败，`HCCL_E_DRV(16)`=驱动调用失败，`HCCL_E_NETWORK(19)`=网络错误，`HCCL_E_AGAIN(20)`=需重试，`HCCL_E_REMOTE(21)`=远端CQE错误，`HCCL_E_SUSPENDING(22)`=通信域挂起中，`HCCL_E_OOM(24)`=内存不足。

**HcclDataType** — 数据类型枚举。`HCCL_DATA_TYPE_INT8`、`HCCL_DATA_TYPE_INT16`、`HCCL_DATA_TYPE_INT32`、`HCCL_DATA_TYPE_INT64`、`HCCL_DATA_TYPE_FP16`、`HCCL_DATA_TYPE_FP32`、`HCCL_DATA_TYPE_BFP16`。不同硬件平台对 int64 可能有性能劣化，prod 操作不支持 int16/bfp16。

**HcclReduceOp** — 归约操作类型。`HCCL_REDUCE_SUM`（求和）、`HCCL_REDUCE_PROD`（求积）、`HCCL_REDUCE_MAX`（最大值）、`HCCL_REDUCE_MIN`（最小值）。

**HcclComm** — 通信域句柄，不透明类型。所有通信操作都在某个 HcclComm 上下文中执行。

**HcclRootInfo** — Root 节点信息结构体，包含 device IP、device ID 等，用于多机场景下广播给所有 rank 做初始化。

**CANN 9.0 新增数据类型**（控制面/自定义算法相关）：`CommTopo`=通信拓扑描述，`CommLink`=通信链路描述，`CommMem`=通信内存描述，`CommMemType`=内存类型枚举。

### 2.2 通信域管理接口

`HcclGetRootInfo(HcclRootInfo *rootInfo)` — 获取 root rank 的信息，用于后续广播给所有 rank。语义：生成一个包含本 rank 网络标识的 rootInfo 结构体。

`HcclCommInitRootInfo(uint32_t rankSize, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm)` — **多机多卡通信域初始化**。所有 rank 调用此接口，传入相同的 rootInfo（从 root rank 广播而来），各自传入自己的 rank 编号。返回 HcclComm 句柄。这是最常用的初始化方式。

`HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)` — 通过集群信息文件（JSON 格式，描述所有 rank 的 device 信息）初始化通信域。适用于静态集群配置。

`HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config, HcclComm *comm)` — 与上一个类似，但额外传入 HcclCommConfig 配置结构体，可定制通信域行为。

`HcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms)` — **单机多卡通信域初始化**。一个进程统一创建多张卡的通信域，devices[0] 自动作为 root rank。仅支持单机场景。

`HcclCommDestroy(HcclComm comm)` — 销毁通信域，释放所有关联资源。每个 HcclComm 在使用完毕后必须调用。

### 2.3 集合通信接口（数据面）

所有集合通信接口都是**异步提交到 stream**，调用后立即返回，需要 `aclrtSynchronizeStream(stream)` 等待完成。

`HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)` — AllReduce。将所有 rank 的 sendBuf 做归约，结果写入所有 rank 的 recvBuf。count 是数据元素个数（不是字节数）。所有 rank 的 count/dataType/op 必须相同。sendBuf 和 recvBuf 的地址对齐要求：int8→1B，int16/fp16/bfp16→2B，int32/fp32→4B，int64→8B。

`HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)` — Broadcast。root rank 的 buf 广播到所有 rank 的 buf。in-place 操作，root 的 buf 既是输入也是输出。

`HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)` — AllGather。每个 rank 贡献 sendCount 个元素，按 rank ID 顺序拼接，结果写入所有 rank 的 recvBuf。recvBuf 大小 = sendCount × rankSize × sizeof(dataType)。

`HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)` — ReduceScatter。先归约，再将结果平均分片到各 rank。每个 rank 收到 recvCount 个元素。sendBuf 大小 = recvCount × rankSize。

`HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)` — Reduce。与 AllReduce 类似但结果只送到 root rank。非 root rank 的 recvBuf 不更新。

`HcclAlltoAll(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclComm comm, aclrtStream stream)` — AlltoAll。全交换，每个 rank 向每个其他 rank 发送 count 个元素的不同数据。sendBuf 和 recvBuf 大小均 = count × rankSize × sizeof(dataType)。MoE 模型必备。

`HcclAlltoAllV(const void *sendBuf, const uint64_t *sendCounts, const uint64_t *sdispls, HcclDataType sendType, void *recvBuf, const uint64_t *recvCounts, const uint64_t *rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)` — 变长 AlltoAll。每对 rank 之间可以交换不同大小的数据。sendCounts[i] 表示发给 rank i 的元素数，sdispls[i] 表示在 sendBuf 中的偏移。

`HcclReduceScatterV(...)` — 变长 ReduceScatter（CANN 9.0 新增）。各 rank 接收不等长的归约结果分片。

`HcclBarrier(HcclComm comm, aclrtStream stream)` — 同步屏障。阻塞直到通信域内所有 rank 都到达。

### 2.4 点对点通信接口

`HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)` — 向 destRank 发送 count 个元素。异步提交。

`HcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)` — 从 srcRank 接收 count 个元素。必须与对端的 HcclSend 配对。

`HcclBatchSendRecv(HcclSendRecvItemDef *sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)` — 批量收发。一次调用同时提交多个 Send/Recv 操作，减少调度开销。

### 2.5 零拷贝功能（高级）

零拷贝允许注册用户内存，HCCL 直接在该内存上做通信，避免额外的 memcpy。

`HcclCommRegMem(HcclComm comm, void *ptr, uint64_t size)` — 向通信域注册一段 Device 内存。注册后该内存可被 HCCL 直接用于零拷贝通信。

`HcclCommDeregMem(HcclComm comm, void *ptr)` — 取消注册。释放内存前必须先取消注册。

### 2.6 拓扑查询接口（CANN 9.0 控制面）

用于自定义算法开发时获取集群拓扑信息。

`HcclGetRankId(HcclComm comm, uint32_t *rankId)` — 获取当前 rank 在通信域中的 ID。

`HcclGetRankSize(HcclComm comm, uint32_t *rankSize)` — 获取通信域内总 rank 数。

`HcclRankGraphGetLayers(HcclComm comm, uint32_t *layerNum)` — 获取拓扑图层级数（例如 server 内部一层 + server 间一层 = 2）。

### 2.7 异常处理

`HcclCommGetAsyncError(HcclComm comm, HcclResult *asyncError)` — 查询通信域的异步错误状态。通信操作在 stream 中异步执行，错误不会立即抛出，需要主动查询。

`HcclCommAbort(HcclComm comm)` — 中止通信域。当检测到不可恢复的错误时调用，中止所有进行中的通信操作。

### 2.8 典型使用流程（C）

```c
#include "hccl/hccl.h"
#include "acl/acl.h"

// 1. 初始化 ACL 和设备
aclInit(nullptr);
aclrtSetDevice(deviceId);

// 2. 获取 rootInfo 并广播（root rank 生成，其他 rank 通过 TCP/MPI 接收）
HcclRootInfo rootInfo;
if (rank == 0) HcclGetRootInfo(&rootInfo);
// ... 广播 rootInfo 到所有 rank ...

// 3. 初始化通信域
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, rank, &hcclComm);

// 4. 分配 Device 内存
void *sendBuf = nullptr, *recvBuf = nullptr;
aclrtMalloc(&sendBuf, count * sizeof(float), ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, count * sizeof(float), ACL_MEM_MALLOC_HUGE_ONLY);

// 5. 创建 stream
aclrtStream stream;
aclrtCreateStream(&stream);

// 6. 执行 AllReduce
HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, 
              HCCL_REDUCE_SUM, hcclComm, stream);
aclrtSynchronizeStream(stream);

// 7. 清理
aclrtFree(sendBuf);
aclrtFree(recvBuf);
aclrtDestroyStream(stream);
HcclCommDestroy(hcclComm);
aclrtResetDevice(deviceId);
aclFinalize();
```

官方完整 API 参考：[https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/hcclapiref/hcclcpp_07_0021.html](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/hcclapiref/hcclcpp_07_0021.html)

---

## 第三层：AscendCL（计算与资源管理）

AscendCL 是昇腾计算的统一编程框架，与 HCCL 平行，覆盖设备管理、内存管理、Stream 管理、模型推理、算子执行、数据类型转换等。头文件 `acl/acl.h`。

### 3.1 全局初始化/去初始化

`aclInit(const char *configPath)` — 初始化 AscendCL 运行环境。configPath 传 nullptr 使用默认配置，或传 JSON 配置文件路径。整个进程只调用一次。

`aclFinalize()` — 去初始化 AscendCL，释放所有全局资源。进程退出前调用。

### 3.2 设备管理

`aclrtSetDevice(int32_t deviceId)` — 设置当前线程使用的 Device（NPU 卡）。隐式创建默认 Context。

`aclrtResetDevice(int32_t deviceId)` — 重置 Device，释放该 Device 上的所有资源（stream、内存等）。

`aclrtGetDevice(int32_t *deviceId)` — 获取当前线程绑定的 Device ID。

`aclrtGetDeviceCount(uint32_t *count)` — 获取系统中可用的 Device 总数。

`aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId)` — 查询两个 Device 之间是否支持 P2P 内存访问。

`aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags)` — 开启到目标 Device 的 P2P 访问能力。

### 3.3 Context 管理

`aclrtCreateContext(aclrtContext *context, int32_t deviceId)` — 显式创建 Context。一个 Device 可以有多个 Context。

`aclrtDestroyContext(aclrtContext context)` — 销毁 Context。

`aclrtSetCurrentContext(aclrtContext context)` — 切换当前线程的 Context。

`aclrtGetCurrentContext(aclrtContext *context)` — 获取当前 Context。

### 3.4 Stream 管理

Stream 是任务队列，控制异步操作的执行顺序。

`aclrtCreateStream(aclrtStream *stream)` — 创建一个 Stream。

`aclrtDestroyStream(aclrtStream stream)` — 销毁 Stream。

`aclrtSynchronizeStream(aclrtStream stream)` — 阻塞等待 Stream 中所有已提交任务完成。HCCL 集合通信异步提交后，靠这个接口等结果。

`aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)` — 让 stream 等待 event 被触发后再继续执行后续任务。用于多 Stream 之间的同步。

`aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)` — 带超时的同步等待。

### 3.5 Event 管理

Event 用于 Stream 之间的精确同步和计时。

`aclrtCreateEvent(aclrtEvent *event)` — 创建 Event。

`aclrtDestroyEvent(aclrtEvent event)` — 销毁 Event。

`aclrtRecordEvent(aclrtEvent event, aclrtStream stream)` — 在 stream 中记录一个时间戳。

`aclrtQueryEvent(aclrtEvent event, aclrtEventRecordedStatus *status)` — 查询 Event 是否已被记录。

`aclrtEventElapsedTime(float *ms, aclrtEvent start, aclrtEvent end)` — 计算两个 Event 之间的耗时（毫秒）。性能 profiling 必备。

### 3.6 内存管理

`aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)` — 在 Device 上分配内存，首地址 64 字节对齐，size 向上对齐到 32 字节整数倍再加 32 字节。policy 可选 `ACL_MEM_MALLOC_HUGE_FIRST`（优先大页）、`ACL_MEM_MALLOC_HUGE_ONLY`（仅大页）、`ACL_MEM_MALLOC_NORMAL_ONLY`（仅普通页）。

`aclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy)` — 与 aclrtMalloc 类似但只做 32 字节对齐，不额外加 32 字节。适合自行管理大块内存的场景。

`aclrtFree(void *devPtr)` — 释放 Device 内存。

`aclrtMallocHost(void **hostPtr, size_t size)` — 在 Host 上分配页锁定内存（pinned memory）。与 Device 之间的 DMA 传输效率更高。

`aclrtFreeHost(void *hostPtr)` — 释放 Host 页锁定内存。

`aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)` — 同步内存拷贝。kind 指定方向：`ACL_MEMCPY_HOST_TO_HOST`、`ACL_MEMCPY_HOST_TO_DEVICE`、`ACL_MEMCPY_DEVICE_TO_HOST`、`ACL_MEMCPY_DEVICE_TO_DEVICE`。

`aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)` — 异步内存拷贝，提交到 stream。

`aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)` — Device 内存初始化。建议在使用 aclrtMalloc 分配的内存前先清零。

`aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream)` — 异步内存初始化。

`aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total)` — 查询 Device 内存使用情况。attr 可选 `ACL_HBM_MEM`（HBM）或 `ACL_DDR_MEM`（DDR）。

### 3.7 模型管理（推理）

`aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId)` — 从 .om 文件加载离线模型到 Device，返回 modelId。

`aclmdlLoadFromMem(const void *model, size_t modelSize, uint32_t *modelId)` — 从内存加载模型。

`aclmdlLoadFromFileWithMem(const char *modelPath, uint32_t *modelId, void *workPtr, size_t workSize, void *weightPtr, size_t weightSize)` — 加载模型到用户预分配的 workspace 和 weight 内存。

`aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output)` — **同步执行模型推理**。阻塞直到推理完成。

`aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream)` — 异步执行模型推理。

`aclmdlUnload(uint32_t modelId)` — 卸载模型，释放 Device 上的模型资源。

`aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId)` — 获取模型描述信息（输入输出个数、维度、数据类型等）。

`aclmdlGetNumInputs(aclmdlDesc *modelDesc)` — 获取模型输入个数。

`aclmdlGetNumOutputs(aclmdlDesc *modelDesc)` — 获取模型输出个数。

`aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index)` — 获取指定输入的内存大小。

`aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index)` — 获取指定输出的内存大小。

### 3.8 数据集构造（输入/输出 buffer 管理）

`aclmdlCreateDataset()` — 创建空的 Dataset 对象。

`aclmdlDestroyDataset(const aclmdlDataset *dataset)` — 销毁 Dataset。

`aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer)` — 向 Dataset 添加一个 DataBuffer。

`aclCreateDataBuffer(void *data, size_t size)` — 创建 DataBuffer，封装 Device 上的内存指针和大小。

`aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)` — 销毁 DataBuffer（不释放底层内存）。

### 3.9 算子加载与执行（单算子模式）

`aclopCompileAndExecute(const char *opType, int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[], int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[], const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileType, const char *opPath, aclrtStream stream)` — 编译并执行单个算子。开发和调试自定义算子时使用。

`aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format)` — 创建张量描述符。

`aclDestroyTensorDesc(const aclTensorDesc *desc)` — 销毁张量描述符。

### 3.10 数据类型与格式

**aclDataType**：`ACL_FLOAT`(0)、`ACL_FLOAT16`(1)、`ACL_INT8`(2)、`ACL_INT32`(3)、`ACL_UINT8`(4)、`ACL_INT16`(6)、`ACL_UINT16`(7)、`ACL_UINT32`(8)、`ACL_INT64`(9)、`ACL_UINT64`(10)、`ACL_DOUBLE`(11)、`ACL_BOOL`(12)、`ACL_BF16`(27) 等。

**aclFormat**：`ACL_FORMAT_NCHW`(0)、`ACL_FORMAT_NHWC`(1)、`ACL_FORMAT_ND`(2)、`ACL_FORMAT_NC1HWC0`(3) 等。昇腾 NPU 内部常用 NC1HWC0（5D 格式），框架层通常自动处理格式转换。

### 3.11 典型推理流程

```c
// 1. 初始化
aclInit(nullptr);
aclrtSetDevice(0);
aclrtCreateStream(&stream);

// 2. 加载模型
uint32_t modelId;
aclmdlLoadFromFile("model.om", &modelId);

// 3. 查询模型信息
aclmdlDesc *desc = aclmdlCreateDesc();
aclmdlGetDesc(desc, modelId);
size_t inputSize = aclmdlGetInputSizeByIndex(desc, 0);
size_t outputSize = aclmdlGetOutputSizeByIndex(desc, 0);

// 4. 分配内存
void *inputDev, *outputDev;
aclrtMalloc(&inputDev, inputSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&outputDev, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 5. 准备输入数据（Host→Device）
aclrtMemcpy(inputDev, inputSize, hostData, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);

// 6. 构造 Dataset
aclmdlDataset *inputDataset = aclmdlCreateDataset();
aclmdlAddDatasetBuffer(inputDataset, aclCreateDataBuffer(inputDev, inputSize));
aclmdlDataset *outputDataset = aclmdlCreateDataset();
aclmdlAddDatasetBuffer(outputDataset, aclCreateDataBuffer(outputDev, outputSize));

// 7. 执行推理
aclmdlExecute(modelId, inputDataset, outputDataset);

// 8. 取回结果（Device→Host）
aclrtMemcpy(hostOutput, outputSize, outputDev, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

// 9. 清理（省略 Dataset/DataBuffer 销毁）
aclmdlUnload(modelId);
aclrtFree(inputDev);
aclrtFree(outputDev);
aclrtDestroyStream(stream);
aclrtResetDevice(0);
aclFinalize();
```

官方完整 API 参考：[https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclcppdevg_03_0095.html](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclcppdevg_03_0095.html)

---

## 三层之间的关系总结

框架层（PyTorch `dist.all_reduce`）内部调用 HCCL C API（`HcclAllReduce`），而 HCCL 的内存分配和 stream 管理又依赖 AscendCL（`aclrtMalloc`、`aclrtCreateStream`、`aclrtSynchronizeStream`）。所以这三层不是完全独立的：HCCL 和 AscendCL 在 C 层面是**相互配合**的——HCCL 管通信，AscendCL 管计算和资源，二者共用同一个 Device 上下文和 Stream。

当你只做模型训练时，框架层帮你封装了一切。当你需要性能调优、自定义通信算法或开发推理应用时，直接操作第二层和第三层就是你的工作界面。
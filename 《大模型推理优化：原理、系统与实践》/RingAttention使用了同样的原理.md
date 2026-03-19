对，完全一样的原理。

Ring Attention 只是把这个"一块一块来，边算边修正"的思路**搬到了多机多卡**的场景上。

## 具体来说

假设你有 4 张 GPU，序列太长，一张卡放不下完整的 KV。于是把 KV 切成 4 份，每张卡各持有一份：

$$\text{GPU}_0: K_0, V_0 \quad \text{GPU}_1: K_1, V_1 \quad \text{GPU}_2: K_2, V_2 \quad \text{GPU}_3: K_3, V_3$$

每张卡都持有完整的 Q（或者 Q 的一部分），需要和**所有的** K、V 做 attention。

Ring Attention 的做法是：把 4 张卡组成一个**环**，KV block 沿着环**传递**。每一步，每张卡拿到一个 KV block，做一次局部 attention，然后把这个 KV block 传给下一张卡，同时从上一张卡接收新的 KV block。

$$\text{Step 1: GPU}_0 \text{ 算 } K_0V_0, \quad \text{GPU}_1 \text{ 算 } K_1V_1, \quad \dots$$

$$\text{Step 2: KV 往右传一步，GPU}_0 \text{ 算 } K_3V_3, \quad \text{GPU}_1 \text{ 算 } K_0V_0, \quad \dots$$

$$\text{Step 3: 再传一步} \dots$$

$$\text{Step 4: 再传一步} \dots$$

转一整圈之后，每张卡就见过了所有的 KV block。

而每一步拿到新的 KV block 时，做的事情和 FlashAttention 一模一样——**更新 $m$，修正 $\ell$，缩放 $O$，加上新贡献**。

## 和 FlashAttention 的关系

它们唯一的区别就是**block 从哪来**：

||block 从哪来|增量修正的数学|
|---|---|---|
|FlashAttention|从**同一张卡的 HBM** 分批搬进 SRAM|online softmax|
|PagedAttention|从**不连续的内存页**逐页取出|online softmax|
|Ring Attention|从**相邻 GPU** 通过网络传过来|online softmax|

底层数学完全一样，只是数据的来源不同。FlashAttention 是卡内搬数据，Ring Attention 是卡间传数据。Ring Attention 还有一个巧妙之处：**通信和计算可以重叠**——在算当前 block 的 attention 时，下一个 block 已经在后台传输了，这样通信开销就被隐藏掉了。

所以你的直觉是对的，从 online softmax 到 FlashAttention 到 PagedAttention 到 Ring Attention，分块做完整 attention 的数学内核**始终是同一个东西**。
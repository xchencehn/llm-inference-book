

# 第7章 投机解码（Speculative Decoding）


自回归语言模型的解码过程有一个根本性的矛盾：模型每次只能生成一个 Token，而每次生成都需要完整地执行一遍前向传播。在 Decode 阶段，由于批量大小极小（往往只有一个 Token），GPU 的数千个计算核心大部分时间都在等待显存数据搬运完成——计算单元严重空闲，访存带宽成为瓶颈。换言之，模型的庞大算力在逐 Token 生成时被严重浪费了。

能否让模型一次"看到"并验证多个候选 Token，从而在一次前向传播中推进多个解码步？这正是投机解码（Speculative Decoding）试图回答的问题。它的核心直觉朴素而深刻：用一个运行速度极快的小模型先"猜"出若干候选 Token，再用大模型一次性并行验证这些猜测。如果猜对了，就相当于大模型一步走了多步；即使猜错了，也能在错误位置及时纠正，不会引入任何精度损失。

本章将系统阐述投机解码的理论基础、各类 Draft 策略的设计思想、主流推理引擎中的工程实现，以及效率分析与调优方法。

---

## 7.1 投机解码的核心思想：以小博大

### 7.1.1 Draft-Then-Verify 范式

投机解码的思想可以追溯到计算机体系结构中经典的"投机执行"（Speculative Execution）概念。在现代 CPU 中，分支预测器会提前猜测分支走向并预先执行指令；如果预测正确就直接采用结果，如果错误则回滚重来。投机解码将这一范式移植到了自回归语言模型的 Token 生成过程中。

**标准自回归解码的瓶颈。** 在标准解码过程中，给定已经生成的前缀序列 x1​,x2​,…,xt​，模型需要计算条件概率分布 p(xt+1​∣x1​,…,xt​)，从中采样得到 xt+1​，然后将 xt+1​ 追加到序列中，再重复这一过程以生成 xt+2​。生成 K 个 Token 就需要串行执行 K 次前向传播。如第2章所分析的，Decode 阶段的每次前向传播都是访存密集型操作：对于一个参数量为 N 的模型，每生成一个 Token 至少需要从显存中读取约 2N 字节的权重数据（FP16 精度下），而实际执行的浮点计算量相对于搬运的数据量而言极为有限。这意味着 GPU 的计算吞吐远未饱和，大量 Tensor Core 处于空闲状态。

**投机解码的两阶段框架。** 投机解码将每一轮解码拆分为两个阶段：

第一阶段为**草稿生成（Draft）**。一个轻量级的 Draft 模型 Mq​（其输出分布记为 q）快速地自回归生成 γ 个候选 Token x~t+1​,x~t+2​,…,x~t+γ​。Draft 模型的参数量远小于目标模型，因此每次前向传播速度极快。虽然这 γ 个 Token 仍然是 Draft 模型逐个生成的，但由于 Draft 模型很小，总耗时远低于目标模型生成同样数量的 Token。

第二阶段为**并行验证（Verify）**。将这 γ 个候选 Token 连同原始前缀一起输入目标大模型 Mp​（其输出分布记为 p）。关键在于，验证过程可以并行执行：大模型在一次前向传播中同时计算所有 γ 个位置的条件概率分布 p(xt+i​∣x1​,…,xt+i−1​)（i=1,2,…,γ）。这与 Prefill 阶段处理多个 Token 的方式相同，属于计算密集型操作，能够充分利用 GPU 的并行计算能力。随后，通过一种专门设计的接受-拒绝采样算法，从前到后逐个检查每个候选 Token：如果 x~t+i​ 被接受，则继续检查下一个；一旦某个位置被拒绝，就从大模型在该位置的修正分布中重新采样一个 Token，并丢弃后续所有候选。

假设在一轮投机解码中，前 α 个候选 Token 都被接受（0≤α≤γ），则大模型实际推进了 α+1 个 Token（α 个接受的加上 1 个从拒绝位置或最后位置额外采样的）。而这只花费了 Draft 模型 γ 次前向传播加上大模型 1 次前向传播的时间。当 α 较大时，效率提升显著。

为使这一过程更加直观，我们用一个具体的例子来说明。假设目标模型是一个 70B 参数的大模型，Draft 模型是一个 1B 参数的小模型，投机长度 γ=5。在一轮投机解码中：

（1）Draft 模型快速生成 5 个候选 Token，例如 "The weather today is very"。由于 Draft 模型很小，这 5 步总共可能只耗时 5 毫秒。

（2）目标大模型将这 5 个 Token 作为一个完整序列输入，执行一次前向传播。由于一次处理 5 个 Token，这与处理一个短的 Prefill 类似，耗时可能约 20 毫秒（远小于逐个生成 5 个 Token 所需的 5×20=100 毫秒）。

（3）验证过程发现前 4 个 Token "The weather today is" 与大模型的分布一致，被全部接受；但第 5 个位置大模型更倾向于 "nice" 而非 "very"，于是拒绝 "very"，改为采样 "nice"。

最终，一轮投机解码用约 25 毫秒生成了 5 个 Token，而标准解码需要约 100 毫秒。这就是投机解码"以小博大"的精髓。

**形式化描述。** 我们用更精确的符号来描述这一过程。设当前已有序列 x1:t​，投机长度为 γ。

在 Draft 阶段，对于 i=1,2,…,γ，Draft 模型计算 qi​(⋅)=Mq​(⋅∣x1:t​,x~t+1​,…,x~t+i−1​)，并采样 x~t+i​∼qi​(⋅)。

在 Verify 阶段，目标模型并行计算所有位置的分布 pi​(⋅)=Mp​(⋅∣x1:t​,x~t+1​,…,x~t+i−1​)（i=1,2,…,γ+1）。注意这里计算到 γ+1 个位置，因为即使所有候选都被接受，大模型也会在最后一个位置额外产生一个新 Token 的分布。

然后执行接受-拒绝判定。对于每个位置 i=1,2,…,γ，生成一个均匀随机数 r∼Uniform(0,1)。如果 r<min(1,qi​(x~t+i​)pi​(x~t+i​)​)，则接受 x~t+i​，继续验证下一个位置。否则，拒绝 x~t+i​，从修正分布 pi′​(⋅)=norm(max(0,pi​(⋅)−qi​(⋅))) 中采样一个替代 Token，丢弃位置 i 之后的所有候选，本轮结束。如果所有 γ 个候选都被接受，则从 pγ+1​(⋅) 中直接采样一个额外 Token。

**投机解码的本质洞察。** 投机解码之所以有效，根植于以下几个关键观察。

其一，大语言模型生成的 Token 中有相当大比例是"容易预测"的。常见的功能词、固定搭配、语法结构等，即使是小模型也能准确预测。真正需要大模型"深思熟虑"的只是序列中的少数关键 Token。投机解码正是利用了这种预测难度的不均匀分布。

其二，Transformer 的 Prefill（多 Token 并行处理）比逐 Token 的 Decode 在硬件利用率上高效得多。验证 γ 个 Token 的耗时远小于逐个生成 γ 个 Token 的耗时，这一"验证比生成快"的不对称性是投机解码获得加速的物理基础。

其三，通过精心设计的接受-拒绝采样机制，投机解码可以严格保证输出分布与目标模型完全一致，不引入任何近似误差。这一无损性质是投机解码相比许多其他加速方法（如量化、剪枝）的重要优势。

### 7.1.2 无损性证明：接受-拒绝采样保证分布一致

投机解码最精妙的设计在于其无损性保证：无论 Draft 模型的质量如何，最终生成的 Token 序列始终严格服从目标模型 Mp​ 的分布。这一性质通过一种修改版的接受-拒绝采样（Modified Rejection Sampling）来实现，由 Leviathan 等人和 Chen 等人在 2023 年同时独立提出。

**经典接受-拒绝采样的回顾。** 在概率论与统计学中，当我们希望从一个目标分布 p 中采样，但直接采样困难时，可以利用一个易于采样的提议分布 q。经典的接受-拒绝采样要求找到一个常数 M 使得对所有 x 都有 p(x)≤M⋅q(x)，然后从 q 中采样候选，以概率 M⋅q(x)p(x)​ 接受。但这种方法在被拒绝时需要重新从 q 采样，无法保证单次采样的成功率。

**投机解码的修改版采样。** 投机解码对经典方法进行了关键修改：当候选 Token x~ 被拒绝时，不是重新从 q 采样，而是从一个精心构造的修正分布中采样。这保证了无论接受还是拒绝，最终输出都服从目标分布 p。

具体而言，对于单个位置的采样过程如下。给定目标分布 p 和 Draft 分布 q，以及 Draft 模型采样得到的候选 x~∼q：

以概率 min(1,q(x~)p(x~)​) 接受 x~；否则，从修正分布 p′ 中采样，其中：

p′(x)=∑x′​max(0,p(x′)−q(x′))max(0,p(x)−q(x))​

**定理 7.1（投机采样的无损性）。** 上述采样过程产生的 Token 服从目标分布 p。

**证明。** 我们计算最终输出为某个特定 Token x 的概率。设 A 表示候选被接受的事件，R 表示候选被拒绝的事件。

首先考虑接受路径。Token x 通过接受路径被输出的概率为：

Pr[x via accept]=q(x)⋅min(1,q(x)p(x)​)=min(q(x),p(x))

接下来考虑拒绝路径。总的拒绝概率为：

Pr[R]=x′∑​q(x′)⋅(1−min(1,q(x′)p(x′)​))=x′∑​max(0,q(x′)−p(x′))

利用 ∑x′​p(x′)=∑x′​q(x′)=1 这一事实，可以验证：

x′∑​max(0,q(x′)−p(x′))=x′∑​max(0,p(x′)−q(x′))

这个等式成立是因为 ∑x′​[q(x′)−p(x′)]=0，所以正差和负差之和相等。

因此，Token x 通过拒绝路径被输出的概率为：

Pr[x via reject]=Pr[R]⋅p′(x)=x′∑​max(0,q(x′)−p(x′))⋅∑x′′​max(0,p(x′′)−q(x′′))max(0,p(x)−q(x))​

=max(0,p(x)−q(x))

最终，Token x 的总输出概率为：

Pr[x]=min(q(x),p(x))+max(0,p(x)−q(x))=p(x)

最后一步等式可以分两种情况验证：当 p(x)≤q(x) 时，min(q(x),p(x))=p(x)，max(0,p(x)−q(x))=0，故总概率为 p(x)；当 p(x)>q(x) 时，min(q(x),p(x))=q(x)，max(0,p(x)−q(x))=p(x)−q(x)，总概率为 q(x)+p(x)−q(x)=p(x)。证毕。

**多步投机的无损性。** 上述证明针对的是单个位置。对于连续 γ 个位置的投机解码，无损性通过归纳法保证。由于第一个位置的输出严格服从 p1​，条件于第一个位置的输出，第二个位置的验证过程等价于在正确的条件下执行同样的采样过程，因此第二个位置也服从正确的条件分布。以此类推，整个序列的联合分布与目标模型逐 Token 生成的联合分布完全相同。

值得注意的是，一旦某个位置被拒绝，后续所有候选 Token 都必须被丢弃，这是因为后续候选是在 Draft 模型基于错误前缀条件生成的，其分布已经偏离了目标模型在正确前缀下的条件分布。

**贪心解码的特例。** 在贪心解码（temperature = 0）的情况下，投机解码的验证过程简化为直接比较：如果 argmaxpi​(⋅)=x~t+i​，则接受；否则拒绝并输出 argmaxpi​(⋅)。这是因为当目标分布退化为确定性分布时，接受-拒绝采样退化为简单的匹配检查。

**温度参数的处理。** 当使用带温度的采样（temperature =1）时，需要对 p 和 q 在相同温度下进行调整后再执行接受-拒绝判定。具体而言，应使用经过温度缩放后的分布 pT​ 和 qT​ 来替换上述公式中的 p 和 q，其中 pT​(x)∝p(x)1/T，qT​ 同理。这样才能保证最终采样分布等于温度调整后的目标分布。

**对 Draft 模型质量的容错性。** 无损性保证的一个重要推论是：投机解码的正确性不依赖于 Draft 模型的质量。即使 Draft 模型与目标模型的分布差异很大，输出分布仍然精确等于目标模型的分布。Draft 模型质量只影响接受率，进而影响加速比——差的 Draft 模型会导致频繁拒绝，加速效果不佳甚至可能因为 Draft 开销而更慢，但绝不会影响输出质量。这一性质使得投机解码成为一种"安全"的加速技术，可以在不牺牲任何精度的前提下进行部署。

---

## 7.2 Draft 模型的选择策略

投机解码的加速效果在很大程度上取决于 Draft 机制的设计：Draft 模型越快、越准确，加速比就越高。围绕如何构建高效的 Draft 机制，研究社区提出了多种截然不同的方案，大致可分为独立 Draft 模型、自投机（Self-Speculative）、以及基于额外预测头的方法。

### 7.2.1 独立小模型（Speculative Decoding 原论文）

最直接的 Draft 策略是使用一个与目标模型同族但参数量小得多的独立模型。这是由 Leviathan 等人（"Fast Inference from Transformers via Speculative Decoding"，2023）和 Chen 等人（"Accelerating Large Language Model Decoding with Speculative Sampling"，2023）同时独立提出的原始方案。

**方案描述。** 选择一个小模型作为 Draft 模型，例如当目标模型为 LLaMA-2-70B 时，可选用 LLaMA-2-7B 作为 Draft 模型。两个模型共享相同的词表，这是必要条件——否则 Draft 模型产生的 Token 无法在目标模型的概率分布中查找对应概率进行验证。Draft 模型独立进行自回归解码，生成 γ 个候选 Token，然后送入目标模型验证。

**优势与局限。** 独立小模型方案的优势在于概念简单、实现直接，且 Draft 模型可以独立于目标模型进行优化和部署。此外，对于同一模型家族中存在多种规模的情况（如 LLaMA 系列有 7B、13B、70B 等规模），Draft 模型可以直接选用现成的小模型，无需额外训练。

但这一方案面临几个固有挑战。首先是 Draft 模型的选择问题：Draft 模型太小则接受率低，太大则 Draft 本身耗时过长，需要在速度和接受率之间仔细权衡。经验上，Draft 模型的参数量通常为目标模型的 1/10 至 1/50。其次，Draft 模型需要单独的显存来存储参数和 KV Cache，这在显存紧张的场景下是一个非平凡的额外开销。对于一个 70B 目标模型加上一个 7B Draft 模型的配置，Draft 模型的参数显存约占额外的 10%，而其 KV Cache 虽然较小但仍然需要管理。最后，Draft 模型与目标模型的分布对齐程度受限于模型容量差异，特别是在需要领域专业知识或复杂推理的 Token 位置，小模型的预测往往偏差较大。

**投机长度 γ 的选择。** 投机长度是一个需要调优的超参数。γ 越大，每轮成功时推进的 Token 数越多，但同时 Draft 阶段的耗时也线性增长，且后续候选 Token 被拒绝的概率累积上升。设单个位置的平均接受率为 α，则 γ 个候选中期望被接受的数量为 1−α1−αγ​（此公式将在 7.4 节详细推导）。实践中，γ 通常取 3 到 7 之间。一些实现还引入了自适应 γ 的策略，根据近期接受率的统计动态调整投机长度。

### 7.2.2 Self-Speculative：模型自草稿（层跳跃、Early Exit）

独立 Draft 模型需要额外的参数存储，且可能与目标模型的分布差异较大。一个自然的问题是：能否让目标模型自身充当 Draft 模型？这催生了"自投机"（Self-Speculative）系列方法，其核心思想是通过减少目标模型的计算量（如跳过部分层或提前退出）来获得一个快速但质量略低的 Draft 版本。

**层跳跃（Layer Skipping）。** Draft 阶段只执行目标模型的部分层。例如，对于一个 80 层的模型，Draft 时可以只执行其中的 20 层（每隔 4 层执行一层，或只执行前 20 层、后 20 层等）。这种方式不需要额外参数，且 Draft 模型与目标模型共享权重，天然保证了词表一致和风格相似。Zhang 等人提出的 Draft & Verify 方案采用了这一思路，通过在目标模型的特定层位置插入辅助的轻量级分类头，实现 Early Exit。

**Early Exit。** 在 Transformer 的逐层前向传播过程中，某些 Token 可能在中间层就已经被"确定"——即后续层对这些 Token 的表示变化很小。Early Exit 利用这一观察，在每一层后通过一个辅助分类器判断当前表示是否已经足够稳定，如果是则提前输出预测结果，不再继续后续层的计算。在投机解码框架中，Early Exit 产生的预测作为 Draft Token，再由完整模型进行验证。

**Self-Speculative Decoding。** 这一方法将自投机的思想进一步系统化。在 Draft 阶段，模型跳过部分层（skip sublayers），同时使用贝叶斯优化等方法确定最优的跳层策略，以在 Draft 速度和接受率之间取得平衡。Verify 阶段则运行完整模型。整个过程只使用一套参数，显存开销几乎不增加。

**自投机方法的技术挑战。** 首先是 KV Cache 的管理复杂性。如果 Draft 阶段和 Verify 阶段使用的层不同，Draft 阶段产生的 KV Cache 在 Verify 阶段可能无法直接复用，或者需要特殊的 KV Cache 管理策略来处理"不完整"的 KV Cache。其次，跳层策略需要针对具体模型进行调优，不同模型、不同任务下的最优跳层方案可能不同。最后，自投机方法的加速潜力受限于跳层带来的速度提升幅度——跳过一半的层最多只能提速约 2 倍，这与使用独立小模型（可能带来 10 倍以上的速度差异）相比，Draft 阶段的速度优势有限。

### 7.2.3 Medusa：多头并行草稿

Medusa 由 Cai 等人（2024）提出，采用了一种与前述方法截然不同的思路：不使用自回归的 Draft 模型，而是在目标模型的顶部附加多个并行的预测头，每个头同时预测未来不同位置的 Token。

**架构设计。** Medusa 在目标模型最后一层隐藏状态的基础上，添加 K 个额外的分类头（通常称为 Medusa heads）。原始的语言模型头（LM head）预测下一个 Token xt+1​，而第 k 个 Medusa head 预测位置 xt+1+k​ 的 Token（k=1,2,…,K）。每个 Medusa head 通常由一两层残差连接的前馈网络加上一个共享的词嵌入投影层组成，参数量很小。

这种设计的关键优势在于：所有 Medusa heads 的预测都在同一次前向传播中完成，不需要额外的串行 Draft 步骤。目标模型只需执行一次前向传播，就能同时获得当前位置的预测以及未来 K 个位置的候选预测。

**训练方法。** Medusa heads 需要通过微调来训练。原论文提出了两种训练方案：Medusa-1 冻结原始模型参数，只训练 Medusa heads；Medusa-2 同时微调原始模型和 Medusa heads。Medusa-1 的优势在于不改变原始模型的能力，但预测精度受限；Medusa-2 可以获得更高的接受率，但需要更多的训练资源，且可能轻微影响原始模型的性能。

**Tree-Based Verification。** 由于 K 个 Medusa heads 各自独立预测不同位置的 Token，它们的预测组合形成了一棵候选树而非一条候选链。例如，如果第一个 head 给出了 top-2 的候选 {a1​,a2​}，第二个 head 给出了 top-2 的候选 {b1​,b2​}，那么可能的候选序列有 {(a1​,b1​),(a1​,b2​),(a2​,b1​),(a2​,b2​)}。Medusa 使用基于树结构的注意力掩码（tree attention mask），在一次前向传播中同时验证多条候选路径。验证完成后，选择被接受 Token 最多的路径作为输出。

**效率特点。** Medusa 的 Draft 开销几乎为零，因为 Draft 预测与正常的前向传播融为一体。然而，由于各 head 的预测是独立的（每个 head 只看到当前时刻的隐藏状态，没有利用前面位置 Draft Token 的信息），预测精度通常低于自回归 Draft。特别是对于第 k 个 head（k 较大时），预测准确率下降显著，因为它需要在不知道前面 Token 的情况下预测更远的未来。

### 7.2.4 EAGLE / EAGLE-2：特征层投机

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）由 Li 等人（2024）提出，它在 Medusa 的基础上引入了一个关键改进：在特征空间（而非 Token 空间）进行自回归 Draft，从而在保持 Draft 开销较低的同时获得更高的预测精度。

**核心设计思想。** EAGLE 观察到，Transformer 模型倒数第二层的特征（hidden states）比最终输出的 Token 概率分布更容易预测。这是因为相邻 Token 的隐藏状态在高维空间中往往具有较强的连续性和可预测性，而从隐藏状态到 Token 分布的映射（LM head）是一个多对一的过程，会放大预测的不确定性。

基于这一洞察，EAGLE 设计了一个轻量级的自回归特征预测网络（通常是单层 Transformer 层），工作在特征空间而非 Token 空间。具体过程如下：

第一步，EAGLE 网络接收目标模型在当前时刻的隐藏状态和当前 Token 的嵌入作为输入，预测下一个时刻的隐藏状态。

第二步，将预测的隐藏状态通过目标模型共享的 LM head 得到下一个 Token 的概率分布，采样得到 Draft Token。

第三步，将该 Draft Token 的嵌入和预测的隐藏状态作为 EAGLE 网络的下一步输入，继续自回归地预测后续的隐藏状态和 Token。

如此反复，EAGLE 可以快速生成一条 Draft Token 序列。由于 EAGLE 网络只有一层 Transformer 层，每步的计算开销远小于完整的目标模型。

**与 Medusa 的对比。** EAGLE 与 Medusa 的关键区别在于 EAGLE 的 Draft 过程是自回归的——每一步的预测都依赖于前一步的结果。这使得 EAGLE 能够建模 Token 之间的依赖关系，从而在较远的未来位置仍然保持较高的预测精度。实验表明，EAGLE 的接受率显著高于 Medusa，特别是在投机长度较大时优势更加明显。

**EAGLE-2：动态 Draft 树。** EAGLE-2 进一步引入了上下文感知的动态 Draft 树结构。与固定拓扑的候选树不同，EAGLE-2 根据 EAGLE 网络输出的置信度动态决定树的扩展方向：对于置信度高的分支进行更深的扩展，对于置信度低的分支进行更宽的扩展（保留更多候选），甚至剪掉置信度极低的分支。这种自适应策略在固定的验证预算（总候选 Token 数）下，能够最大化期望的被接受 Token 数量。

**训练过程。** EAGLE 的特征预测网络需要训练。训练数据通过对目标模型进行正常推理来收集——记录每个位置的隐藏状态和 Token 嵌入，然后训练 EAGLE 网络在给定当前隐藏状态和 Token 嵌入的条件下预测下一步的隐藏状态。训练的监督信号是目标模型本身产生的隐藏状态，这使得 EAGLE 网络能够学习到目标模型特征空间中的动态规律。由于 EAGLE 网络只有单层 Transformer，训练成本很低，通常在一两个 GPU 上训练数小时即可完成。

**EAGLE-3 与最新进展。** 在 EAGLE 和 EAGLE-2 之后，该系列的研究仍在继续发展。后续工作进一步优化了特征预测网络的架构，探索了更高效的树结构搜索算法，以及与量化等其他优化技术的协同集成方案。

**EAGLE-3：数据扩展的 Scaling Law。** EAGLE-3 由 Li 等人于 2025 年 3 月提出，是 EAGLE 系列的最新一代。EAGLE-3 做出了两个根本性的改变。

第一个改变是放弃特征预测约束，转向直接 Token 预测。EAGLE 要求 Draft 网络的输出在特征空间中逼近目标模型的顶层隐藏状态，这在训练中引入了一个额外的特征预测损失 $l_{\text{fea}}$。EAGLE-3 的作者发现，这一约束实际上限制了 Draft 网络的表达能力，导致增加训练数据时性能提升受限。EAGLE-3 移除了特征预测损失，只保留 Token 预测损失，让 Draft 网络完全自由地优化其内部表示。

第二个改变是多层特征融合（Multi-Layer Feature Fusion）。由于不再需要预测顶层特征，EAGLE-3 不再仅仅依赖目标模型的顶层隐藏状态。相反，它从目标模型的低层、中层和高层分别提取特征，将这三个特征向量拼接后通过一个全连接层降维，得到一个融合了不同层级语义信息的输入特征 $g$。这使得 Draft 网络能够利用更丰富的上下文信息，从单纯的"下一个 Token 的表示"拓展到"多层次的语义理解"。

为了解决去除特征预测后的自回归误差累积问题，EAGLE-3 引入了一种称为"Training-Time Test"的训练策略。在训练过程中，Draft 网络不仅在真实特征序列上进行预测，还会模拟推理时的多步生成过程——将自身的输出作为下一步的输入进行递归预测——从而让网络在训练阶段就适应推理时面对自身预测结果（而非目标模型真实特征）作为输入的场景。

这些设计改进使 EAGLE-3 首次发现了推理加速领域的 Scaling Law：随着训练数据量的增加，Draft 模型的接受率和加速比持续提升，这在此前的 EAGLE 和 Medusa 架构中从未观察到。实验表明，EAGLE-3 在 LLaMA-3.1 8B 上可达到最高 6.5 倍的加速比，较 EAGLE-2 提升约 1.4 倍。

---

## 7.3 投机解码在推理引擎中的实现

从算法原理到工程系统的落地，投机解码面临着大量实现层面的挑战：如何与连续批处理协同工作？如何管理 Draft 模型和目标模型各自的 KV Cache？如何高效地实现树结构的并行验证？本节深入分析 vLLM 和 SGLang 两大推理引擎中投机解码的具体实现方案。

### 7.3.1 vLLM 中的 Speculative Decoding 实现

vLLM 提供了一套全面的投机解码支持框架，涵盖了从独立 Draft 模型到 EAGLE/EAGLE-3、N-gram 匹配、MLP Speculator、多 Token 预测（MTP）、以及后缀解码等多种 Draft 策略。其投机解码模块的设计目标是在保证无损性的前提下，与 vLLM 的连续批处理和 PagedAttention 无缝集成。

**整体架构。** vLLM 的投机解码采用一种称为"Lookahead Scheduling"的调度策略。在每一轮调度中，调度器会为每个处于 Decode 阶段的请求分配额外的 KV Cache 空间，用于存放即将生成的 Draft Token。整个流程可以分为三步：

第一步，**Draft 阶段**。根据所选的 Draft 方法（如 EAGLE head、独立小模型、N-gram 等），为批次中的每个请求生成若干候选 Token。对于基于模型的 Draft 方法，这一步需要执行 Draft 模型的前向传播；对于 N-gram 方法，则直接从已生成的上下文中匹配重复的 N-gram 模式。

第二步，**Verify 阶段**。将所有请求的 Draft Token 拼接成一个扩展后的输入序列，送入目标模型执行一次前向传播。目标模型在这次前向传播中同时为所有位置计算概率分布。

第三步，**Accept/Reject 阶段**。对每个请求的候选 Token 执行接受-拒绝采样，确定本轮实际接受的 Token 数量，更新各请求的生成状态和 KV Cache。

**与连续批处理的兼容性。** 投机解码与连续批处理的结合带来了额外的复杂性。在标准连续批处理中，每个请求在每次迭代中只前进一个 Token。引入投机解码后，不同请求的接受 Token 数量不同：有的请求可能接受了全部 5 个候选，有的可能只接受了 1 个。vLLM 通过在每次迭代结束时统一更新各请求的状态来处理这种异质性——接受了多个 Token 的请求直接推进多步，然后所有请求再次进入下一轮 Draft-Verify 循环。

**KV Cache 管理。** 在使用独立 Draft 模型时，需要为 Draft 模型和目标模型分别维护 KV Cache。Draft 模型的 KV Cache 通常较小（因为 Draft 模型参数量小），但仍然占用额外的显存。vLLM 的 Block Manager 需要为两套 KV Cache 分别进行分页管理。当使用 EAGLE 等基于目标模型隐藏状态的 Draft 方法时，KV Cache 管理相对简单——EAGLE head 只有单层 Transformer，其 KV Cache 占用极小。

**配置与使用。** 在 vLLM 中启用投机解码通过 `speculative_config` 参数进行配置。以 EAGLE-3 为例，典型的配置如下：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
)
```

其中 `num_speculative_tokens` 控制每轮 Draft 的候选 Token 数量，`method` 指定 Draft 策略类型。vLLM 支持的 Draft 方法包括 `eagle`、`eagle3`、`draft_model`、`ngram`、`mlp`、`pard`（Parallel Draft Model）、`suffix` 等。

**无损性保证的工程实现。** vLLM 对投机解码的无损性保证进行了严格的工程验证。在算法层面，vLLM 实现了标准的接受-拒绝采样器（Rejection Sampler），并通过收敛性测试验证采样分布与目标分布的一致性。在端到端层面，vLLM 的测试套件中包含大量"贪心采样等价性"测试：验证在贪心解码模式下，启用投机解码的输出与不启用投机解码的输出完全一致。然而，由于浮点精度差异和批处理中的数值稳定性问题，在非贪心采样设置下，输出可能存在极微小的统计差异，这是硬件层面的固有限制。

**方法选择指南。** vLLM 文档提供了一个方法选择矩阵，总结如下：在低 QPS（延迟敏感）场景下，EAGLE、MTP 和独立 Draft 模型的效果最好，可提供显著的延迟降低；在高 QPS（吞吐敏感）场景下，EAGLE 和 MTP 仍然表现较好，而 N-gram 和后缀解码由于不需要额外的模型计算开销，在高负载下也能提供温和的收益而不会带来额外的 GPU 压力。MTP 方法在目标模型本身具有原生 MTP 支持（如 DeepSeek-V3 的多 Token 预测头）时效果最佳，因为此时无需额外训练 Draft 模型。

### 7.3.2 SGLang 中的 EAGLE 集成

SGLang 将 EAGLE 系列作为其主推的投机解码方案，并在引擎层面进行了深度集成和优化。SGLang 的投机解码实现目标是"最大化吞吐量"——在保持无损性的前提下，尽可能充分利用 GPU 的计算能力。

**SGLang 的 EAGLE 推理流程。** SGLang 同时支持 EAGLE-2 和 EAGLE-3。在推理时，用户通过命令行参数进行配置：

```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-algorithm eagle3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3-Instruct-8B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 10
```

这里的三个关键参数分别控制投机解码的行为。`speculative-num-steps` 指定 EAGLE 网络自回归执行的步数，即 Draft 树的最大深度。`speculative-eagle-topk` 指定每一步保留的 top-k 个候选，控制 Draft 树的宽度（分支因子）。`speculative-num-draft-tokens` 限制 Draft 树的总候选 Token 数，当树中的 Token 总数达到此上限时停止扩展。

这三个参数共同决定了 Draft 树的形状：更大的步数和 top-k 值会产生更大更深的树，提高期望的接受 Token 数量，但也增加了验证阶段的计算开销。SGLang 提供了一个专用的基准测试脚本 `bench_speculative.py`，帮助用户在特定硬件和模型上搜索最优的参数组合。

**SpecForge：SGLang 配套的 Draft 模型训练框架。** SGLang 团队开发了 SpecForge，一个专门用于训练 EAGLE-3 Draft 模型的框架。SpecForge 解决了此前开源 EAGLE 训练工具存在的维护不善、功能有限、与推理框架不兼容等问题，提供了从训练到部署的端到端闭环体验。

SpecForge 提供两种训练模式：在线模式（Online Mode）在训练过程中动态调用目标模型生成隐藏状态，适合快速迭代；离线模式（Offline Mode）预先收集目标模型的隐藏状态存储到磁盘上，训练时直接加载，适合需要复现性和数据复用的场景（但需要较大的磁盘空间，例如处理 UltraChat + ShareGPT 数据集需要约 12TB 存储）。SpecForge 支持 FSDP（Fully Sharded Data Parallel）和张量并行等分布式训练策略，能够为超大规模模型（如 LLaMA-4 Maverick、Scout 等 MoE 模型）训练 Draft 模型。

**性能表现。** 使用 SpecForge 训练的 LLaMA-4 Maverick Draft 模型在 MT-Bench 上实现了 2.18 倍的加速比，Scout 变体则达到 2.0 倍。这些数字验证了 EAGLE-3 在现代大模型上的实用价值。

### 7.3.3 Token Tree Verification 与批量验证

在 Medusa、EAGLE-2、EAGLE-3 等方法中，Draft 阶段生成的不再是一条线性的候选序列，而是一棵分支状的候选树。树结构的引入使得在固定的验证计算预算下能够覆盖更多的候选路径，但也带来了验证阶段的额外复杂性。

**树结构的候选空间。** 考虑一个两步、每步 top-2 的 Draft 树。第一步生成两个候选 Token ${a_1, a_2}$，第二步在每个第一步候选的基础上各生成两个候选，得到 ${(a_1, b_{11}), (a_1, b_{12}), (a_2, b_{21}), (a_2, b_{22})}$。这棵树包含 6 个候选 Token 节点（2 + 4），涵盖了 4 条完整路径。如果使用线性候选链（长度为 6），只能覆盖 1 条路径。树结构通过牺牲单条路径的深度来换取路径的多样性，在接受率较低的场景下特别有利。

**Tree Attention 机制。** 验证一棵候选树的关键在于设计合适的注意力掩码（Attention Mask），使得树中每个节点只能看到其祖先节点，而不能看到兄弟节点或其他分支上的节点。这被称为 Tree Attention。

具体而言，设候选树有 $N$ 个节点，目标模型需要对这 $N$ 个节点执行一次并行的前向传播。注意力掩码矩阵 $M \in {0, 1}^{N \times N}$ 的构造规则为：$M_{ij} = 1$ 当且仅当节点 $j$ 是节点 $i$ 的祖先（包括自身）。这保证了每个节点在计算 Attention 时只聚合其前缀路径上的信息，不会泄漏其他分支的内容。

在实际实现中，Tree Attention 需要处理与 PagedAttention（vLLM）或连续 KV Cache（SGLang）的兼容问题。由于树中不同分支共享前缀部分的 KV Cache，但在分支点之后各自拥有独立的 KV Cache 条目，KV Cache 的存储和索引变得更加复杂。一种常见的实现方式是将树展平为一个线性序列，通过自定义的注意力掩码来编码树结构的依赖关系。

**批量验证的流程。** 在 Tree Attention 前向传播完成后，需要从树中选出一条被接受的最长路径。验证过程从根节点开始，沿着树的层次逐层检查。在每一层，对该层的所有候选 Token 执行接受-拒绝判定。如果某个 Token 被接受，则沿该分支继续深入；如果所有分支在某层都被拒绝，则在该层从目标模型的修正分布中采样一个新 Token，验证终止。

对于 EAGLE-2 和 EAGLE-3 的动态 Draft 树，验证完成后还需要根据本轮的接受情况更新 Draft 树的构造策略。例如，如果某些类型的分支总是被拒绝，后续可以减少该类型分支的分配，将预算分配给更有希望的分支。

**SpecInfer 的多 Draft 模型树。** 值得一提的是，SpecInfer（Miao 等人，2024）将树结构验证推向了更极端的设计：它允许同时使用多个不同的 Draft 模型，每个模型生成自己的候选序列，所有候选合并成一棵大树，由目标模型一次性验证。这进一步提高了候选的多样性，但也使得树的管理和验证更加复杂。

---

## 7.4 投机解码的效率分析

投机解码并不是"免费的午餐"——它引入了 Draft 阶段的额外计算开销和验证阶段的 KV Cache 管理开销。理解投机解码的加速效果需要精确分析接受率、Draft 开销、Verify 开销之间的平衡关系。

### 7.4.1 接受率（Acceptance Rate）与加速比

**单步接受率。** 设 Draft 模型在单个位置的平均接受率为 $\alpha$（$0 \le \alpha \le 1$），即对于一个 Draft Token，它被目标模型接受的概率为 $\alpha$。这个值取决于 Draft 模型与目标模型的分布匹配程度，以及当前上下文的"可预测性"。

**期望接受长度。** 在投机长度为 $\gamma$ 的一轮投机解码中，期望被接受的 Token 数量（不计最后额外采样的一个 Token）为：

$$\mathbb{E}[\text{accepted tokens}] = \sum_{i=1}^{\gamma} \alpha^i = \frac{\alpha(1 - \alpha^\gamma)}{1 - \alpha}$$

这个公式基于如下推导：第 $i$ 个 Draft Token 被接受的前提是前 $i-1$ 个都被接受，因此第 $i$ 个 Token 被接受的概率为 $\alpha^i$（假设各位置的接受率独立且相同，这是一个近似假设）。总期望接受数就是各位置接受概率之和。

加上投机解码保证至少产生 1 个 Token（即使所有 Draft 都被拒绝，也会从目标模型的修正分布中采样一个），每轮投机解码期望产生的 Token 数为：

$$\mathbb{E}[\text{tokens per round}] = 1 + \frac{\alpha(1 - \alpha^\gamma)}{1 - \alpha} = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

当 $\gamma \to \infty$ 时，这个期望趋向于 $\frac{1}{1 - \alpha}$。这给出了投机解码理论加速比的上界。

**加速比分析。** 设目标模型单次前向传播（Decode 一个 Token）的耗时为 $T_p$，Draft 模型单次前向传播的耗时为 $T_q$。在一轮投机长度为 $\gamma$ 的投机解码中，总耗时为 Draft 阶段的 $\gamma \cdot T_q$ 加上 Verify 阶段的 $c \cdot T_p$（其中 $c$ 是验证 $\gamma$ 个 Token 相对于 Decode 一个 Token 的时间比率，通常 $c$ 略大于 1 但远小于 $\gamma$），期望产出 $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$ 个 Token。

因此，每个 Token 的平均生成时间为：

$$T_{\text{spec}} = \frac{\gamma \cdot T_q + c \cdot T_p}{\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}}$$

而标准自回归解码每个 Token 的生成时间为 $T_p$。加速比为：

$$\text{Speedup} = \frac{T_p}{T_{\text{spec}}} = \frac{T_p \cdot \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}}{\gamma \cdot T_q + c \cdot T_p}$$

从这个公式可以清晰看出影响加速比的三个核心因素：接受率 $\alpha$（越高越好）、Draft 速度比 $T_q / T_p$（越小越好，即 Draft 模型越快越好）、以及验证开销比 $c$（越接近 1 越好，即验证多个 Token 的额外开销越小越好）。

**数值示例。** 假设 $\alpha = 0.8$，$\gamma = 5$，$T_q / T_p = 0.05$（Draft 模型比目标模型快 20 倍），$c = 1.2$（验证 5 个 Token 只比验证 1 个 Token 慢 20%）。则期望每轮产出 Token 数为 $\frac{1 - 0.8^6}{1 - 0.2} \approx 3.9$，每轮总耗时为 $5 \times 0.05 T_p + 1.2 T_p = 1.45 T_p$，加速比为 $3.9 / 1.45 \approx 2.7$ 倍。

如果接受率降至 $\alpha = 0.5$，则期望产出降至约 $\frac{1 - 0.5^6}{0.5} \approx 1.97$ 个 Token，加速比降至 $1.97 / 1.45 \approx 1.36$ 倍。可见接受率是决定性因素——低接受率下投机解码的收益急剧下降。

**实际测量值。** 在实践中，EAGLE-3 在 LLaMA-3.1 8B 上的平均接受长度（average acceptance length, $\tau$）可达 4 到 6 个 Token（取决于任务类型），对应约 3 到 6.5 倍的加速比。代码生成和数学推理等结构化程度高的任务通常具有更高的接受率，而开放式对话的接受率相对较低。

### 7.4.2 Draft 开销与 Verify 开销的平衡

投机解码的效率优化本质上是一个平衡问题：Draft 生成的候选越多，潜在收益越大，但 Draft 开销和 Verify 开销也相应增加。

**Draft 开销的分解。** Draft 阶段的开销主要包括：Draft 模型的前向传播计算（对于基于模型的方法）、Draft Token 的采样过程、以及 Draft 模型 KV Cache 的更新。对于独立 Draft 模型方案，Draft 前向传播是主要开销，其大小与 Draft 模型参数量和投机长度成正比。对于 EAGLE 系列，由于 Draft 网络只有单层 Transformer，计算开销很小，但需要额外的 LM head 计算（与目标模型共享）。对于 Medusa，Draft 开销几乎为零（与正常前向传播合并）。对于 N-gram 匹配，Draft 开销纯粹是 CPU 上的字符串匹配操作，对 GPU 无影响。

**Verify 开销的分析。** 验证阶段的核心开销是目标模型对扩展输入的前向传播。验证 $\gamma$ 个线性候选 Token 等价于将一个长度为 $\gamma$ 的 Prefill 追加到已有的 KV Cache 之后。由于 Prefill 是计算密集型的，当 $\gamma$ 较小（如 3-7）时，额外的计算开销相对于加载模型权重的开销而言并不大，因此 $c$ 通常接近 1。但对于树结构的候选，验证的 Token 总数可能远大于 $\gamma$（例如一棵有 30 个节点的 Draft 树），此时验证开销会显著增加。

**与 Batch Size 的交互效应。** 投机解码的效益与 Batch Size 之间存在复杂的交互关系。在小 Batch Size（低 QPS）场景下，GPU 的计算资源大量闲置，验证阶段增加的计算几乎是"免费的"，投机解码的加速效果最为显著。然而，随着 Batch Size 增大，GPU 逐渐趋于饱和，验证 Draft Token 的额外计算开始与其他请求争夺计算资源。更重要的是，投机解码对 Batch 中不同请求的处理带来了异质性——不同请求的接受 Token 数不同——这使得 Batch 内的对齐（Alignment）成为问题，需要用 padding 或其他策略来处理长度不一致的情况，而对齐过程本身会引入额外的开销。

研究表明，在高 QPS 场景下，投机解码的吞吐量优势会减小，甚至可能因为 Draft 和对齐开销而导致总吞吐量下降。因此，在生产部署中，需要根据实际的 QPS 水平和 SLO 要求来决定是否以及如何启用投机解码。一种策略是动态地根据当前负载调整投机长度：低负载时使用较大的 $\gamma$ 以最大化延迟优势，高负载时减小 $\gamma$ 甚至关闭投机解码以避免吞吐量损失。

### 7.4.3 与其他优化技术的兼容性

投机解码作为一种解码层的优化技术，需要与推理系统中的其他优化技术协同工作。这种协同既带来了叠加收益的机会，也引入了兼容性挑战。

**与量化的兼容性。** 投机解码与模型量化在大多数情况下是正交且互补的。目标模型可以使用 FP8、INT8、INT4 等量化方案来降低显存占用和计算开销，同时启用投机解码来减少解码步数。Draft 模型同样可以被量化以进一步减少 Draft 开销。然而，量化可能改变模型的输出分布，当目标模型被量化后，为未量化目标模型训练的 Draft 模型的接受率可能下降。因此，理想情况下应该为量化后的目标模型专门训练或微调 Draft 模型。

**与 KV Cache 优化的兼容性。** 投机解码与 PagedAttention、RadixAttention 等 KV Cache 管理技术可以协同工作，但需要注意以下几点。在投机解码中，被拒绝的 Draft Token 对应的 KV Cache 条目需要被清理——这些"试探性"的 KV 向量不应该保留在缓存中。PagedAttention 的分页机制使得这种部分回滚成为可能，但增加了 Block Manager 的管理复杂度。RadixAttention 的前缀缓存也需要注意不要将未验证的 Draft Token 作为前缀进行缓存。KV Cache 压缩（量化、剪枝等）与投机解码的结合相对简单，只需确保验证阶段使用正确的 KV Cache 数据即可。

**与 Chunked Prefill 的兼容性。** 投机解码的验证阶段本质上是一次小规模的 Prefill（处理多个 Token），因此与 Chunked Prefill 的交互需要仔细设计。在 vLLM V1 中，Chunked Prefill 默认启用，投机解码的验证步骤会被当作一个 Prefill 块来处理，与其他正在进行 Prefill 的请求一起调度。

**与张量并行的兼容性。** 投机解码与张量并行是兼容的。目标模型可以跨多个 GPU 进行张量并行推理，Draft 模型可以使用独立的张量并行配置（通常使用更少的 GPU）。在 vLLM 中，可以通过 `draft_tensor_parallel_size` 参数独立配置 Draft 模型的并行度。

**与流水线并行的兼容性。** 截至目前，投机解码与流水线并行的结合尚未在主流推理引擎中得到很好的支持。这是因为流水线并行的层间通信模式与投机解码的 Draft-Verify 交替模式存在调度上的冲突，实现起来较为复杂。

**与 PD 分离的兼容性。** 投机解码与 Prefill-Decode 分离（PD 分离）架构存在有趣的交互。在 PD 分离架构中，Decode 节点专门负责逐 Token 生成，而投机解码可以显著减少 Decode 节点上的实际解码步数，从而降低 Decode 节点的负载。然而，验证阶段的多 Token 并行处理在计算特性上更接近 Prefill，这模糊了 Prefill 节点和 Decode 节点的职责边界。如何在 PD 分离架构中优化投机解码的部署是一个活跃的研究方向。

---

## 本章小结

投机解码是大模型推理优化领域中一项极具开创性的技术。它直面自回归解码的根本瓶颈——每次只生成一个 Token 导致的 GPU 利用率低下——通过"先猜后验"的范式将串行的逐 Token 生成转化为并行的批量验证，在不牺牲任何输出精度的前提下实现了显著的延迟降低。

本章系统讲述了投机解码从理论到实践的完整知识体系。7.1 节阐述了 Draft-Then-Verify 的核心范式及其无损性的数学证明，建立了理论基础。7.2 节全面梳理了各类 Draft 策略——从最朴素的独立小模型方案，到自投机的层跳跃方法，再到 Medusa 的并行多头预测，最终到 EAGLE 系列的特征层自回归方案——展现了从简单到精巧的技术演进路径。特别是 EAGLE-3 通过放弃特征预测约束、引入多层特征融合和 Training-Time Test 训练策略，首次在推理加速领域发现了 Scaling Law 现象，为 Draft 模型的持续改进开辟了数据驱动的路径。7.3 节深入分析了 vLLM 和 SGLang 两大引擎的工程实现，包括与连续批处理的协同、KV Cache 管理、Tree Attention 验证机制等实际挑战及其解决方案。7.4 节提供了严谨的效率分析框架，帮助读者理解接受率、Draft 开销、Verify 开销三者之间的平衡关系，以及投机解码在不同负载条件下的适用性。

展望未来，投机解码的发展方向包括：更强大的 Draft 模型架构（如利用 Scaling Law 持续提升精度）、更智能的动态投机策略（根据上下文难度和系统负载自适应调整）、与其他优化技术（量化、PD 分离、多模态处理等）的深度协同、以及在推理模型（Reasoning Model）的长思维链生成中的特殊优化（推理模型往往具有更强的 Token 可预测性，因此可能获得更高的接受率和加速比）。随着投机解码在 vLLM、SGLang 等主流引擎中的日益成熟，它正在从研究技术演进为生产级推理系统的标准配置之一。

---

_附：本章推荐论文与扩展阅读_

（1）Leviathan, Y., Kalman, M., & Matias, Y. (2023). “Fast Inference from Transformers via Speculative Decoding.” _ICML 2023._ —— 投机解码的原始论文之一，提出了完整的 Draft-Then-Verify 框架和无损性证明。

（2）Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). “Accelerating Large Language Model Decoding with Speculative Sampling.” —— 与前者同期独立提出的另一篇奠基论文。

（3）Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). “Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.” —— 多头并行草稿的开创性工作。

（4）Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). “EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.” _ICML 2024._ —— 特征层自回归 Draft 的突破性工作。

（5）Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). “EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees.” —— 动态 Draft 树的引入。

（6）Li, Y., Wei, F., Zhang, C., & Zhang, H. (2025). “EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.” —— 推理加速 Scaling Law 的发现。

（7）Miao, X., Oliaro, G., Zhang, Z., Cheng, X., Wang, Z., Wong, R. Y. Y., Chen, Z., Arfeen, D., Zhu, R., & Jia, Z. (2024). “SpecInfer: Accelerating Large Language Model Serving with Tree-Based Speculative Inference and Verification.” _ASPLOS 2024._ —— 多 Draft 模型的树结构推理系统。
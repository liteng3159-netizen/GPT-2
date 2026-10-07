# GPT-2 模型复现与权重验证

基于 PyTorch 从零实现 GPT-2 Decoder-only Transformer 架构，并加载 OpenAI GPT-2 官方预训练权重，将官方模型参数映射到自定义实现中，通过前向计算结果对比验证模型结构与参数加载的正确性。

## 项目简介

本项目主要用于深入理解 GPT-2 的模型架构、Transformer 核心组件以及预训练模型权重的加载与映射过程。

项目没有直接调用现成的 GPT-2 模型进行推理，而是基于 PyTorch 自己实现 GPT-2 的核心网络结构，并将 OpenAI GPT-2 官方预训练权重转换后加载到自定义模型中，最终通过相同输入下的模型输出进行数值验证。

整体流程如下：

```text
文本输入
   ↓
GPT-2 Tokenizer
   ↓
Token IDs
   ↓
Token Embedding + Position Embedding
   ↓
GPT-2 Transformer Blocks
   ├── Layer Normalization
   ├── Multi-Head Self-Attention
   ├── Causal Mask
   ├── Feed Forward
   └── Residual Connection
   ↓
Language Model Head
   ↓
Logits
   ↓
Next Token Prediction
   ↓
自回归文本生成
```

## 核心内容

### 1. GPT-2 模型结构实现

基于 PyTorch 实现 GPT-2 Decoder-only Transformer 的核心组件，包括：

- Token Embedding
- Position Embedding
- Multi-Head Self-Attention
- Causal Mask
- Layer Normalization
- Feed Forward Network
- Residual Connection
- Transformer Block
- Language Model Head

通过模块化实现完整的 GPT-2 前向计算流程。

### 2. GPT-2 官方权重加载

使用 OpenAI GPT-2 官方预训练模型权重，并分析官方模型参数结构，将其映射到自定义 PyTorch 模型中的对应参数。

主要包括：

- Embedding 权重映射
- Transformer Block 参数映射
- Attention 参数映射
- Feed Forward 参数映射
- Layer Normalization 参数映射
- Language Model Head 权重加载

该过程用于验证自定义 GPT-2 模型结构与官方模型参数之间的对应关系。

### 3. 模型输出验证

为了验证模型实现的正确性，使用相同的输入分别运行：

1. 官方 GPT-2 模型
2. 自定义 PyTorch GPT-2 模型

对比模型的中间层输出及最终 Logits，在允许的浮点数误差范围内验证两种实现的一致性。

验证流程：

```text
相同输入
   ↓
官方 GPT-2 ──────────→ 官方 Logits
                         │
                         │ 数值对比
                         ↓
自定义 GPT-2 ─────────→ 自定义 Logits
```

通过输出一致性验证模型结构、参数映射以及前向计算过程的正确性。

### 4. 自回归文本生成

基于复现后的 GPT-2 模型实现自回归生成：

```text
输入 Prompt
    ↓
GPT-2 Forward
    ↓
获取最后一个位置的 Logits
    ↓
选择 / 采样 Next Token
    ↓
将 Token 加入输入序列
    ↓
重复上述过程
    ↓
生成完整文本
```

通过逐 Token 预测理解 GPT-2 的自回归语言建模过程。

## 技术栈

- Python
- PyTorch
- Transformer
- GPT-2
- OpenAI GPT-2 Pretrained Weights
- Hugging Face Transformers

## 项目结构

```text
GPT-2/
├── modules/
│   ├── attention.py
│   └── gpt2_layer.py
├── models/
│   └── gpt2.py
├── classifier.py
├── optimizer.py
├── sanity_check.py
├── ...
├── README.md
└── LICENSE
```

## 项目收获

通过从零实现 GPT-2 并进行官方权重加载与输出验证，深入理解了 Decoder-only Transformer 的内部结构以及 GPT-2 的前向计算过程。

同时，通过分析和转换预训练模型参数，进一步理解了模型参数组织方式、权重映射以及模型复现中的数值一致性验证方法。

## References

- OpenAI GPT-2
- Stanford CS224N
- Hugging Face Transformers
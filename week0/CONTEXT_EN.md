# Week 0 — Context

## What is vLLM?

[vLLM](https://github.com/vllm-project/vllm) is an open-source library for **fast and memory-efficient inference and serving of Large Language Models (LLMs)**. Developed by the research group at UC Berkeley, vLLM maximizes GPU utilization through advanced techniques such as PagedAttention and continuous batching.

### Key Features

- **PagedAttention**: Manages the KV cache using fixed-size blocks (pages) that are dynamically allocated on demand. Rather than pre-allocating a large contiguous memory region for each sequence, PagedAttention divides the KV cache into non-contiguous blocks, significantly reducing internal fragmentation. When GPU memory is exhausted, the system can swap blocks to CPU or recompute to free resources.

- **Continuous Batching** (Dynamic Batching): Instead of waiting for a full static batch before processing, vLLM inserts new requests into the running batch between decode steps and immediately removes completed sequences. This keeps the GPU active nearly continuously, avoiding idle resource waste.

- **Multi-GPU support** via tensor parallelism, enabling large models to be distributed across multiple GPUs.

- **Quantization**: Supports quantization methods to reduce memory consumption.

- **OpenAI-compatible API**: Provides serving endpoints compatible with the OpenAI API, allowing straightforward integration into existing systems.

### Performance

- Up to **24x** higher throughput compared to vanilla Hugging Face Transformers, thanks to fused kernels and intelligent batching.
- **50–75%** reduction in GPU memory usage thanks to PagedAttention and quantization.
- Well-suited for high-concurrency applications such as chatbots, customer support systems, and enterprise agents.

### Comparison with Other Engines

| Engine | Strength | Limitation vs vLLM |
|---|---|---|
| **llama.cpp** | Low latency for single-user scenarios | Does not scale well under concurrent multi-user load |
| **TensorRT-LLM** | Optimized for specific NVIDIA hardware | Narrower model support |
| **SGLang / LMDeploy** | Highest batch throughput (C++) | Less flexible for diverse workloads |
| **TGI / Ollama** | Easy setup | Inferior scalability for interactive applications |

### When to Use vLLM?

- Serving LLMs for many concurrent users (production deployment).
- Requiring high throughput with stable latency.
- Deploying on GPU clusters (A100/H100/H200).
- Rapid prototyping with multi-GPU configurations.

> References: [vLLM Documentation](https://docs.vllm.ai/), [Red Hat — Why vLLM](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today), [Inference.net — vLLM Overview](https://inference.net/content/vllm)

---

## Finetuning Overview — The Concept Map

Based on [The Finetuning Landscape](https://theneuralmaze.substack.com/p/the-finetuning-landscape-a-map-of), the following sections cover the foundational concepts one should understand before proceeding to finetuning.

### 1. Attention Mechanism

Attention is the mechanism that allows a model to **look back at different parts of the input** rather than compressing the entire sequence into a single fixed vector. The core idea:

- Each token produces a **Query (Q)**, **Key (K)**, and **Value (V)**.
- The Query is compared against all Keys to compute attention weights.
- The output is a weighted sum of the Values.

Attention was originally introduced as an improvement to encoder-decoder RNN models in machine translation, predating the Transformer itself.

### 2. Self-Attention

Self-attention extends the attention idea: **each token in the same sequence attends to all other tokens** (including itself). For example, in the sentence "The cat sat on the mat", the token "sat" may attend strongly to "cat" (who sat) and "mat" (where they sat).

### 3. Scaled Dot-Product Attention & Multi-Head Attention

- **Scaled Dot-Product Attention**: The specific method by which the Transformer computes attention — dot product between Q and K, divided by √d_k, passed through softmax, then multiplied by V.
- **Multi-Head Attention**: Runs multiple attention heads in parallel, each of which can specialize in different patterns (short-range vs long-range, syntax vs semantics). The results are concatenated and projected through an output matrix.

### 4. Positional Encoding

Attention is inherently agnostic to token order. Positional encoding is added to token representations so the model can distinguish positions. The original Transformer uses sin/cos functions at varying frequencies — low frequencies encode coarse positional information, while high frequencies capture fine-grained differences.

### 5. Transformer Architecture — 3 Variants

| Architecture | Characteristics | Example |
|---|---|---|
| **Encoder-only** | Bidirectional self-attention. Well-suited for understanding, classification, and retrieval. | BERT |
| **Encoder-Decoder** | Encoder reads input, decoder generates output token-by-token with cross-attention. Well-suited for seq2seq tasks (translation, summarization). | T5, BART |
| **Decoder-only** | Uses only causal self-attention. Each token attends solely to preceding tokens. Predicts the next token. | GPT, LLaMA, Qwen |

> Decoder-only is the architecture behind most modern LLMs (ChatGPT, Claude, Qwen...) and is the central focus of this finetuning course.

### 6. Scaling Laws

Transformer performance improves following a power-law when increasing:
- **Model size** (number of parameters)
- **Training data** (number of tokens)
- **Compute** (FLOPs)

Decoder-only architectures are particularly well-suited for scaling thanks to their simple objective (next-token prediction), access to massive unlabeled data, and efficient parallel training.

### 7. LLM Training Pipeline

The standard pipeline was formalized by InstructGPT (OpenAI, 2022):

1. **Pretraining**: Train on massive raw text data with the causal language modeling objective (predict next token). No instructions, no human feedback. This is the phase where the model "learns about the world" — grammar, syntax, factual knowledge, code patterns. The result is a **base model** (foundation model).

2. **Supervised Fine-Tuning (SFT)**: Continue training on high-quality, structured instruction-response data. Teach the model how to answer questions, follow instructions, and format output appropriately.

3. **Alignment (RLHF/GRPO)**: Align the model with human preferences through Reinforcement Learning from Human Feedback or other alignment methods such as GRPO (Group Relative Policy Optimization).

> A two-phase perspective: **Pretraining** (building general language capability) → **Post-training** (fine-tuning, adjusting, aligning).

### 8. Pretraining & Base Model

- Uses **self-supervised learning** — the label is simply the next token in the sequence.
- Data sources: web pages, books, articles, code repositories.
- The base model excels at **continuing text** but does not yet know how to follow instructions or reject inappropriate requests.
- **Continued pretraining** is useful when additional domain knowledge is needed (medical, legal), when supporting low-resource languages, or when the data distribution differs significantly from the original training data.

### 9. Finetuning Techniques We Will Cover

- **SFT (Supervised Fine-Tuning)**: Full finetuning on instruction-response data.
- **LoRA (Low-Rank Adaptation)**: Only trains small low-rank matrices instead of updating all weights — significantly reducing memory and compute requirements.
- **QLoRA**: Combines quantization (4-bit) with LoRA to enable finetuning on consumer-grade GPUs.
- **RLHF / GRPO**: Alignment techniques to steer the model toward human preferences.

### References

- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [OpenAI (2022) — InstructGPT: Aligning Language Models to Follow Instructions](https://openai.com/index/instruction-following/)
- [Kaplan et al. (2020) — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Chip Huyen (2023) — RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
- [Colah (2015) — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Dive into Deep Learning — Zhang et al.](https://d2l.ai/)
- [NVIDIA — AI Scaling Laws](https://blogs.nvidia.com/blog/ai-scaling-laws/)

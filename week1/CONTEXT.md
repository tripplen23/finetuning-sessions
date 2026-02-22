# Week 1: Finetuning Landscape & Continued Pretraining

## Tổng quan

Week 1 giới thiệu nền tảng về kiến trúc Transformer và quy trình training của Large Language Models (LLMs), đặc biệt tập trung vào **Continued Pretraining (CPT)** - giai đoạn đầu tiên trong pipeline finetuning.

## 1. Kiến trúc Transformer

### Pipeline Overview

```mermaid
graph LR
    A[Raw Text] --> B[Pretraining]
    B --> C[Base Model]
    C --> D[Continued Pretraining]
    D --> E[Domain-Adapted Model]
    E --> F[Supervised Fine-Tuning]
    F --> G[Instruction Model]
    G --> H[RLHF/Alignment]
    H --> I[Production Model]
    
    style B fill:#e1f5ff
    style D fill:#e1f5ff
    style F fill:#fff4e1
    style H fill:#ffe1e1
```

### Lịch sử Attention Mechanism

- **Attention không được phát minh trong paper Transformer!**
- Ban đầu được thiết kế để cải thiện encoder-decoder RNN models cho machine translation
- Giải quyết vấn đề: thay vì nén toàn bộ input thành 1 vector cố định, cho phép model "nhìn lại" các phần khác nhau của input

### Attention Pooling

Cơ chế cơ bản:
- **Keys** (k1, k2, ..., kn): Tập các khóa
- **Values** (v1, v2, ..., vn): Tập các giá trị tương ứng
- **Query**: Thể hiện thông tin đang tìm kiếm

Attention so sánh query với tất cả keys, gán weight cho mỗi key, và tạo weighted sum của values.

### Self-Attention

```mermaid
graph TD
    subgraph "Input Sequence"
        T1[The]
        T2[cat]
        T3[sat]
        T4[on]
        T5[the]
        T6[mat]
    end
    
    subgraph "Self-Attention Process"
        T3 --> Q[Query: sat]
        T1 --> K1[Key: The]
        T2 --> K2[Key: cat]
        T3 --> K3[Key: sat]
        T4 --> K4[Key: on]
        T5 --> K5[Key: the]
        T6 --> K6[Key: mat]
        
        Q -.strong.-> K2
        Q -.medium.-> K6
        Q -.weak.-> K1
        Q -.weak.-> K5
    end
    
    style T3 fill:#ffeb3b
    style Q fill:#ffeb3b
    style K2 fill:#4caf50
    style K6 fill:#8bc34a
```

Mỗi token trong sequence attend to tất cả tokens khác trong cùng sequence:

```
Input: "The cat sat on the mat"

- Mỗi token → Query (Q), Key (K), Value (V)
- Query của mỗi token so sánh với Keys của tất cả tokens
- Tạo weighted sum của Values
```

Ví dụ: token "sat" có thể:
- Attend mạnh đến "cat" (chủ ngữ)
- Attend đến "mat" (địa điểm)
- Ignore "the" (function words)

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

1. Queries so sánh với Keys bằng dot product
2. Normalize bằng softmax
3. Weighted sum của Values

### Multi-Head Attention

Thay vì 1 attention head, sử dụng nhiều heads song song:
- Mỗi head học các patterns khác nhau
- Short-range vs long-range dependencies
- Syntactic vs semantic relationships

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
```

### Positional Encoding

Attention không có khái niệm về thứ tự → cần inject positional information:
- Sử dụng sine/cosine functions ở các tần số khác nhau
- Low-frequency: coarse position
- High-frequency: fine-grained differences

## 2. Ba kiến trúc Transformer

```mermaid
graph TB
    subgraph "Encoder-Only (BERT)"
        E1[Token 1] <--> E2[Token 2]
        E2 <--> E3[Token 3]
        E3 <--> E1
        E1 --> O1[Classification]
        E2 --> O1
        E3 --> O1
    end
    
    subgraph "Encoder-Decoder (T5)"
        ED1[Input Tokens] --> ENC[Encoder]
        ENC --> DEC[Decoder]
        DEC --> ED2[Output Tokens]
        DEC -.cross-attention.-> ENC
    end
    
    subgraph "Decoder-Only (GPT)"
        D1[Token 1] --> D2[Token 2]
        D2 --> D3[Token 3]
        D3 --> D4[Token 4]
        D4 -.predict.-> D5[Next Token]
    end
    
    style E1 fill:#e3f2fd
    style E2 fill:#e3f2fd
    style E3 fill:#e3f2fd
    style D1 fill:#fff3e0
    style D2 fill:#fff3e0
    style D3 fill:#fff3e0
    style D4 fill:#fff3e0
```

### Encoder-Only (BERT)

- Bidirectional self-attention
- Tốt cho: classification, retrieval, semantic similarity
- Không generate text, chỉ encoding
- Pretraining: Masked Language Modeling

### Encoder-Decoder (T5, BART)

- Encoder: đọc và represent input
- Decoder: generate output từng token
- Cross-attention: decoder nhìn vào encoder outputs
- Causal self-attention: không peek future tokens
- Tốt cho: translation, summarization

### Decoder-Only (GPT, LLaMA, Qwen)

- **Kiến trúc của modern LLMs**
- Chỉ có causal self-attention
- Mỗi token chỉ attend to tokens trước nó
- Training objective: predict next token
- Scaling remarkably well

## 3. LLM Training Pipeline

### InstructGPT Framework (2022)

Pipeline 3 giai đoạn chuẩn:

1. **Pretraining**: Large-scale raw text
2. **Supervised Fine-Tuning (SFT)**: High-quality task examples
3. **Alignment (RLHF)**: Reinforcement Learning from Human Feedback

Hoặc view 2 phases:
- **Pretraining**: Build general capabilities
- **Post-training**: Refine, adapt, align

## 4. Pretraining - "Learning the World"

### Causal Language Modeling (CLM)

- Predict next token given previous tokens
- Self-supervised learning
- No labels, no instructions
- Dataset: web pages, books, articles, code

### Base Models

Kết quả của pretraining:
- Excellent at text completion
- Broad language patterns, facts, world knowledge
- **Chưa thể**: follow instructions, refuse requests, optimize for safety
- Powerful but unpredictable (Shoggoth analogy)

### Scaling Laws

Performance cải thiện theo power-law khi tăng:
- Model size (parameters)
- Training data (tokens)
- Training compute

## 5. Continued Pretraining (CPT)

### CPT vs SFT Comparison

```mermaid
graph LR
    subgraph "CPT - Buffet Approach"
        CPT1[Raw Text] --> CPT2[Loss on ALL tokens]
        CPT2 --> CPT3[Packing: Fill context]
        CPT3 --> CPT4[Domain Knowledge]
    end
    
    subgraph "SFT - Multi-course Meal"
        SFT1[Q&A Pairs] --> SFT2[Loss on Assistant only]
        SFT2 --> SFT3[Padding-free batching]
        SFT3 --> SFT4[Behavior & Structure]
    end
    
    style CPT1 fill:#e1f5ff
    style CPT4 fill:#e1f5ff
    style SFT1 fill:#fff4e1
    style SFT4 fill:#fff4e1
```

### Khi nào dùng CPT?

- Add new domain knowledge (legal, medical, finance)
- Support underrepresented languages
- Data distribution khác với original model

### CPT vs SFT

**CPT (Buffet approach):**
- Raw text, maximize knowledge per FLOP
- Loss calculated on EVERY token
- Packing: concatenate documents into full context (8k-128k tokens)
- Goal: absorb domain knowledge

**SFT (Multi-course meal):**
- Structured Q&A pairs
- Loss ONLY on Assistant tokens (User tokens = -100)
- Padding-free with Flash Attention 2
- Goal: teach behavior and structure

### Curriculum Learning

```mermaid
graph TD
    A[Start Training] --> B[Phase 1: Gold Standard Data]
    B --> C[Wikipedia, Textbooks]
    C --> D[Phase 2: Medium Quality]
    D --> E[News, Articles]
    E --> F[Phase 3: Long Tail]
    F --> G[Reddit, Web Crawls]
    
    A2[Context: 512 tokens] --> B2[Context: 2048 tokens]
    B2 --> C2[Context: 8192 tokens]
    C2 --> D2[Context: 32k-128k tokens]
    
    style B fill:#4caf50
    style C fill:#4caf50
    style D fill:#8bc34a
    style E fill:#8bc34a
    style F fill:#ffc107
    style G fill:#ffc107
```

Organize data by quality/complexity:

1. **Data Quality Sorting**: 
   - Start: Gold Standard (Wikipedia, textbooks)
   - Gradually: Long Tail (Reddit, web crawls)

2. **Sequence Length Scaling**:
   - Start: 512 tokens (local syntax)
   - Gradually: 4k → 8k → 128k (long-range dependencies)

**Benefits:**
- Faster convergence
- Lower final loss
- More robust to noise

### Knowledge Distillation

Transfer intelligence từ large "teacher" model sang small "student":

- Teacher provides soft targets (probability distribution)
- Student learns "Dark Knowledge" (relationships between words)
- Result: 10x smaller, 10x faster, 90% capability
- Chain-of-Thought Distillation: transfer reasoning process

## 6. Hardware & Memory Optimization

### Memory Breakdown Visualization

```mermaid
pie title "GPU Memory Usage (Full Precision)"
    "Model Weights" : 1.2
    "Gradients" : 1.2
    "Optimizer States" : 2.4
    "Activations" : 8.0
```

```mermaid
pie title "GPU Memory Usage (8-bit Optimizer)"
    "Model Weights" : 1.2
    "Gradients" : 1.2
    "Optimizer States" : 0.6
    "Activations" : 8.0
```

### Memory Wall

Training memory ≠ Model size:
- Weights: 1.2GB (0.6B params in FP16)
- Gradients: 1.2GB
- Optimizer states: 2.4GB (3-4x weights!)
- Activations: Variable (batch_size × context_length)

### Optimization Techniques

**Mixed Precision (BF16):**
- Range of FP32, memory of FP16
- Faster without FP16 instability

**Gradient Accumulation:**
- Calculate gradients over micro-batches
- Update weights after N steps

**Activation Checkpointing:**
- Discard activations during forward pass
- Re-calculate during backward pass
- Trade: 25% more compute for massive VRAM savings

**Flash Attention 2:**
- Old: Activations grow O(N²) with context length
- New: Linear O(N) scaling
- Requires GPU Compute Capability ≥ 8.0

## 7. Lab 1 Implementation

### Tech Stack

- **Model**: Qwen3-0.6B-Base (small, fast, capable)
- **Dataset**: pritamdeb68/Math-Pretraining-Data
- **Framework**: Unsloth (speed optimization)
- **Trainer**: TRL SFTTrainer
- **Tracking**: Comet.ml

### Key Parameters

```python
model_name = "Qwen/Qwen3-0.6B-Base"
max_seq_length = 1024
batch_size = 4
gradient_accumulation_steps = 4  # Effective batch = 16
learning_rate = 2e-5  # Low LR for full finetuning
optim = "adamw_8bit"  # Save optimizer memory
packing = True  # Critical for CPT
```

### Memory Breakdown (A10G 24GB)

```
Full precision optimizer:
- Weights: 1.2GB
- Gradients: 1.2GB
- Optimizer: 2.4GB
- Activations: ~8GB
Total: ~13GB ✅

8-bit optimizer:
- Weights: 1.2GB
- Gradients: 1.2GB
- Optimizer: 0.6GB (4x smaller!)
- Activations: ~8GB
Total: ~11GB ✅
```

### Running on HF Jobs

```bash
hf jobs uv run \
  --flavor a10g-small \
  --timeout 3h \
  -s COMET_API_KEY="..." \
  -s HF_TOKEN="..." \
  -e COMET_PROJECT_NAME="finetuning-sessions-week1" \
  main.py
```

## Key Takeaways

1. **Attention is all you need** - nhưng cần hiểu cách nó hoạt động
2. **Decoder-only** là kiến trúc của modern LLMs
3. **Pretraining** builds raw intelligence, **finetuning** shapes behavior
4. **CPT** teaches domain knowledge, **SFT** teaches structure
5. **Memory optimization** là critical cho training efficiency
6. **Curriculum Learning** và **Knowledge Distillation** là advanced techniques
7. **Scaling laws** explain why bigger models work better

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/pdf/1706.03762)
- [Understanding LSTM Networks (Olah, 2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Dive into Deep Learning (Zhang et al., 2023)](https://d2l.ai/)
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [InstructGPT (OpenAI, 2022)](https://openai.com/index/instruction-following/)

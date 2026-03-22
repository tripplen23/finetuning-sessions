# Week 0 — Context

## vLLM là gì?

[vLLM](https://github.com/vllm-project/vllm) là một thư viện mã nguồn mở dùng để **inference và serving các mô hình ngôn ngữ lớn (LLM)** một cách nhanh chóng và tiết kiệm bộ nhớ GPU. Được phát triển bởi nhóm nghiên cứu tại UC Berkeley, vLLM tối ưu hóa việc sử dụng GPU thông qua các kỹ thuật tiên tiến như PagedAttention và continuous batching.

### Các tính năng chính

- **PagedAttention**: Quản lý KV cache theo các block (trang) có kích thước cố định, được cấp phát động khi cần. Thay vì pre-allocate một vùng nhớ lớn liên tục cho mỗi sequence, PagedAttention chia KV cache thành các block không liên tục, giảm đáng kể lãng phí bộ nhớ nội bộ (internal fragmentation). Khi bộ nhớ GPU đầy, hệ thống có thể swap block ra CPU hoặc recompute để giải phóng tài nguyên.

- **Continuous Batching** (Dynamic Batching): Thay vì chờ đủ một batch tĩnh rồi mới xử lý, vLLM thêm request mới vào batch đang chạy giữa các bước decode, đồng thời loại bỏ ngay các sequence đã hoàn thành. Điều này giữ GPU hoạt động gần như liên tục, tránh lãng phí tài nguyên khi chờ đợi.

- **Hỗ trợ multi-GPU** thông qua tensor parallelism, cho phép phân tán mô hình lớn trên nhiều GPU.

- **Quantization**: Hỗ trợ các phương pháp lượng tử hóa để giảm bộ nhớ sử dụng.

- **API tương thích OpenAI**: Cung cấp endpoint serving tương thích với OpenAI API, dễ dàng tích hợp vào hệ thống hiện có.

### Hiệu năng

- Throughput cao hơn đến **24x** so với Hugging Face Transformers thông thường nhờ fused kernel và batching thông minh.
- Giảm **50-75%** bộ nhớ GPU nhờ PagedAttention và quantization.
- Phù hợp cho các ứng dụng high-concurrency như chatbot, customer support, và enterprise agent.

### So sánh với các engine khác

| Engine | Điểm mạnh | Hạn chế so với vLLM |
|---|---|---|
| **llama.cpp** | Latency thấp cho single-user | Không scale tốt khi nhiều user đồng thời |
| **TensorRT-LLM** | Tối ưu phần cứng NVIDIA cụ thể | Hỗ trợ model hẹp hơn |
| **SGLang / LMDeploy** | Throughput batch cao nhất (C++) | Ít linh hoạt cho workload đa dạng |
| **TGI / Ollama** | Dễ setup | Scalability kém hơn cho interactive apps |

### Khi nào dùng vLLM?

- Serving LLM cho nhiều user đồng thời (production deployment).
- Cần throughput cao với latency ổn định.
- Deploy trên GPU cluster (A100/H100/H200).
- Prototype nhanh với multi-GPU.

> Nguồn tham khảo: [vLLM Documentation](https://docs.vllm.ai/), [Red Hat — Why vLLM](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today), [Inference.net — vLLM Overview](https://inference.net/content/vllm)

---

## Tổng quan về Finetuning — Bản đồ các khái niệm

Dựa trên bài viết [The Finetuning Landscape](https://theneuralmaze.substack.com/p/the-finetuning-landscape-a-map-of), dưới đây là các chủ đề nền tảng cần nắm trước khi đi vào finetuning.

### 1. Attention Mechanism

Attention là cơ chế cho phép mô hình **nhìn lại các phần khác nhau của input** thay vì nén toàn bộ sequence vào một vector cố định. Ý tưởng cốt lõi:

- Mỗi token tạo ra **Query (Q)**, **Key (K)**, và **Value (V)**.
- Query được so sánh với tất cả Key để tính trọng số attention.
- Output là tổng có trọng số (weighted sum) của các Value.

Attention ban đầu được giới thiệu như một cải tiến cho mô hình encoder-decoder RNN trong machine translation, trước khi Transformer ra đời.

### 2. Self-Attention

Self-attention mở rộng ý tưởng attention: **mỗi token trong cùng một sequence attend đến tất cả các token khác** (bao gồm cả chính nó). Ví dụ với câu "The cat sat on the mat", token "sat" có thể attend mạnh đến "cat" (ai ngồi) và "mat" (ngồi ở đâu).

### 3. Scaled Dot-Product Attention & Multi-Head Attention

- **Scaled Dot-Product Attention**: Cách cụ thể Transformer tính attention — dot product giữa Q và K, chia cho √d_k, qua softmax, rồi nhân với V.
- **Multi-Head Attention**: Chạy nhiều head attention song song, mỗi head có thể chuyên biệt hóa cho các pattern khác nhau (short-range vs long-range, cú pháp vs ngữ nghĩa). Kết quả được concat lại và chiếu qua một ma trận output.

### 4. Positional Encoding

Attention không có khái niệm về thứ tự. Positional encoding được thêm vào token representation để mô hình phân biệt được vị trí. Transformer gốc dùng hàm sin/cos ở các tần số khác nhau — tần số thấp cho thông tin vị trí thô, tần số cao cho sự khác biệt chi tiết.

### 5. Kiến trúc Transformer — 3 biến thể

| Kiến trúc | Đặc điểm | Ví dụ |
|---|---|---|
| **Encoder-only** | Bidirectional self-attention. Tốt cho understanding, classification, retrieval. | BERT |
| **Encoder-Decoder** | Encoder đọc input, decoder sinh output token-by-token với cross-attention. Tốt cho seq2seq (dịch, tóm tắt). | T5, BART |
| **Decoder-only** | Chỉ dùng causal self-attention. Mỗi token chỉ attend đến các token trước nó. Predict next token. | GPT, LLaMA, Qwen |

> Decoder-only là kiến trúc đằng sau hầu hết các LLM hiện đại (ChatGPT, Claude, Qwen...) và là trọng tâm của khóa học finetuning.

### 6. Scaling Laws

Hiệu năng của Transformer cải thiện theo power-law khi tăng:
- **Model size** (số parameters)
- **Training data** (số tokens)
- **Compute** (FLOPs)

Decoder-only đặc biệt phù hợp cho scaling nhờ objective đơn giản (next-token prediction), dữ liệu unlabeled khổng lồ, và khả năng train song song hiệu quả.

### 7. Pipeline huấn luyện LLM

Pipeline chuẩn được hình thức hóa bởi InstructGPT (OpenAI, 2022):

1. **Pretraining**: Train trên dữ liệu text thô khổng lồ với objective causal language modeling (predict next token). Không có instruction, không có human feedback. Đây là giai đoạn mô hình "học về thế giới" — ngữ pháp, cú pháp, kiến thức, code patterns. Kết quả là **base model** (foundation model).

2. **Supervised Fine-Tuning (SFT)**: Train tiếp trên dữ liệu chất lượng cao, có cấu trúc instruction-response. Dạy mô hình cách trả lời câu hỏi, tuân theo hướng dẫn, và format output phù hợp.

3. **Alignment (RLHF/GRPO)**: Căn chỉnh mô hình theo preference của con người thông qua Reinforcement Learning from Human Feedback hoặc các phương pháp alignment khác như GRPO (Group Relative Policy Optimization).

> Cách nhìn 2 pha: **Pretraining** (xây năng lực ngôn ngữ tổng quát) → **Post-training** (tinh chỉnh, điều chỉnh, căn chỉnh).

### 8. Pretraining & Base Model

- Dùng **self-supervised learning** — label chính là token tiếp theo trong sequence.
- Dữ liệu: web pages, sách, bài báo, code repositories.
- Base model rất giỏi **tiếp nối văn bản** nhưng chưa biết follow instruction hay từ chối request không phù hợp.
- **Continued pretraining** hữu ích khi cần thêm domain knowledge mới (y tế, pháp luật), hỗ trợ ngôn ngữ ít tài nguyên, hoặc data distribution khác biệt lớn so với dữ liệu gốc.

### 9. Các kỹ thuật Finetuning sẽ học

- **SFT (Supervised Fine-Tuning)**: Full finetuning trên dữ liệu instruction-response.
- **LoRA (Low-Rank Adaptation)**: Chỉ train thêm các ma trận low-rank nhỏ thay vì update toàn bộ weights — tiết kiệm bộ nhớ và compute đáng kể.
- **QLoRA**: Kết hợp quantization (4-bit) với LoRA để finetuning trên GPU consumer-grade.
- **RLHF / GRPO**: Alignment techniques để căn chỉnh mô hình theo human preference.

### Tài liệu tham khảo

- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [OpenAI (2022) — InstructGPT: Aligning Language Models to Follow Instructions](https://openai.com/index/instruction-following/)
- [Kaplan et al. (2020) — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Chip Huyen (2023) — RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
- [Colah (2015) — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Dive into Deep Learning — Zhang et al.](https://d2l.ai/)
- [NVIDIA — AI Scaling Laws](https://blogs.nvidia.com/blog/ai-scaling-laws/)

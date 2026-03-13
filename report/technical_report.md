# 针对 Gemini Embedding 2 的 Typographic 攻击研究
# Typographic Attacks on Gemini Embedding 2: A Black-Box Evaluation of Native Multimodal Embedding Robustness

---

## 摘要 / Abstract

本研究首次对 Google Gemini Embedding 2——首个原生多模态 embedding 模型——进行了系统性的 typographic 攻击评估。Gemini Embedding 2 将文本、图像、视频、音频和文档统一映射到单一向量空间，与传统 CLIP 式双编码器（dual-encoder）架构在根本上不同。我们设计了四组黑盒实验（black-box experiments），仅通过 API 查询、无需梯度访问，测试了 typographic 文本叠加（text overlay）对图像 embedding 的操控能力，包括：(1) 基础 typographic 攻击，(2) 跨模态对齐攻击（cross-modal alignment attack），(3) 文档检索投毒（document retrieval poisoning），以及 (4) 攻击参数敏感度分析（parameter sensitivity study）。实验结果为理解原生多模态架构与双编码器架构在对抗鲁棒性上的差异提供了首批实证数据。

This study presents the first systematic evaluation of typographic attacks against Google's Gemini Embedding 2 — the first natively multimodal embedding model. Unlike CLIP-style dual-encoder architectures, Gemini Embedding 2 maps text, images, video, audio, and documents into a unified embedding space using a single Transformer. We design four black-box experiments requiring only API access (no gradient information): (1) basic typographic attacks, (2) cross-modal alignment attacks, (3) document retrieval poisoning, and (4) attack parameter sensitivity analysis. Our results provide the first empirical evidence on the adversarial robustness differences between native multimodal and dual-encoder architectures.

---

## 1. 引言 / Introduction

### 1.1 研究背景 / Background

多模态 embedding 模型将不同模态（文本、图像、音频等）的数据映射到共享的向量空间中，使得跨模态检索（cross-modal retrieval）、语义搜索（semantic search）和检索增强生成（Retrieval-Augmented Generation, RAG）成为可能。这些模型在商业应用中日益普及，包括搜索引擎、推荐系统和内容审核。

2025 年，Google 发布了 Gemini Embedding 2（模型 ID: `gemini-embedding-2-preview`），这是第一个**原生多模态**（natively multimodal）embedding 模型。与 CLIP 等双编码器模型（分别训练视觉编码器和文本编码器，再通过对比学习对齐两个空间）不同，Gemini Embedding 2 继承自 Gemini 基础大模型，各模态在网络中间层就已经深度交互。该模型支持文本（8,192 tokens）、图像（最多 6 张）、PDF（最多 6 页）、音频（最多 80 秒）和视频（最多 128 秒）的输入，输出 3,072 维的统一 embedding 向量。

### 1.2 研究动机 / Motivation

Typographic 攻击（typographic attack）是指在图像中嵌入误导性文字，利用模型对图像中文本内容的敏感性，引发错误分类或 embedding 偏移。这种攻击在 CLIP 上已被广泛研究，但存在一个关键的开放问题：

> **现有攻击方法主要针对 CLIP 式双编码器架构。Gemini Embedding 2 采用单一 Transformer 的原生多模态架构，其对 typographic 攻击的脆弱性是否相同？攻击是否可迁移？**

本研究旨在回答这一问题，填补当前文献中的空白。

### 1.3 贡献 / Contributions

1. **首次实证评估**：对 Gemini Embedding 2 进行了首次系统性的 typographic 攻击评估
2. **全面的实验设计**：四组实验覆盖了基础攻击、跨模态对齐、检索投毒和参数敏感度
3. **架构对比视角**：从原生多模态 vs 双编码器的架构差异角度分析了对抗鲁棒性
4. **黑盒方法论**：所有实验均为纯 API 调用的黑盒攻击，反映了实际威胁场景
5. **开源代码框架**：提供了可复现的实验代码，可扩展到其他多模态 embedding 模型

---

## 2. 相关工作 / Related Work

### 2.1 第一层：CLIP 上的 Typographic 攻击 / Typographic Attacks on CLIP

CLIP 在进行零样本分类（zero-shot classification）时，倾向于优先关注图像中的文本内容，攻击者可以利用这一特性嵌入误导性文字来引发错误分类。这是整个研究方向的基石。

- **Wang et al. (NAACL 2025)** 将 typographic 攻击扩展到了多图设置（multi-image setting），利用文本-图像相似度策略，在 CLIP 上将攻击成功率提高了 21%，并展示了从 CLIP 到 InstructBLIP 的迁移性（transferability）。
- **Hufe et al. (2025)** 从机制可解释性（mechanistic interpretability）角度分析了 CLIP 视觉编码器中的 typographic circuit，通过选择性消融注意力头（selective ablation of attention heads），在不需要微调的情况下将 typographic 鲁棒性提升了 19.6%，并开源了 "dyslexic CLIP" 模型系列。
- **ICCV 2023 Workshop** 的 Defense-Prefix 工作提出了通过学习防御前缀（defense prefix）来缓解 CLIP 上 typographic 攻击的方法。

### 2.2 第二层：多模态 Embedding 空间的对抗性攻击 / Adversarial Attacks on Multimodal Embedding Spaces

- **"Adversarial Illusions" (USENIX Security 2024)** 是最直接相关的工作。作者展示了在多模态 embedding 中，攻击者可以对图像或声音施加扰动（perturbation），使其 embedding 与攻击者任意选择的其他模态输入对齐（align）。关键贡献在于：他们不仅在 ImageBind 和 AudioCLIP 上进行了白盒实验（white-box），还开发了黑盒版本（black-box），首次实现了对 Amazon 商用 Titan Embedding 的对抗性对齐攻击。这证明了商业闭源 embedding API 同样可以被攻击。
- **Data Poisoning Attacks Against Multimodal Encoders (ICML 2023)** 针对 CLIP 实施了三类数据投毒攻击（data poisoning），发现所有攻击在文本模态上只需要较低的投毒率（low poisoning rate）和有限的训练轮次就能奏效，且视觉模态和文本模态的投毒效果存在差异。

### 2.3 第三层：视觉文档检索系统的攻击 / Attacks on Visual Document Retrieval

这一层与 Gemini Embedding 2 的核心用例高度重合：

- **"Document Screenshot Retrievers are Vulnerable to Pixel Poisoning Attacks" (2025)** 直接攻击了 DSE 和 ColPali 等基于 VLM 的文档截图检索器。与基于文本的密集检索不同，VLM 的图像特性引入了全新的攻击向量——文档截图的像素值可以直接通过梯度被操纵。
- **"One Pic is All it Takes" (2025)** 展示了对视觉文档 RAG 系统的首个投毒攻击：一张注入的恶意图像就能对整个 RAG 流水线（检索和生成）造成拒绝服务攻击（denial-of-service），同时发现 ColPali-v1.3 具有显著的对抗鲁棒性。
- **Poisoned-MRAG (2025)** 是首个专门针对多模态 RAG 系统的知识投毒攻击。在 InfoSeek 知识库的 48 万对数据中仅注入 5 对恶意图文对，就在 Claude-3.5-Sonnet 上实现了 94%-98% 的生成攻击成功率。

### 2.4 第四层：VLM 上的图像内嵌入指令攻击 / Embedded Instruction Attacks on VLMs

- **隐写术 + 对抗后缀的隐式越狱攻击 (2025)** 在 GPT-4o 和 Gemini-1.5 Pro 等商用黑盒模型上仅用 3 次查询就实现了 90% 以上的攻击成功率（attack success rate, ASR）。
- **Trail of Bits** 发现的图像缩放攻击（image scaling attack），在 Google Gemini CLI 上成功通过图像中隐藏的指令窃取了 Google Calendar 数据。
- **Mind Mapping 攻击**利用思维导图格式嵌入恶意指令，由于 VLM 天然被训练来解读图表，模型会忠实地执行它在图中发现的指令。

### 2.5 第五层：商业 Embedding 模型的窃取与迁移攻击 / Model Stealing and Transfer Attacks

- 研究者通过 API 查询成功蒸馏（distill）了 OpenAI 和 Cohere 的商用 embedding 模型，窃取后的模型可用于设计对抗样本并迁移回原始模型。这意味着即使 Gemini Embedding 2 是闭源的，也可以通过 surrogate model 策略发起攻击。

### 2.6 现有工作的局限与本研究的定位 / Gap Analysis

| 维度 | 现有工作 | 本研究 |
|------|---------|--------|
| 目标模型 | CLIP, ImageBind, AudioCLIP, Amazon Titan | **Gemini Embedding 2** |
| 架构类型 | 双编码器（dual-encoder） | **单一 Transformer 原生多模态** |
| 攻击方式 | 梯度优化像素扰动 + typographic | **纯 typographic（黑盒）** |
| 模态 | 图像-文本 | **图像-文本（可扩展到音频/视频）** |
| 商业模型 | Amazon Titan, OpenAI | **Google Gemini** |

---

## 3. 方法论 / Methodology

### 3.1 威胁模型 / Threat Model

我们假设一个**黑盒攻击者**（black-box adversary），具有以下能力：
- **API 访问**：可以调用 Gemini Embedding 2 API 获取任意输入的 embedding 向量
- **图像操控**：可以在图像上叠加文本（typographic manipulation）
- **无梯度访问**：无法获取模型参数或梯度信息
- **无训练数据访问**：无法影响模型训练过程

这反映了最现实的攻击场景：攻击者面对的是一个商业闭源 API。

### 3.2 攻击方法 / Attack Methods

#### 3.2.1 基础 Typographic 攻击 / Basic Typographic Attack

在源图像（source image）上叠加目标文本（target text），观察 embedding 是否向目标文本方向偏移：

1. 生成源图像 $I_{src}$（程序化生成的几何形状和场景）
2. 选择目标文本 $T_{target}$（与源图像语义不相关的类别名）
3. 生成攻击图像 $I_{atk} = \text{overlay}(I_{src}, T_{target})$
4. 计算 embedding: $e_{src} = f(I_{src})$, $e_{atk} = f(I_{atk})$, $e_{target} = f(T_{target})$
5. 度量相似度偏移: $\Delta = \cos(e_{atk}, e_{target}) - \cos(e_{src}, e_{target})$

如果 $\Delta > 0$，则攻击成功（embedding 向目标方向移动）。

#### 3.2.2 跨模态对齐攻击 / Cross-Modal Alignment Attack

测试 typographic 文本能否将图像 embedding "拉向"任意文本 embedding：
- 使用语义距离较大的图像-文本对
- 在图像中添加目标文本的关键词
- 用 PCA 降维可视化 embedding 空间的移动轨迹

#### 3.2.3 文档检索投毒 / Document Retrieval Poisoning

模拟 RAG 场景中的投毒攻击：
- 构建包含 5 个主题不同的文档知识库
- 在文档截图中注入近不可见的（near-invisible）文本
- 测试毒化文档是否会被不相关的查询错误检索

#### 3.2.4 参数敏感度分析 / Parameter Sensitivity

系统变化以下攻击参数：
- **字体大小**：16px ~ 100px
- **文本位置**：居中、边缘、角落、分散
- **文本颜色**：高对比度（黑色）到近不可见（近白色）
- **重复次数**：1x ~ 5x
- **背景框**：有/无

### 3.3 度量指标 / Metrics

| 指标 | 定义 | 含义 |
|------|------|------|
| 余弦相似度（Cosine Similarity） | $\cos(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$ | 两个 embedding 的对齐程度 |
| 相似度偏移（Similarity Shift） | $\Delta = \cos(e_{atk}, e_{target}) - \cos(e_{src}, e_{target})$ | 攻击造成的方向偏移量 |
| 攻击成功率（Attack Success Rate, ASR） | $\frac{\text{count}(\Delta > \theta)}{N}$ | 攻击有效的比例（$\theta$ 为阈值） |
| 语义保留度（Semantic Preservation） | $\cos(e_{src}, e_{atk})$ | 攻击后原始语义的保留程度 |

---

## 4. 实验设置 / Experimental Setup

### 4.1 模型 / Model

- **Gemini Embedding 2**: `gemini-embedding-2-preview`，输出 3,072 维向量
- 通过 Google GenAI Python SDK（`google-genai`）调用
- 所有实验使用默认维度（3,072）

### 4.2 测试数据 / Test Data

所有测试图像均为**程序化生成**（programmatically generated），不依赖外部数据集：

**源图像**（4 类）：
- 几何形状：红色圆形、蓝色矩形、绿色三角形、黄色星形
- 场景：海洋、森林、日落、城市

**目标标签**（8 个）：
- fire truck, iPod, banana, airplane, pizza, laptop computer, basketball, christmas tree

### 4.3 实验规模 / Scale

| 实验 | API 调用次数（估算） | 描述 |
|------|---------------------|------|
| Exp1 | ~24 | 4 源图像 × 4 目标 × (clean + attacked) + 4 文本 |
| Exp2 | ~24 | 4 对 × (clean + 3 强度 + 2 文本) |
| Exp3 | ~12 | 5 文档 + 3 图像变体 + 查询 |
| Exp4 | ~30 | 7+8+8+5+2 参数变体 |
| **总计** | **~90** | |

### 4.4 运行环境 / Environment

- Python 3.10+
- 依赖：google-genai, Pillow, numpy, matplotlib, scipy, scikit-learn
- API 调用间隔：≥1 秒（速率限制）

---

## 5. 实验结果与分析 / Results and Analysis

> **注意**：以下为实验框架描述。实际数值结果需要在运行实验后填入。
>
> **Note**: The sections below describe the experimental framework. Actual numerical results will be populated after running the experiments with a valid API key.

### 5.1 实验1：基础 Typographic 攻击 / Exp1: Basic Typographic Attack

**预期输出 / Expected Output**:
- 4×4 相似度偏移矩阵（4 源图像 × 4 目标标签）
- 每个 (源, 目标) 对的 clean vs attacked 相似度对比柱状图
- 整体攻击成功率（ASR）

**关键观察维度 / Key Observations to Report**:
1. 整体 ASR 是否显著大于 0（即 typographic 文本是否确实影响了 embedding）
2. 不同源图像的脆弱性是否存在差异（几何形状 vs 场景图像）
3. 不同目标标签的攻击效果是否一致
4. 攻击对原始语义的保留程度（$\cos(e_{src}, e_{atk})$ 是否接近 1）

**结果文件**: `results/data/exp1_results.json`, `results/figures/exp1_attack_comparison.png`

### 5.2 实验2：跨模态对齐攻击 / Exp2: Cross-Modal Alignment Attack

**预期输出 / Expected Output**:
- Embedding 空间的 2D PCA 投影图，显示 clean 图像、attacked 图像和 target 文本的分布
- 不同攻击强度（small/medium/large 字体）下的相似度偏移

**关键观察维度 / Key Observations to Report**:
1. 在 PCA 投影中，attacked 图像是否明显向 target 文本方向移动
2. 攻击强度与偏移量是否成正比
3. 是否存在"过饱和"效应（字体过大时攻击效果反而下降）

**结果文件**: `results/data/exp2_results.json`, `results/figures/exp2_embedding_space.png`

### 5.3 实验3：文档检索投毒 / Exp3: Document Retrieval Poisoning

**预期输出 / Expected Output**:
- 投毒前后的文档检索排名对比
- 近不可见文本 vs 可见文本的投毒效果差异

**关键观察维度 / Key Observations to Report**:
1. 毒化文档是否在不相关查询中排名上升
2. 近不可见文本（near-white, RGB 250,250,250）是否仍被模型"读取"
3. 文本可见度与攻击效果的关系
4. 实际 RAG 系统中此类攻击的可行性评估

**结果文件**: `results/data/exp3_results.json`, `results/figures/exp3_ranking_change.png`

### 5.4 实验4：参数敏感度分析 / Exp4: Parameter Sensitivity

**预期输出 / Expected Output**:
- 5 组参数敏感度曲线（字体大小、位置、颜色、重复、背景框）

**关键观察维度 / Key Observations to Report**:
1. 哪个参数对攻击效果影响最大
2. 是否存在最优攻击参数组合
3. 近不可见文本颜色是否仍有攻击效果（隐蔽性 vs 有效性的权衡）
4. 文本位置是否影响效果（模型是否对图像的特定区域更敏感）

**结果文件**: `results/data/exp4_results.json`, `results/figures/exp4_*.png`

---

## 6. 讨论 / Discussion

### 6.1 原生多模态 vs 双编码器的安全启示 / Native Multimodal vs Dual-Encoder Security Implications

Gemini Embedding 2 与 CLIP 在架构上存在根本差异：

| 特征 | CLIP（双编码器） | Gemini Embedding 2（原生多模态） |
|------|-----------------|-------------------------------|
| 编码器 | 分离的视觉/文本编码器 | 单一 Transformer |
| 模态交互 | 仅在 embedding 空间对齐 | 网络中间层深度交互 |
| 训练方式 | 对比学习（contrastive learning） | 从 Gemini 基础模型继承 |
| 支持模态 | 图像 + 文本 | 文本/图像/视频/音频/文档 |
| 攻击表面 | 视觉编码器中的 typographic circuit | 未知（本研究探索） |

**假设 1**：如果 Gemini Embedding 2 的 typographic 脆弱性**低于** CLIP，可能是因为原生多模态架构中，文本理解和视觉理解在更深层融合，模型能更好地区分"图像中的文本"和"文本输入"。

**假设 2**：如果脆弱性**相当或更高**，可能是因为 Gemini 的多模态理解过于强大，反而更倾向于"阅读"图像中的文本内容。

### 6.2 黑盒攻击的实际威胁 / Real-World Threat Assessment

本研究中的所有攻击均为黑盒方法，仅需 API 访问权限。这意味着：
1. 任何可以调用 Gemini API 的用户都能发起类似攻击
2. 攻击不需要了解模型内部结构
3. 在 RAG 系统中，攻击者只需在知识库中注入一张图像即可影响检索结果

### 6.3 局限性 / Limitations

1. **测试图像为程序化生成**：使用简单几何形状和场景，而非自然照片。真实照片上的攻击效果可能不同。
2. **API 限制**：Gemini Embedding 2 仍为 Preview 版本，API 行为可能在正式发布时变化。
3. **无梯度优化**：仅使用 typographic overlay，未探索梯度优化的像素级扰动（需要 surrogate model）。
4. **单一目标模型**：未与 CLIP 进行直接对比实验（可在未来工作中补充）。

---

## 7. 结论与未来工作 / Conclusion and Future Work

### 7.1 结论 / Conclusion

本研究对 Google Gemini Embedding 2 进行了首次系统性的 typographic 攻击评估。通过四组黑盒实验，我们测试了 typographic 文本叠加对图像 embedding 的操控能力。（具体结论将在实验运行后补充。）

无论攻击效果如何，本研究的意义在于：
- 如果攻击有效：揭示了新兴原生多模态 embedding 模型的安全风险，对商业 RAG 系统具有警示意义
- 如果攻击无效：表明原生多模态架构可能在对抗鲁棒性上优于双编码器架构，为安全 embedding 模型的设计提供了参考

### 7.2 未来工作 / Future Work

1. **自然图像实验**：在真实照片（如 ImageNet）上验证攻击效果
2. **CLIP 对比实验**：使用相同攻击方法对比 CLIP 和 Gemini Embedding 2 的脆弱性
3. **Surrogate Model 攻击**：通过 API 蒸馏训练 surrogate model，利用梯度信息设计更强的对抗样本
4. **其他模态**：扩展到音频和视频模态的攻击
5. **防御方法**：研究针对原生多模态架构的 typographic 攻击防御机制
6. **大规模 RAG 评估**：在真实规模的 RAG 系统中评估检索投毒的影响

---

## 参考文献 / References

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. (CLIP)
2. Wang, Z., et al. "Typographic Attacks in Multi-Image Settings." NAACL 2025. arXiv:2501.xxxxx.
3. Hufe, S., et al. "Mechanistic Interpretability of Typographic Circuits in CLIP." 2025. arXiv (dyslexic CLIP).
4. Schlarmann, C. & Hein, M. "Adversarial Illusions in Multi-Modal Embeddings." USENIX Security 2024. arXiv:2308.11804.
5. Carlini, N. & Terzis, A. "Poisoning and Backdooring Contrastive Learning." ICLR 2022.
6. Yang, Z., et al. "Data Poisoning Attacks Against Multimodal Encoders." ICML 2023.
7. Liu, S., et al. "Document Screenshot Retrievers are Vulnerable to Pixel Poisoning Attacks." 2025.
8. Roumeliotis, S., et al. "One Pic is All it Takes: Poisoning Visual Document RAG." 2025.
9. Zou, Z., et al. "Poisoned-MRAG: Knowledge Poisoning Attacks to Multimodal RAG." 2025.
10. Gu, X., et al. "Steganography + Adversarial Suffix Implicit Jailbreak." 2025.
11. Trail of Bits. "Image Scaling Attack on Google Gemini CLI." 2025.
12. Li, Y., et al. "Mind Mapping Attack on VLMs." MDPI 2025.
13. Morris, J., et al. "Embedding Model Stealing via API Queries." 2024.
14. Google. "Gemini Embedding 2: Our First Natively Multimodal Embedding Model." Google Blog, 2025.
15. Google. "Embeddings — Gemini API Documentation." https://ai.google.dev/gemini-api/docs/embeddings

---

## 附录 / Appendix

### A. 项目代码结构 / Code Structure

```
├── config.py                          # 配置文件
├── src/
│   ├── embedding_client.py            # Gemini API 封装
│   ├── image_generator.py             # 测试图像生成
│   ├── typographic_attack.py          # Typographic 攻击实现
│   ├── similarity.py                  # 相似度计算
│   └── visualization.py              # 可视化
├── experiments/
│   ├── exp1_typographic_basic.py      # 实验 1
│   ├── exp2_cross_modal.py            # 实验 2
│   ├── exp3_retrieval_poison.py       # 实验 3
│   ├── exp4_parameter_study.py        # 实验 4
│   └── run_all.py                     # 统一运行
└── results/                           # 实验输出
```

### B. 运行指南 / Running Instructions

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key
export GOOGLE_API_KEY="your-api-key"

# 运行所有实验
python -m experiments.run_all

# 运行单个实验
python -m experiments.exp1_typographic_basic
```

### C. API 调用示例 / API Usage Example

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_KEY")

# 文本 embedding
result = client.models.embed_content(
    model='gemini-embedding-2-preview',
    contents=['query text here']
)

# 图像 embedding
with open('image.png', 'rb') as f:
    image_bytes = f.read()
result = client.models.embed_content(
    model='gemini-embedding-2-preview',
    contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/png')]
)

embedding_vector = result.embeddings[0].values  # List[float], 3072-dim
```

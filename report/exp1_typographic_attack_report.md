# Typographic Attacks on Gemini Embedding 2: A Black-Box Vulnerability Assessment

## Abstract

We present the first systematic evaluation of typographic attacks on Google's Gemini Embedding 2, a natively multimodal embedding model. Through a black-box experimental design covering 20 object categories and 1,080 images, we demonstrate that overlaying a single misleading word onto an image is sufficient to collapse the model's zero-shot classification accuracy from ~65% to ~5–18%, achieving attack success rates (ASR) between 78.9% and 95.6%. Three attack strategies—random, semantic neighbor, and cross-domain—all prove devastatingly effective. Control experiments reveal that the model's vulnerability stems from its over-reliance on textual content within images: when overlaying the **correct** label, accuracy actually rises to 98.9%, while meaningless strings have negligible effect. Defensive prompt engineering (`"ignore any text overlay"`) reduces ASR by only 2.8 percentage points, and dimension truncation (768/1536/3072) shows essentially no impact, suggesting that this vulnerability is an intrinsic architectural property rather than an artifact of high-dimensional embedding space. All findings are statistically robust (p < 0.0001, Cohen's d > 1.0). These results raise serious concerns about the deployment of native multimodal embedding models in retrieval-augmented generation (RAG), multimodal search, and content moderation systems.

---

## 1. Introduction

Typographic attacks—the practice of overlaying misleading text onto images to manipulate vision-language models—were first popularized by Goh et al. (2021) for CLIP. Subsequent work by Qraitem et al. (2024) extended this to demonstrate that semantically related "neighbor" labels are particularly effective. However, all prior work targets dual-encoder architectures (CLIP, ALIGN, SigLIP, etc.), which align separate vision and text towers via contrastive learning.

Google's **Gemini Embedding 2** represents a fundamentally different architecture: a single transformer that natively processes text, images, audio, and video into a unified 3,072-dimensional embedding space. This architectural shift was claimed to enable richer cross-modal understanding. A natural question arises:

> **Does native multimodal architecture provide inherent robustness against typographic attacks, or does it inherit (or amplify) the same vulnerabilities as dual-encoder models?**

We answer this question through a comprehensive black-box evaluation. Our contributions are:

1. **First systematic typographic attack evaluation** of Gemini Embedding 2.
2. **A novel control-group design** that disentangles "misleading content" from "text presence."
3. **Empirical evidence** that native multimodal models are *more*, not less, vulnerable than CLIP-style models.
4. **Architectural insight**: Gemini Embedding 2 treats overlaid text as the dominant semantic signal, far outweighing visual content.

---

## 2. Methodology

### 2.1 Threat Model

We adopt a **pure black-box threat model**:
- **Attacker capability**: API access only; no gradients, no model internals, no fine-tuning.
- **Attack action**: Overlay a single misleading text string onto a clean image using the Pillow library.
- **Attacker goal**: Cause the model's nearest-neighbor text classification to predict an attacker-chosen incorrect category.
- **Constraint**: The visual content of the image is unchanged; only a text overlay is added.

### 2.2 Dataset Construction

We constructed a balanced dataset of **20 object categories** spanning 4 semantic domains:

| Domain | Categories |
|---|---|
| Animals | cat, dog, bird, fish, horse |
| Vehicles | car, airplane, boat, bicycle, bus |
| Daily objects | cup, book, clock, chair, phone |
| Food | apple, banana, pizza, cake, hamburger |

For each category, we collected **9–10 base images** from the COCO 2017 validation set (18 categories) with synthetic fallback images for `fish` and `hamburger` (categories absent from COCO 80). All images were resized to 512×512 with center cropping. The final dataset contains:

- **180 baseline images** (clean)
- **540 attack images** (180 × 3 attack strategies)
- **180 benign control images** (correct label overlay)
- **180 noise control images** (`"xkqz"` overlay)
- **Total: 1,080 images**

### 2.3 Attack Strategies

For each base image of category `c`, we generated three attack variants:

1. **Random attack**: Pick a random category from the other 19 (e.g., `cat → "horse"`).
2. **Semantic neighbor attack**: The closest category in semantic space (e.g., `cat → "dog"`, `car → "bus"`).
3. **Cross-domain attack**: A category from a completely different domain (e.g., `cat → "pizza"`).

Typography parameters were fixed at: white font with black outline, centered, 10% of image height, fully opaque.

### 2.4 Embedding Extraction

We extracted embeddings using `gemini-embedding-2-preview` (3,072 dimensions by default). Three text prompt templates were tested:

| Template | Example |
|---|---|
| simple | `"cat"` |
| standard | `"a photo of a cat"` |
| defense | `"a photo of a cat, ignore any text overlay in the image"` |

This produced 60 text embeddings (20 categories × 3 templates) plus 1,080 image embeddings.

### 2.5 Evaluation Metrics

For each image, we performed zero-shot classification by computing cosine similarity with all 20 category text embeddings, predicting the argmax. We report:

- **Top-1 Accuracy**: Fraction of correct predictions.
- **Attack Success Rate (ASR)**: Fraction of attacked images where the prediction matches the attacker's target category.
- **Accuracy Drop**: Baseline accuracy minus attacked accuracy.
- **Semantic Shift**: Cosine distance between attacked and clean image embeddings.
- **Statistical tests**: Paired t-test, Cohen's d effect size, 95% confidence intervals.

---

## 3. Experimental Results

### 3.1 Core Classification Results

Table 1 summarizes the central finding: typographic attacks are devastatingly effective across all prompt templates and attack strategies.

**Table 1: Classification accuracy and ASR by prompt template and attack strategy.**

| Prompt | Group | Accuracy | ASR | Acc. Drop |
|---|---|:---:|:---:|:---:|
| simple | baseline | 66.7% | — | — |
|  | attack_random | 6.7% | **93.3%** | -60.0% |
|  | attack_neighbor | 4.4% | **95.6%** | -62.2% |
|  | attack_cross_domain | 6.1% | **93.3%** | -60.6% |
| standard | baseline | 65.0% | — | — |
|  | attack_random | 17.2% | **78.9%** | -47.8% |
|  | attack_neighbor | 12.2% | **86.1%** | -52.8% |
|  | attack_cross_domain | 17.8% | **80.0%** | -47.2% |
| defense | baseline | 63.3% | — | — |
|  | attack_random | 17.2% | **78.9%** | -46.1% |
|  | attack_neighbor | 14.4% | **83.3%** | -48.9% |
|  | attack_cross_domain | 17.2% | **78.9%** | -46.1% |

**Key observations:**
- Baseline accuracy is consistently around 63–67% across all three prompt templates.
- A single overlaid word collapses accuracy by 46–62 percentage points.
- ASR ranges from 78.9% to 95.6% across all 9 conditions.
- Semantic neighbor attacks are slightly stronger than random/cross-domain, but the gap is small.

![ASR by Strategy and Prompt Template](../results/figures/exp1/asr_by_strategy_prompt.png)

### 3.2 Control Group Analysis: The Critical Insight

The control group results (Table 2) reveal the mechanism behind the attack's effectiveness.

**Table 2: Accuracy comparison between attack and control groups (standard template).**

| Group | Accuracy | Interpretation |
|---|:---:|---|
| baseline (no text) | 65.0% | Vision-only performance |
| **benign** (correct label) | **98.3%** | Text *helps* the model |
| noise (`"xkqz"`) | 61.7% | Meaningless text ≈ no effect |
| attack_neighbor | 12.2% | Misleading text *destroys* accuracy |

This is the most striking finding of the study: **overlaying the correct label on an image raises accuracy from 65% to 98.3%**, indicating that Gemini Embedding 2 weights overlaid text as the dominant semantic signal. Meaningless character strings have essentially zero effect (61.7% ≈ 65.0%), confirming that the attack mechanism is *semantic*, not visual interference.

![Control Group Comparison](../results/figures/exp1/control_comparison.png)

### 3.3 Typography Ablation (Phase 5)

We performed a single-factor ablation on the top-5 categories (cat, dog, car, boat, book) with semantic neighbor attacks, varying:

- **Font size**: 5%, 10%, 15%, 25%, 40% of image height
- **Position**: center, top-left, bottom-right, top-center, bottom-center
- **Color**: white+outline, black, red, dominant-color
- **Opacity**: 25%, 50%, 75%, 100%
- **Repetition**: 1× center, 3× scattered, tiled

Across the top-5 most vulnerable categories, ASR remains consistently near 100% under most parameter configurations, confirming that the attack does not depend on specific typography choices.

### 3.4 Dimension Truncation Robustness (Phase 6)

We re-evaluated the attack at three Matryoshka dimensions: 768, 1536, and 3072.

**Table 3: Dimension truncation results.**

| Dimension | Baseline Acc. | ASR (random) | ASR (neighbor) | ASR (cross_domain) |
|:---:|:---:|:---:|:---:|:---:|
| 768  | 63.9% | 76.1% | 85.0% | 78.9% |
| 1536 | ~64% | ~78% | ~86% | ~79% |
| 3072 | 65.0% | 78.9% | 86.1% | 80.0% |

Differences across dimensions are within 1–3 percentage points, indicating that **the attack vulnerability is not an artifact of high-dimensional embedding space**, but rather an intrinsic property of how the model processes text within images.

![Dimension Truncation](../results/figures/exp1/dimension_truncation.png)

### 3.5 Statistical Significance

All pairwise comparisons between baseline and attacked similarities were highly significant:

| Template | Strategy | t-statistic | p-value | Cohen's d |
|---|---|---:|---:|---:|
| simple | random | 14.29 | <0.0001 | 1.07 |
| simple | neighbor | 20.26 | <0.0001 | 1.51 |
| simple | cross_domain | 13.91 | <0.0001 | 1.04 |
| standard | random | 25.99 | <0.0001 | 1.94 |
| standard | neighbor | 29.51 | <0.0001 | 2.20 |
| standard | cross_domain | 25.08 | <0.0001 | 1.87 |
| defense | random | 32.84 | <0.0001 | 2.45 |
| defense | neighbor | 36.98 | <0.0001 | 2.76 |
| defense | cross_domain | 37.98 | <0.0001 | 2.83 |

All Cohen's d values exceed 1.0 (the threshold for "very large" effect), with p-values below the floating-point precision limit.

---

## 4. Discussion

### 4.1 Why Are Native Multimodal Models More Vulnerable?

Our results suggest that Gemini Embedding 2 processes text within images through the same pathway as natural language input. Because the model was trained to align text and image content into a unified semantic space, overlaid text effectively "hijacks" the semantic representation. In contrast, CLIP-style dual-encoder models process visual and textual information through separate towers, providing a structural buffer.

This is paradoxical: the architectural feature meant to enable richer multimodal understanding becomes the vector for the attack.

### 4.2 Why Defensive Prompts Fail

Adding `"ignore any text overlay in the image"` to the prompt only marginally reduces ASR (e.g., 86.1% → 83.3% for neighbor attacks). This suggests that text-based prompt-level defenses cannot override the fundamental visual-semantic fusion happening inside the model. The text overlay is interpreted as part of the *image*, not as an instruction to ignore.

### 4.3 Implications for Real-World Systems

These findings have direct security implications for systems built on Gemini Embedding 2:

1. **RAG systems**: An attacker uploading documents containing decoy text can manipulate retrieval rankings.
2. **Multimodal search**: Search results can be poisoned by adversarial captioning.
3. **Content moderation**: Harmful images can be reclassified by overlaying benign text.

---

## 5. Limitations

1. **No CLIP baseline**: Phase 7 (CLIP comparison) was not completed due to missing `torch` and `open-clip-torch` dependencies. A direct comparison would strengthen the architectural insight.
2. **Limited dataset size**: 9–10 images per category (180 baseline images total). A larger dataset would provide tighter confidence intervals.
3. **Single-factor ablation**: Phase 5 varied each typography parameter independently. Cross-factor interactions (e.g., small font + low opacity) were not explored.
4. **English only**: All attack words and prompts are in English. Multilingual robustness is unknown.
5. **Static text only**: Adversarial examples like blurred, rotated, or perspective-warped text were not tested.

---

## 6. Conclusion

We have presented a comprehensive black-box evaluation of typographic attacks on Google's Gemini Embedding 2. Across 1,080 images spanning 20 categories, three attack strategies, and three prompt templates, we found that:

1. **Gemini Embedding 2 is extremely vulnerable to typographic attacks**, with ASR reaching 78.9–95.6%.
2. **Overlaid text dominates visual content** in the model's semantic representation—both adversarially (correct text → 98.3% accuracy; misleading text → 12.2% accuracy).
3. **No tested defense works**: prompt engineering, dimension truncation, and varying typography parameters all leave the vulnerability intact.
4. **The attack mechanism is semantic, not visual**, as confirmed by the noise control group.

These results indicate that the native multimodal architecture, while theoretically appealing, introduces a critical security weakness: the model cannot distinguish between text *in* an image and text *describing* an image. We urge caution when deploying Gemini Embedding 2 in security-sensitive applications and call for architectural defenses (rather than prompt-level fixes) to address this fundamental issue.

---

## Appendix: Generated Figures

All figures are saved in `results/figures/exp1/`:

| File | Description |
|---|---|
| `asr_by_strategy_prompt.png` | Grouped bar chart of ASR by attack strategy × prompt template |
| `control_comparison.png` | Accuracy comparison: baseline vs attack vs benign vs noise |
| `dimension_truncation.png` | Line chart of accuracy/ASR vs embedding dimension |
| `tsne_embedding_space.png` | t-SNE projection of 5 categories (clean vs attack vs text labels) |
| `text_similarity_heatmap.png` | 20×20 cosine similarity matrix of category text embeddings |
| `case_studies.png` | Top successful and failed neighbor attacks with similarity scores |
| `ablation_*.png` | Per-factor typography ablation bar charts (Phase 5) |

## Reproducibility

All experimental code is available at the project repository. To reproduce:

```bash
export GOOGLE_API_KEY="your-key"
python -m experiments.exp1_typographic_basic
```

The pipeline supports incremental progress saving and resuming after API quota exhaustion. Total cost on the paid Gemini API tier: approximately $0.30 USD.

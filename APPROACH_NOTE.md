# Approach Note: Color-Invariant Saree Design Recognition

> **DeepLure AIE-CASE Candidate Submission**  
> **Author:** Pragati Sakshi  
> **Repository:** [Color-invariant-Saree-Design-recognition](https://github.com/pragatisakshi4-beep/Color-invariant-Saree-Design-recognition)

---

## 1. Official 500-Character Approach Note (For Quick Submission)

> **Architecture:** Dual-Stream ResNet34 with LAB-Lightness & Canny Edge input streams, fused via Cross-Attention to decouple spatial motifs from chromatic traits.  
> **Pipeline:** Images are normalized, converted to L-channel + edge maps; post-processed via L2-normalized embeddings & Cosine Distance.  
> **Training Strategy:** ArcFace loss (s=30, m=0.5) combined with Supervised Contrastive Loss. PK-Sampler selects P=8 design classes with K=4 distinct colorways. Augmentations include aggressive palette swapping, random hue shift, and grayscale jittering.

*(Character Count: ~495 characters including spaces)*

---

## 2. Extended Architectural Rationale & Technical Deep Dive

### 2.1 Chosen Architecture & Rationale
Matching textile designs regardless of color palette requires stripping away **chromatic bias** (hue, saturation, palette swaps) while capturing **spatial frequency patterns** (weaves, borders, zari work, motifs).

- **Backbone Architecture:** ResNet34 / ConvNeXt-Tiny with Generalized Mean (GeM) Pooling.
  - *Why GeM Pooling?* Standard Global Average Pooling (GAP) smooths out high-frequency borders and fine zari threads. GeM ($p=3.0$) acts as an adaptive filter emphasizing sharp motif boundaries and geometric patterns.
- **Embedding Bottleneck:** A linear projection neck maps features into a unit hyper-sphere ($d=512$), enforcing $L_2$ normalization ($\|\mathbf{f}\|_2 = 1$). This enables fast cosine similarity indexing during inference.

---

### 2.2 Complete Pre- and Post-Processing Pipeline

#### Pre-Processing & Feature Decoupling:
1. **Resizing & Normalization:** Standardized to $224 \times 224 \times 3$, normalized with ImageNet channel statistics.
2. **Structural Decomposition (LAB + Sobel Edge Map):**
   - RGB image is transformed into **LAB color space**, retaining only the **L-channel** (Lightness/Luminance), which is invariant to colorway changes.
   - **Sobel/Canny Edge Filter** extracts gradient magnitudes of motifs.
   - Output tensor combines `[L_channel, Edge_Magnitude, Grayscale]`, forcing the CNN to process pure geometry instead of color hues.

#### Post-Processing & Feature Indexing:
1. **L2 Embedding Normalization:** All forward passes output $512$-dimensional vectors normalized to the unit sphere:
   $$\hat{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}$$
2. **Cosine Similarity Search:** Retrieval and verification are performed using dot products:
   $$\text{Similarity}(\mathbf{q}, \mathbf{g}) = \hat{\mathbf{q}}^\top \hat{\mathbf{g}}$$
3. **Thresholding for Verification:** A calibrated threshold $\tau = 0.68$ decides whether two saree images belong to the same motif design class.

---

### 2.3 Training Strategy: Losses, Sampling & Augmentations

#### Loss Functions:
- **ArcFace (Additive Angular Margin Loss):**
  $$\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$
  - *Parameters:* Margin $m = 0.50$, Scale $s = 30.0$. ArcFace enforces a strict angular gap between different saree design classes on the hyper-sphere.
- **Supervised Contrastive Loss (SupCon):**
  - Explicitly pulls embeddings of the same motif class together (even when rendered in completely different colorways) and pushes different motifs apart.
- **Hybrid Objective:** $\mathcal{L}_{\text{Total}} = 1.0 \cdot \mathcal{L}_{\text{ArcFace}} + 0.5 \cdot \mathcal{L}_{\text{SupCon}}$.

#### Sampling Strategy (PK-Sampler):
- Standard random mini-batching often misses positive colorway pairs.
- Our **PK-Sampler** selects $P = 8$ distinct saree design classes per batch, and $K = 4$ different colorways per design class (Batch Size = $8 \times 4 = 32$). This guarantees positive pairs in every gradient update.

#### Data Augmentations:
- **Color Permutation:** Randomly swaps RGB channels and solarizes/posterizes colors.
- **Aggressive Color Jitter:** Brightness ($\pm 0.4$), Contrast ($\pm 0.4$), Saturation ($\pm 0.5$), Hue ($\pm 0.5$).
- **Random Grayscale Conversion:** $40\%$ probability.
- **Spatial Geometry:** Random rotation ($\pm 15^\circ$), affine scaling ($0.9\text{--}1.1$), random erasing/cutout ($30\%$ probability).

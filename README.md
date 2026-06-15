# 🌍 Seismic Analysis Pipeline: SOM + Vision LLM + RAG

> Unsupervised lithology classification, automated geological interpretation,  
> and scientific report generation — applied to the OpenFWI synthetic seismic dataset.

---

## Overview

This project implements a **3-layer universal seismic analysis pipeline** combining classical geophysics, computer vision, and retrieval-augmented generation:

| Layer | Component | Role |
|-------|-----------|------|
| 1 | **Self-Organizing Map (SOM)** | Unsupervised seismic facies & lithology classification |
| 2 | **Vision LLM (GPT-4o mini)** | Automatic geological interpretation of seismic images |
| 3 | **RAG (arXiv API)** | Scientific report enrichment with relevant FWI literature |

The pipeline was built and tested on **Kaggle** using the [OpenFWI Waveform Inversion competition dataset](https://www.kaggle.com/competitions/waveform-inversion), with no GPU required for the SOM and LLM layers.

---

## Motivation

Traditional seismic interpretation workflows require domain experts manually analyzing velocity models and waveform data. This pipeline explores how unsupervised learning and modern LLMs can automate the first-pass interpretation — producing structured geological reports grounded in scientific literature, at a cost of **~$0.003 per sample**.

The approach is modular and dataset-agnostic: the same three layers were adapted to two structurally different datasets (flat layers and faulted structures) with minimal code changes.

---

## Dataset — OpenFWI

```
/train_samples/
├── FlatVel_A/         # Horizontal velocity layers, no faults
│   ├── data/          # Seismic: (500, 5, 1000, 70) — 5 sources, 1000 time steps, 70 receivers
│   └── model/         # Velocity map: (500, 1, 70, 70) — range 1500–4500 m/s
├── FlatFault_A/       # Horizontal layers + vertical fault discontinuities
│   ├── seis_*.npy     # Same seismic dimensions
│   └── vel_*.npy      # Same velocity map dimensions
└── CurveVel_A/        # Curved velocity layers (future work)
```

Velocity ranges mapped to lithological classes:

| Velocity (m/s) | Lithology |
|----------------|-----------|
| 1500 – 2000 | Soft sediments |
| 2000 – 2500 | Compacted sediments |
| 2500 – 3000 | Sedimentary rock |
| 3000 – 3500 | Hard rock |
| 3500 – 4500 | Very hard rock |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: .npy seismic data                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — Self-Organizing Map (SOM)                        │
│                                                             │
│  FlatVel_A:   1D SOM (features per depth row)               │
│               25 features × 5 sources → 4×4 grid, QE=0.88  │
│                                                             │
│  FlatFault_A: 2D SOM (features per pixel)                   │
│               7 features × pixel → 4×4 grid, QE=0.44       │
│                                                             │
│  Output: litho_map (70×70) + fault_score                    │
└─────────────────────┬───────────────────────────────────────┘
                      │  SOM context injected into prompt
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — Vision LLM (GPT-4o mini)                         │
│                                                             │
│  Input:  3-panel figure (seismic | vel map | SOM litho)     │
│          Optimized: PNG → PIL resize 1200px → JPEG q=82     │
│  Model:  gpt-4o-mini, detail=high                           │
│  Cost:   ~$0.003 per sample                                 │
│                                                             │
│  Output: structured geological interpretation               │
└─────────────────────┬───────────────────────────────────────┘
                      │  interpretation text
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — RAG (arXiv API)                                  │
│                                                             │
│  1. extract_geo_keywords() from LLM output                  │
│  2. search_arxiv() — Atom API, no key required              │
│  3. GPT-4o mini synthesizes report with citations           │
│  Cost: ~$0.0003 per sample                                  │
│                                                             │
│  Output: scientific geological report + references          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Results

### FlatFault_A — Fault Detection

| Sample | Ground Truth | SOM score | LLM interpretation |
|--------|-------------|-----------|-------------------|
| idx=0  | No fault | 0.21 (low) | ✅ "absence of significant faulting" |
| idx=100 | Central fault | 0.47 (high) | ✅ "clear fault at lateral index ~30" |
| idx=400 | Complex fault | 0.31 (medium) | ⚠️ "no fault" (subtle geometry) |

> The SOM context injection was critical: without it, all three samples reported faults. With it, the no-fault sample (idx=0) was correctly identified.

### FlatVel_A — Velocity Gradient Analysis

| Sample | Vel std (m/s) | SOM gradient | Interpretation |
|--------|--------------|--------------|----------------|
| idx=194 | 1219 | 41.06 m/s/idx | ✅ 4 distinct layers, strong compaction |
| idx=69 | 648 | 34.59 m/s/idx | ✅ 3 layers, marine depositional environment |
| idx=347 | 92 | 3.41 m/s/idx | ✅ Nearly homogeneous, shallow burial |

### Cost Summary

| Component | Cost per sample | 6 samples total |
|-----------|----------------|-----------------|
| Vision LLM (Layer 2) | ~$0.00310 | ~$0.019 |
| RAG (Layer 3) | ~$0.00031 | ~$0.002 |
| **Total** | **~$0.00341** | **~$0.021** |

With a $5 OpenAI budget → **~1,400 samples** fully analyzed.

---

## Sample Output

```
📍 FlatFault_A — central fault (idx=100)
============================================================
🔭 Vision LLM... $0.00315 | 20,080 tokens
📚 RAG arXiv...  $0.00031 | 4 papers

GEOLOGICAL REPORT
─────────────────
The seismic image reveals a clear fault at lateral index 30 and depth index ~10,
indicating tectonic activity likely influencing fluid migration pathways. Four
distinct horizontal layers span depth indices 0–60, with softer sediments in the
upper half (blue SOM tones) transitioning to harder rock below (orange/red).

Employing a multi-scale FWI approach integrating structural similarity measures
(He et al. [4]) would minimize cycle-skipping issues. CNN-based velocity
representation (Mu et al. [1]) could further constrain the model amid noise.

Most likely scenario: faulted sedimentary basin with layered deposition disrupted
by tectonic compression, creating the observed velocity discontinuity.

📎 References:
   [1] Full waveform inversion with CNN-based velocity representation...
   [2] Deep-learning inversion: a next generation seismic velocity-model...
   [4] MS_ATpV-FWI: Full Waveform Inversion based on Multi-scale Structure...

💰 Total cost: $0.00346
```

---

## Technical Notes

### Why SOM for seismic facies?
SOMs are well-established in exploration geophysics for unsupervised facies classification. They map high-dimensional seismic attribute spaces to a 2D grid where geologically similar facies cluster together — without requiring labeled training data.

### Why GPT-4o mini over open-source Vision models?
We tested four free Vision models accessible from Kaggle (Moondream2, Florence-2, nanollava, BLIP2). All failed due to rate limits, auth errors, or Kaggle network restrictions. GPT-4o mini at `detail:high` with image optimization (~$0.003/call) provided the best cost-quality balance.

### Image optimization
Sending raw matplotlib PNG (1972×745px) to GPT-4o mini consumed ~48K tokens ($0.007). Resizing to 1200px JPEG quality=82 reduced this to ~20K tokens ($0.003) with no loss in interpretation quality.

### SOM → LLM context bridge
The most impactful architectural decision was injecting the SOM's numerical fault score into the LLM prompt. This grounded the LLM's interpretation in quantitative evidence, dramatically reducing false positives on no-fault samples.

---

## Connectivity Notes (Kaggle-specific)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `api-inference.huggingface.co` | ❌ Blocked | Kaggle network restriction |
| `huggingface.co` (Spaces) | ✅ Accessible | Rate limits on public Spaces |
| `api.openai.com` | ✅ Accessible | Used for Layers 2 and 3 |
| `export.arxiv.org` | ✅ Accessible | Used for Layer 3 RAG |

---

## Installation

```bash
# Kaggle environment — only minisom needs installation
pip install minisom

# Standard libraries (pre-installed on Kaggle)
# numpy, pandas, matplotlib, sklearn, PIL, requests
```

Secrets required in Kaggle:
- `OPENAI_API_KEY` — for Vision LLM and RAG synthesis

---

## Future Work

- **CurveVel_A**: curved velocity layers require the 2D pixel-wise SOM (same as FlatFault_A). Pipeline structure is ready — `compute_litho_map` and the gradient prompt need adaptation.
- **Layer 3 v2**: replace live arXiv search with a FAISS local index over pre-downloaded FWI papers for faster, more controlled retrieval.
- **Quantitative evaluation**: compare SOM lithology maps against ground truth velocity bins using IoU or per-class accuracy.
- **Batch processing**: run the full pipeline over all 500 samples and aggregate geological statistics.

---

## Repository Structure

```
seismic-pipeline-som-llm/
│
├── README.md
├── seismic_pipeline_documented.ipynb   # Full documented notebook
└── assets/
    ├── fault_litho_0.png               # No-fault sample
    ├── fault_litho_100.png             # Central fault sample
    └── fault_litho_400.png             # Complex fault sample
```

---

## References

- Chen, L. et al. (2022). **OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Full Waveform Inversion**. *NeurIPS 2022 Datasets and Benchmarks Track*. [arXiv:2111.02926](https://arxiv.org/abs/2111.02926)
- Yang, F. & Ma, J. (2019). **Deep-learning inversion: A next-generation seismic velocity model building method**. *Geophysics*. [arXiv:1902.06267](https://arxiv.org/abs/1902.06267)
- He, L. et al. (2024). **MS_ATpV-FWI: Full Waveform Inversion based on Multi-scale Structural Similarity**. [arXiv:2504.01695](https://arxiv.org/abs/2504.01695)
- Mu, X. et al. (2025). **Full waveform inversion with CNN-based velocity representation**. [arXiv:2504.15826](https://arxiv.org/abs/2504.15826)

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built on Kaggle · OpenFWI dataset · GPT-4o mini · MiniSom · arXiv*

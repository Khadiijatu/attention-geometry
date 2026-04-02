# The Geometry of Collective Attention

**Topological and dynamical analysis of multi-modal engagement on digital platforms**

[![Status](https://img.shields.io/badge/status-data%20collection-amber?style=flat-square)](https://github.com/Khadiijatu/attention-geometry)
[![License](https://img.shields.io/badge/license-MIT-teal?style=flat-square)](LICENSE)
[![Site](https://img.shields.io/badge/site-live-green?style=flat-square)](https://khadiijatu.github.io/attention-geometry/)

> *Online engagement is not a number. It is a path through time, bending,
> accelerating, curling back on itself. This project reads the curvature of that path.*

**[→ Live research site](https://khadiijatu.github.io/attention-geometry/)**

---

## Overview

Standard engagement metrics, view counts, like rates, watch time, are scalars.
They collapse a rich temporal trajectory into a single number, discarding everything
interesting about *when* and *how* engagement changes.

This project proposes a geometric framework: treat the engagement signal as a
**path** $\mathbf{e}(t) = (r(t),\, \ell(t),\, c(t))$ in $\mathbb{R}^3$
(retention rate, like-accumulation rate, comment rate), and study its
**curvature**, **path signature**, and **persistent homology**.

The central formula:

$$\kappa(t) = \frac{\|\mathbf{e}'(t) \times \mathbf{e}''(t)\|}{\|\mathbf{e}'(t)\|^3}, \qquad W = \int_0^T \kappa(t)\,\|\mathbf{e}'(t)\|\,dt$$

$\kappa(t)$ measures sharpness of the audience response at time $t$;
$W$ is the total attention work, a coordinate-free, scale-invariant summary
of the trajectory's geometric complexity.

**Primary hypothesis:** $W$ predicts viral potential and long-term stickiness
better than any scalar engagement metric.

---

## Research questions

1. Does trajectory curvature ($W$) outperform scalar engagement metrics
   as a predictor of virality?
2. Which content features (auditory, visual, textual, structural) are
   associated with high-curvature events (emotional peaks)?
3. Does the persistence entropy of the retention curve predict comment
   engagement independently of content-level features?
4. Can the engagement signal be modelled as a **sheaf** over a modality
   poset, and does the cohomological coherence defect
   $\delta_X = \dim H^1(\mathcal{M}, \mathcal{E}_X)$
   predict engagement stability?

---

## Dataset

800 videos across 10 content categories, collected entirely from public sources.

| Category | n | Key contrast |
|---|---|---|
| Science education | 80 | Dense, monologue, slow |
| Comedy sketches | 80 | Fast cuts, emotional spikes |
| Meditation / wellness | 80 | Sustained low-arousal attention |
| Finance / investing | 80 | Authority signals, trust |
| Gaming | 80 | High motion, reactive |
| Beauty / fashion | 80 | Visual richness, tutorial |
| Cooking | 80 | Procedural, completion-driven |
| Personal vlog | 80 | Conversational, parasocial |
| Political commentary | 80 | Controversy, high comment rate |
| Mathematics / philosophy | 80 | Niche, extreme loyalty |

Feature groups: engagement trajectory (4 snapshots), audio (librosa),
visual (PIL + OpenCV on keyframes), text (spaCy + VADER), structural
(yt-dlp), modality flags (zero-shot CLIP).

---

## Mathematical framework

### Layer 1: Path signatures (rough path theory)

Each engagement trajectory $\mathbf{e}:[0,T]\to\mathbb{R}^3$ is encoded
by its **path signature** — a sequence of iterated integrals:

$$S(\mathbf{e})^{i_1,\ldots,i_k} = \int_{0<t_1<\cdots<t_k<T}
d\mathbf{e}^{i_1}_{t_1}\cdots d\mathbf{e}^{i_k}_{t_k}$$

Truncated at depth 4, this gives a fixed-dimensional feature vector
for arbitrary-length trajectories, invariant to reparametrisation.
Implementation: [`iisignature`](https://github.com/bottler/iisignature).

### Layer 2: Persistent homology (TDA)

The retention curve $r:[0,T]\to[0,1]$ is treated as a Morse function.
Its sublevel-set persistent homology gives a barcode encoding local
attention dips and recoveries. The **persistence entropy**:

$$H_{\mathrm{pers}} = -\sum_i \frac{\ell_i}{L}\log\frac{\ell_i}{L},
\quad L = \sum_i \ell_i$$

measures the topological complexity of the retention trajectory.
The full feature point cloud (800 videos) is analysed via Vietoris-Rips
persistent homology and the Mapper algorithm.
Implementation: [`ripser`](https://github.com/scikit-tda/ripser.py),
[`kepler-mapper`](https://github.com/scikit-tda/kepler-mapper).

### Layer 3: Sheaf coherence (category theory)

Content modalities $\{V, A, T, S\}$ form a poset $\mathcal{M}$.
Each video defines a functor $F_X:\mathcal{M}\to\mathbf{Meas}$;
aggregate engagement is the colimit. The failure of the natural
transformation to commute is the **coherence defect**
$\delta_X = \dim H^1(\mathcal{M}, \mathcal{E}_X)$,
computed from cross-modal synchronisation correlations.

This framework is inspired by Curry's sheaf-theoretic TDA
[(Curry et al. 2025)](https://arxiv.org/abs/2507.22010)
and connects to the author's manuscript on Whitney stratifications
of gradient flow closures [(in preparation)](https://khadiijatu.github.io/morse-geometry-ct/).

---

## Modelling strategy

| Model | Features | Target |
|---|---|---|
| M1 Baseline | Duration + category | Like-rate |
| M2 Feature model | All extracted features | Like-rate |
| M3 Signature model | Path signatures (depth 4) | Like-rate |
| M4 Topological model | M3 + persistence entropy + Betti numbers | Like-rate + comment rate |
| M5 Trajectory VAE | Content features → trajectory | Full trajectory $\mathbf{e}(t)$ |

Models are compared sequentially; each layer is evaluated
against the previous one. Explanation is prioritised over prediction.

---

## Repository structure

```
attention-modeling/
│
├── data/
│   ├── raw/            # Raw API JSON, SocialBlade exports
│   ├── processed/      # Cleaned CSVs, trajectory arrays (.npy)
│   └── README.md       # Data provenance
│
├── docs/
│   └── index.html      # GitHub Pages site (live)
│
├── latex/
│   └── writeup.tex     # Technical write-up (in progress)
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_curvature_analysis.ipynb
│   ├── 04_path_signatures.ipynb
│   ├── 05_persistence_homology.ipynb
│   └── 06_sheaf_coherence.ipynb
│
├── results/
│   ├── figures/        # Saved plots
│   └── tables/         # Model comparison CSVs
│
├── src/
│   ├── collect.py      # YouTube API pipeline
│   ├── features.py     # Feature extraction
│   ├── curvature.py    # κ(t) and W
│   ├── signatures.py   # Path signatures
│   ├── topology.py     # Persistent homology + Mapper
│   └── models.py       # ML models
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.10+, pip, git.

```bash
# 1. Clone the repository
git clone https://github.com/Khadiijatu/attention-geometry.git
cd attention-geometry

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**

```
numpy scipy pandas matplotlib
yt-dlp librosa Pillow opencv-python
spacy textstat vaderSentiment
iisignature ripser persim kmapper
scikit-learn torch
google-api-python-client
```

---

## Usage

### Data collection

```bash
# Set your YouTube API key
export YOUTUBE_API_KEY="your_key_here"

# Collect 80 videos per category (800 au total)
python src/collect.py --category_ids 27,23,22,25,20,26,1,24,25,28 --n_per_cat 80

# Output: data/raw/videos_raw.json
```

### Feature extraction

```bash
python src/features.py --input data/raw/videos_raw.json --output data/processed/

# Output:
#   data/processed/features_audio.csv
#   data/processed/features_visual.csv
#   data/processed/features_text.csv
#   data/processed/trajectories.npy
```

### Curvature analysis

```python
from src.curvature import attention_curvature, total_attention_work
import numpy as np

# Load a trajectory e(t): shape (T, 3) — retention, likes, comments
e = np.load('data/processed/trajectories.npy')[video_idx]

kappa = attention_curvature(e)   # shape (T,)
W     = total_attention_work(e)  # scalar
```

### Notebooks

Run the notebooks in order:

```bash
jupyter lab notebooks/
```

Start with `01_data_collection.ipynb` for a guided walkthrough
of the collection pipeline with inline explanations.

---

## Status

| Phase | Status |
|---|---|
| Site and framework design | ✅ Complete |
| Data collection pipeline | 🔄 In progress |
| Audio / visual extraction | 🔄 In progress |
| Curvature analysis | ⏳ Planned |
| Path signatures | ⏳ Planned |
| Persistent homology | ⏳ Planned |
| Sheaf coherence | ⏳ Planned |
| Models M1-M4 | ⏳ Planned |
| Trajectory VAE | ⏳ Planned |
| Write-up | ⏳ Planned |

---

## Mathematical connections

This project sits at the intersection of three bodies of literature:

- **Rough path theory / path signatures:**
  Lyons (1994); Chevyrev & Kormilitzin (2016)
- **Topological data analysis:**
  Edelsbrunner & Harer (2010); Bauer (2021, ripser);
  Singh, Mémoli & Carlsson (2007, Mapper)
- **Sheaf theory for data:**
  Curry (2014, PhD thesis); Curry et al. (2025, arXiv:2507.22010)

The connection to the author's mathematical research:
the engagement trajectory $\mathbf{e}(t)$ viewed via delay embedding
forms a stratified space whose volume growth exponents
are conjectured to be controlled by the Morse indices of $r(t)$,
directly applying the framework of
[Cissé (2026)](https://khadiijatu.github.io/morse-geometry-ct/).

---

## Author

**K. Cissé**

[GitHub](https://github.com/Khadiijatu) ·

---

## License

MIT - see [LICENSE](LICENSE).
Data collected via YouTube Data API v3 is subject to
[YouTube's Terms of Service](https://www.youtube.com/t/terms).
No private or non-public data is used.

# Modeling Human Attention Across Modalities, Content Features, and Algorithmic Exposure

**Status:** Independent research project (ongoing)

---

## Overview
Human attention is a central concept in cognition, learning, and digital interaction,
yet it is often approximated using coarse engagement metrics that ignore content
structure and algorithmic context.

This project investigates **attention as an observable, structured signal**
emerging from the interaction between content modality, content features,
and algorithmic exposure.

The goal is not to claim a definitive measure of attention, but to build an
**interpretable, quantitative framework** that makes explicit what is measured,
how it is modeled, and where its limits lie.

---

## Research Questions
- How does observed attention vary across content modalities (text, audio, video)
  when controlling for topic and length?
- Which content features (linguistic, auditory, visual, narrative) are most
  strongly associated with sustained attention?
- How does algorithmic exposure (recommended vs non-recommended content)
  interact with content features?
- Are there multiple distinct content–feature configurations that lead
  to high attention?

---

## Dataset Design
The dataset is constructed from publicly available digital content sampled
across multiple platforms and domains.

**Modalities**
- Text (blogs, articles)
- Audio (podcasts, spoken content)
- Video (short-form and long-form)

**Topics**
Lifestyle, mental health, science, technology, culture, art, environment,
education, and related domains.

**Exposure Annotation**
Content is labeled as *algorithmically recommended* or *non-recommended*
based on observable platform signals. The study is observational and does not
manipulate recommendation systems.

---

## Attention Score
Attention is approximated using a **composite behavioral attention score**,
combining normalized engagement proxies such as:
- dwell time
- completion rate
- replay or revisit signals
- interaction-adjusted engagement

The score is **modality-aware** and normalized for content length.
It is interpreted as a proxy for *behavioral attention*, not cognitive or
neural attention.

---

## Feature Extraction Pipeline

### Linguistic Features
- text length and structure
- lexical diversity and complexity
- sentence and paragraph statistics

### Auditory Features
- speech rate and pauses
- pitch variability
- background sound or music indicators

### Visual Features
- motion intensity
- color dynamics
- visual complexity and cuts

### Structural / Narrative Features
- pacing and segmentation
- conversational markers
- storytelling indicators

---

## Modeling Strategy
The modeling approach prioritizes **explanation over prediction**.

- Mixed-effects models to account for topic and creator variability
- Feature-group comparisons to assess explanatory power
- Selective nonlinear modeling for theoretically motivated features
- Unsupervised clustering to identify distinct attention profiles

Prediction models are used only as robustness checks, not as objectives.

---

## Key Insights (In Progress)
- Attention distributions differ across modalities even within the same topic
- Structural and narrative features often explain more variance than low-level
  perceptual features
- High attention is achieved through multiple distinct feature configurations
  rather than a single optimal style

---

## Limitations
- Observational data only; no causal claims
- Platform-dependent engagement proxies
- Behavioral attention is an approximation
- Limited scale relative to industrial datasets

---

## Future Extensions
- Alignment of behavioral attention scores with neural or physiological measures
  (e.g., EEG, eye-tracking)
- Modeling attention and recommendation systems as coupled feedback processes
- Exploration of higher-level mathematical abstractions for attention dynamics

---

## Author
**Khadidiatou Cissé**  
MSc Mathematics | Complex Systems | Data & ML Research

Links: [GitHub](https://github.com/Khadiijatu) • [LinkedIn](https://www.linkedin.com/in/khadijatea/)

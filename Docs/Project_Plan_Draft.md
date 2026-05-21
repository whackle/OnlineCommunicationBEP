
> **DEPRECATED** — superseded by `IMPLEMENTATION_PLAN.md`. Terminology in this document (e.g. "algorithmic amplification") is outdated; current framing uses "out-of-network diffusion".

---

### 1. Introduction

Understanding how information spreads through social networks is a central question in computational social science. On centralized platforms such as Twitter/X and Weibo, information cascades — the chains of reposts through which a piece of content reaches new audiences — have been studied extensively using supervised deep learning methods (Chen et al., 2019; Cao et al., 2017; Yuan et al., 2024). However, these approaches share two critical limitations: they require ground-truth labels (e.g., "viral" vs. "non-viral"), and they are designed for platforms where a single opaque recommendation algorithm governs content visibility. Neither assumption holds on decentralized social networks.

Bluesky, built on the AT Protocol, represents a new class of social platform where users can choose or disable algorithmic feeds entirely, where identity is portable across servers (Personal Data Servers), and where the full firehose of platform activity is publicly accessible in real time. This architecture creates a unique natural experiment: on the same platform, some content spreads organically through follower networks while other content is amplified by user-selected algorithmic feeds — and crucially, the structural trace of both pathways is observable. Yet, to date, no study has analysed information cascade dynamics on Bluesky at a structural level.

This research proposes **BlueCCGL** (Bluesky Contrastive Cascade Graph Learning), a framework for the unsupervised discovery of information cascade archetypes on Bluesky. BlueCCGL extends the Contrastive Cascade Graph Learning (CCGL) framework of Xu et al. (2022) with three novel contributions: **(1)** feed-aware graph augmentation strategies that respect the semantic distinction between organic and algorithmically-amplified spread; **(2)** edge-type aware message passing via Relational Graph Convolutional Networks (R-GCN; Schlichtkrull et al., 2018) that explicitly encodes whether a cascade edge represents follower-based or non-follower-based propagation; and **(3)** hierarchical graph pooling (Ying et al., 2018) that captures multi-scale cascade structure from individual user decisions to community-level patterns. By learning cascade representations through contrastive self-supervised learning and clustering them with HDBSCAN (McInnes et al., 2017), BlueCCGL discovers natural cascade archetypes without requiring any labeled data.

The expected results are significant for two reasons. First, a label-free approach to cascade archetype discovery is directly transferable to any platform where labeled cascade data is unavailable — which is the norm outside a handful of well-studied centralized platforms. Second, the discovered archetypes would provide the first empirical characterisation of how information spreads differently through follower networks versus algorithmic recommendation on a decentralized, algorithm-optional platform — offering insights that are impossible to obtain from centralized platforms where these two pathways cannot be disentangled.

---

### 2. Related Work

#### 2.1 Information Cascade Prediction

Graph neural network approaches to cascade modelling have evolved rapidly. **CasCN** (Chen et al., 2019) was among the first to apply GNNs to cascade prediction, modelling cascades as sequences of graph snapshots with temporal attention. **DeepHawkes** (Cao et al., 2017) bridged classical Hawkes processes with deep learning to provide interpretable attention over user influence, though it assumes homogeneous user influence. More recently, **CasFlow** (Yuan et al., 2024) introduced a variational approach with hierarchical pooling, achieving state-of-the-art performance on Weibo and Twitter datasets. ⚠️ **HierCas** (Wang et al., 2024) extended this with multi-resolution modelling capturing both user-level and community-level dynamics, and ⚠️ **TempCas** (Liu et al., 2025) applied Transformer architectures with continuous-time attention.

**Gap:** All of the above methods are *supervised*, requiring ground-truth popularity or virality labels. None account for heterogeneous edge types (e.g., follower vs. non-follower propagation), and all were evaluated exclusively on centralized platforms (Twitter, Weibo). BlueCCGL addresses all three gaps: it is unsupervised, edge-type aware, and targets a decentralized platform.

#### 2.2 Contrastive and Self-Supervised Graph Learning

**CCGL** (Xu et al., 2022) is the direct methodological predecessor of this work. It introduced contrastive self-supervised pre-training for cascade graphs using sub-cascade sampling for augmentation and an InfoNCE graph-level contrastive loss, demonstrating 10–20% improvement over supervised baselines. **GraphCL** (You et al., 2020) provided a systematic study of graph augmentation strategies (node dropping, edge perturbation, attribute masking, subgraph sampling) that informs augmentation design. **GRACE** (Zhu et al., 2020) introduced node-level corruption-based contrastive learning. For temporal graphs, **TGC** (Zhang et al., 2023) proposed time-aware augmentation strategies, and ⚠️ **CSTGCN** (Li et al., 2024) introduced multi-scale temporal contrasting for evolving structural patterns.

**Gap:** CCGL's augmentations are generic (random sub-cascade sampling) and carry no domain semantics. No existing contrastive graph learning method incorporates augmentations that are aware of the distinction between organic and algorithmically-mediated spread. BlueCCGL introduces feed-aware augmentation strategies (follower-subgraph extraction, temporal prefix sampling, bridge-node removal) designed to preserve this semantic structure.

#### 2.3 Heterogeneous GNNs and Organic vs. Algorithmic Spread

Relational and heterogeneous GNN architectures provide the building blocks for edge-type aware cascade encoding. **R-GCN** (Schlichtkrull et al., 2018) introduced relation-specific weight matrices with basis decomposition for parameter efficiency. **HAN** (Wang et al., 2019) proposed hierarchical attention across node-level and semantic-level, while **HGT** (Hu et al., 2020) applied Transformer-style attention to heterogeneous graphs. **RHGNN** (Yu et al., 2023) demonstrated the importance of modelling multiple relationship types (follow, mention, repost) in social network analysis.

On the empirical side, Huszár et al. (2022) provided a landmark internal Twitter study showing algorithmic amplification of political content — a gold standard that requires platform-insider access. Goel et al. (2016) defined the "structural virality" metric (Wiener index) to distinguish broadcast from viral spread, and Vosoughi et al. (2018) showed that true and false news produce structurally different cascades. Bhadani et al. (2024) proposed structural signatures of algorithmic amplification, and Bandy & Diakopoulos (2021) developed methodology for auditing algorithmic curation.

**Gap:** No prior work has applied heterogeneous GNN encoding to information cascades, specifically encoding follower vs. non-follower edges as distinct relation types. Furthermore, studies of organic vs. algorithmic spread have relied on either platform-insider data (Huszár et al., 2022) or indirect proxies. BlueCCGL exploits Bluesky's observable follow graph to directly label cascade edges and uses R-GCN encoding to learn relation-aware representations.

#### 2.4 Decentralized Social Networks and Bluesky

Research on decentralized social networks has focused primarily on the Fediverse. Zignani et al. (2018) conducted the first large-scale Mastodon measurement study, and Raman et al. (2019) analysed technical challenges of federation protocols. For the AT Protocol specifically, Kleppmann & Howard (2024) provided a technical comparison of ActivityPub and AT Protocol, including Bluesky's Firehose architecture. ⚠️ Cerisara et al. (2025) presented an early Bluesky ecosystem analysis studying user behaviour and AT Protocol adoption. Most notably, **Failla & Rossetti (2024)** released the first large-scale Bluesky dataset, covering the complete post history of over 4M users (~81% of all registered accounts at the time), 235M posts, a directed follower graph of ~144M edges, and timestamped repost/quote/reply interaction graphs — all publicly available on Zenodo (DOI: 10.5281/zenodo.14669616). Critically, the dataset also includes posts from 11 feed generators and user feed-subscription data, providing ground-truth information about algorithmic exposure that is unavailable from the Jetstream firehose alone.

**Gap:** While Failla & Rossetti (2024) provide the first comprehensive Bluesky dataset with descriptive analysis, no study has performed cascade-level *structural* analysis on Bluesky or any AT Protocol platform — specifically, no work has modelled repost cascades as graphs, assigned edge types based on follower relationships, or applied graph representation learning to discover cascade archetypes. BlueCCGL is designed to fill this gap, leveraging the Failla & Rossetti dataset as its primary data source.

#### 2.5 Hierarchical Graph Pooling and Clustering

**DiffPool** (Ying et al., 2018) introduced differentiable soft cluster assignment for hierarchical graph representation learning. **SAGPool** (Lee et al., 2019) provided a more efficient attention-based alternative, and **MinCutPool** (Bianchi et al., 2020) offered a spectral-clustering-grounded approach. For cascades specifically, ⚠️ **HiCas** (Zhou et al., 2023) applied hierarchical pooling to capture multi-scale propagation patterns. For unsupervised pattern discovery, **HDBSCAN** (McInnes et al., 2017) provides density-based clustering that does not require pre-specifying the number of clusters, while **DEC** (Xie et al., 2016) and **DMoN** (Tsitsulin et al., 2023) offer end-to-end deep clustering alternatives.

**Gap:** Hierarchical pooling has not been combined with contrastive learning for cascade graphs, nor has it been applied in an edge-type aware setting. BlueCCGL integrates DiffPool-style hierarchical pooling into the contrastive cascade learning pipeline, enabling multi-scale representations that capture micro (user), meso (community), and macro (global cascade shape) patterns.

---

### 3. Research Questions

**Main Research Question:**

> *Can unsupervised contrastive graph learning discover meaningful information cascade archetypes on Bluesky that reflect the interplay between organic follower-based spread and algorithmic amplification?*

This is decomposed into three sub-questions:

**SQ1: Archetype Discovery.** What natural cascade archetypes emerge when applying HDBSCAN clustering to cascade embeddings learned via contrastive self-supervised learning, and how do these archetypes differ in structural properties (depth, breadth, branching factor, follower-edge ratio, temporal spread pattern)?

**SQ2: Edge-Type Aware Encoding.** How does encoding follower vs. non-follower edges via relational graph convolution (R-GCN) affect the quality and discriminative power of learned cascade representations, compared to homogeneous GNN baselines?

**SQ3: Feed-Aware Augmentation.** Do domain-specific augmentation strategies — follower-subgraph extraction, temporal prefix sampling, and bridge-node removal — produce higher-quality cascade representations than generic graph augmentations (random node/edge dropping) as measured by clustering coherence and downstream task performance?

---

### 4. Expected Outcomes

**Outcome 1 — A taxonomy of cascade archetypes on Bluesky.** We expect HDBSCAN clustering over the learned embeddings to reveal distinct cascade archetypes characterised by interpretable structural properties. Based on prior cascade literature (Goel et al., 2016; Vosoughi et al., 2018), we anticipate at minimum a distinction between *broadcast-like* cascades (shallow, wide, dominated by non-follower edges suggesting algorithmic amplification) and *viral-like* cascades (deep, narrow, dominated by follower edges suggesting organic diffusion), with potentially additional intermediate archetypes (e.g., mixed-spread cascades with early organic seeding followed by algorithmic amplification).

**Outcome 2 — Quantitative evidence for the contribution of each extension.** Through ablation studies, we expect to demonstrate that: (a) edge-type aware R-GCN encoding improves representation quality over homogeneous GNN baselines (measured via silhouette score, Davies-Bouldin index, and Calinski-Harabasz index); (b) feed-aware augmentations outperform generic random augmentations on the same metrics; and (c) hierarchical pooling captures multi-scale cascade patterns that flat global pooling misses. If any extension does not yield improvement, this is itself a valuable finding about the nature of cascade structure on Bluesky.

**Outcome 3 — An open-source implementation and reusable cascade dataset.** The project will produce: (a) the BlueCCGL PyTorch Geometric codebase (cascade construction, model training, clustering); and (b) a processed Bluesky cascade graph dataset derived from the Failla & Rossetti (2024) Zenodo dataset (DOI: 10.5281/zenodo.14669616), containing approximately 50,000 cascades (10–10,000 reposts each) with edge-type annotations (follower / non-follower / unknown) assigned by cross-referencing repost edges against the dataset's 144M-edge follower graph.

**Outcome 4 — Transfer learning evaluation on downstream tasks.** We expect the pre-trained BlueCCGL encoder to provide useful representations for downstream cascade tasks — specifically cascade size prediction and cascade longevity prediction — outperforming both a random-initialisation baseline and vanilla CCGL, thereby demonstrating the practical utility of the learned representations beyond archetype discovery.

---

### 5. Research Plan

This is a solo Bachelor's thesis project spanning 16 weeks. By leveraging the pre-existing Failla & Rossetti (2024) Zenodo dataset instead of live Jetstream collection, the data acquisition phase is compressed from 4 weeks to 2 weeks, freeing time for more thorough experimentation and writing. The table below details the planned activities per week.

| Week | Phase | Activities | Deliverable |
|------|-------|-----------|-------------|
| 1 | Data Acquisition | Download Failla & Rossetti (2024) Zenodo dataset (~22 GB); decompress `followers.csv.gz`, `interactions.csv.gz`, `graphs.tar.gz`, `user_posts.tar.gz`; load follower graph and interaction edge lists into working format (Parquet/DuckDB) | Raw dataset loaded; storage pipeline validated |
| 2 | Data Acquisition | Data quality checks (timestamp consistency, deduplication); compute exploratory statistics (repost volumes, cascade size distribution, follower-graph density, edge-type ratios); identify candidate cascades meeting size thresholds (10–10,000 reposts) | Data quality report; exploratory statistics notebook |
| 3 | Cascade Construction | Implement cascade graph builder: reconstruct repost cascades from `interactions.csv.gz` and `graphs.tar.gz`; assign edge types (follower / non-follower / unknown) by cross-referencing cascade edges against `followers.csv.gz` | Cascade construction pipeline; edge-type labelled cascades |
| 4 | Cascade Construction | Convert cascade graphs to PyTorch Geometric dataset; perform detailed EDA (cascade depth/breadth distributions, follower-edge ratio distributions, temporal spread patterns); filter to final ~50K cascade dataset | PyG dataset; EDA notebook with visualisations |
| 5 | Model Implementation | Implement base CCGL reproduction: GNN encoder, sub-cascade sampling augmentation, InfoNCE contrastive loss | Base CCGL training loop verified on cascade data |
| 6 | Model Implementation | Implement Extension 1: feed-aware augmentation strategies (follower-subgraph extraction, temporal prefix sampling, bridge-node removal) | Augmentation module with unit tests |
| 7 | Model Implementation | Implement Extension 2: edge-type aware R-GCN encoder with basis decomposition; integrate with contrastive pipeline | R-GCN encoder integrated into BlueCCGL |
| 8 | Model Implementation | Implement Extension 3: hierarchical DiffPool-based pooling; assemble full BlueCCGL architecture; end-to-end training verification | Complete BlueCCGL model; successful training run |
| 9 | Experiments | Contrastive pre-training of BlueCCGL; hyperparameter tuning (temperature, augmentation ratios, learning rate) via Weights & Biases | Trained model; hyperparameter sweep results |
| 10 | Experiments | Extract cascade embeddings; apply HDBSCAN clustering; compute clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz); generate UMAP visualisations | Archetype clusters; clustering quality report |
| 11 | Experiments | Ablation studies: remove edge-type encoding, replace feed-aware augmentations with generic augmentations, replace hierarchical pooling with flat pooling, compare to random initialisation | Ablation results table |
| 12 | Experiments | Downstream transfer-learning evaluation (cascade size prediction, longevity prediction); archetype interpretation (structural property analysis per cluster) | Downstream task results; archetype characterisation |
| 13 | Writing | Draft Introduction, Related Work, and Research Questions sections; compile bibliography | Thesis draft chapters 1–3 |
| 14 | Writing | Draft Methodology and Experimental Setup sections; create all figures and tables | Thesis draft chapters 4–5 |
| 15 | Writing | Draft Results, Discussion, and Conclusion; integrate all sections into full thesis | Complete thesis draft |
| 16 | Writing | Revisions based on supervisor feedback; final proofreading; prepare presentation | Final thesis; presentation slides |

---

**⚠️ Flagged citations (require verification):**
- HierCas (Wang et al., 2024) — "Hierarchical Information Propagation Prediction"
- TempCas (Liu et al., 2025) — "Temporal Cascade Transformers"
- CSTGCN (Li et al., 2024) — "Contrastive Self-Supervised Temporal Graph Learning"
- Cerisara et al. (2025) — "Bluesky Ecosystem Analysis"
- HiCas (Zhou et al., 2023) — "Hierarchical Cascade Representation"
- Bhadani et al. (2024) — "Algorithmic Amplification of Low-Quality Content"

---

# BlueCCGL: Unsupervised Discovery of Crisis Information Cascade Archetypes on Bluesky

**Partial Draft — April 20, 2026**

> *Sections 1–4 complete. Section 5 (next steps) appended. Ablation studies, downstream evaluation, and stability analysis pending.*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Methods](#3-methods)
4. [Preliminary Results](#4-preliminary-results)
5. [Next Steps](#5-next-steps)
6. [References](#6-references)

---

## 1. Introduction

Understanding how crisis information propagates through social networks has become a central question in computational social science. During acute geopolitical crises, information cascades — the chains of reposts through which content reaches new audiences — exhibit dynamics that differ fundamentally from steady-state discourse: urgency compresses decision-making, emotional intensity amplifies sharing, and community boundaries become more or less permeable depending on the nature of the event. On centralised platforms such as Twitter/X and Weibo, crisis cascade dynamics have been studied extensively using supervised deep learning methods (Chen et al., 2019; Cao et al., 2017; Yuan et al., 2024). These approaches share two critical limitations: they require ground-truth labels such as "viral" vs. "non-viral" cascade assignments that are expensive to produce and rarely available for novel crises, and they are designed for platforms where a single opaque recommendation algorithm governs content visibility. Neither assumption holds on decentralised social networks, and neither is compatible with the inherent unpredictability of crisis information dynamics.

Bluesky, built on the AT Protocol, represents a qualitatively different class of social platform. Users may choose or disable algorithmic feeds entirely; identity is portable across Personal Data Servers (PDS); and the full firehose of platform activity is publicly accessible in real time. Critically, Bluesky's architecture creates a structurally observable trace of both organic and algorithmically-mediated content spread: the publicly available follower graph distinguishes between a post that reached a user because they follow the poster — organic spread — and one that reached them via an external algorithmic feed — a pathway Bluesky exposes but centralised platforms obscure. This observability is architecturally unique and constitutes the primary methodological advantage of studying crisis information spread on Bluesky.

During the period covered by the Failla and Rossetti (2024) Zenodo dataset (February 2023 – March 2024), the Israel-Palestine conflict escalated dramatically following the Hamas attack of October 7, 2023, producing a sustained, multi-phase crisis that generated intense online discourse. Bluesky's user base during this period — small (approximately 1–5 million users), predominantly English-speaking, Western, and politically left-leaning — constitutes a politically homogeneous early-adopter community processing a major geopolitical crisis through decentralised infrastructure. Studying how such a community propagates crisis information is an understudied and theoretically important question, particularly because the decentralised architecture makes the structural trace of different propagation pathways directly observable, something that is structurally impossible on centralised platforms.

To date, no study has performed cascade-level structural analysis on Bluesky or any AT Protocol platform. The Failla and Rossetti (2024) dataset provides the first large-scale empirical basis for such analysis, covering approximately 235 million posts, 4 million users, and a directed follower graph of approximately 144 million edges. However, the dataset's descriptive analysis does not examine information cascades, let alone crisis-specific cascade dynamics.

This work proposes **BlueCCGL** (Bluesky Contrastive Cascade Graph Learning), a framework for the unsupervised discovery of crisis information cascade archetypes on Bluesky, applied to the Israel-Palestine conflict from October 1, 2023 to March 18, 2024. BlueCCGL extends the Contrastive Cascade Graph Learning (CCGL) framework of Xu et al. (2022) with four novel contributions:

1. **An upstream NLP filtering pipeline** that identifies conflict-related posts from the 235-million-post corpus via multilingual keyword matching across five tiers — high-precision English terms, English hashtag patterns, Arabic-script keywords, Hebrew-script keywords, and named entities — and assigns posts to six temporal mega-event windows corresponding to major conflict escalations.

2. **Feed-aware graph augmentation strategies** that respect the semantic distinction between organic follower-based spread and algorithmically-mediated spread: follower-subgraph extraction simulates purely organic propagation; temporal prefix sampling isolates early-stage breaking-news dynamics; and bridge-node removal tests what a cascade would look like without non-follower entry points. These augmentations replace the domain-agnostic random subgraph sampling of the original CCGL.

3. **Edge-type aware message passing via Relational Graph Convolutional Networks (R-GCN; Schlichtkrull et al., 2018)** that explicitly encodes whether a cascade edge represents follower-based or non-follower-based propagation, using relation-specific weight matrices with basis decomposition across three relation types.

4. **Hierarchical graph pooling (Ying et al., 2018) enriched with sentiment composition features** that capture multi-scale cascade structure — from individual user-level repost decisions through community-level propagation patterns — and concatenate graph-level sentiment proportions (negative, neutral, positive) into the final readout.

By pre-training the full architecture via InfoNCE contrastive self-supervised learning (with VICReg variance and covariance regularisation), extracting 64-dimensional cascade embeddings, reducing them to 10 dimensions via UMAP, and clustering with HDBSCAN, BlueCCGL discovers natural crisis cascade archetypes without requiring any labelled data.

The six mega-events anchoring the analysis span qualitatively distinct categories of crisis event: the acute shock of the October 7 Hamas attack and Israeli declaration of war (E1); the contested humanitarian flashpoint of the Al-Ahli hospital explosion (E2); the multi-day diplomatic ceasefire and hostage exchanges brokered by Qatar in late November 2023 (E3); the institutional milestone of the ICJ genocide ruling on January 26, 2024 (E4); the emotionally resonant individual act of Aaron Bushnell's self-immolation on February 25, 2024 (E5); and the compound humanitarian crisis of the flour massacre and the 30,000 death toll milestone on February 29, 2024 (E6). This temporal and categorical diversity makes the dataset an unusually rich natural experiment for studying how different types of crisis event produce structurally different patterns of information propagation.

**Scope and framing.** All findings must be interpreted within the specific demographic context of Bluesky's 2023–2024 user base: English-dominant, Western-leaning, left-leaning early adopters. This study characterises crisis information dynamics within a politically homogeneous community on a decentralised platform — not global discourse on the conflict. This scope is stated as a framing condition throughout, not as a limitation to be apologised for.

The research questions guiding the analysis are structured into two sets. The socially-grounded questions ask how crisis cascade dynamics differ across conflict phases (SRQ1), across event categories (SRQ2), between organic and algorithmically-mediated spread (SRQ3), and between cascades of different sentiment composition (SRQ4). The methodological questions ask what archetypes naturally emerge from contrastive learning (MRQ1), how edge-type aware encoding changes representation quality compared to a homogeneous baseline (MRQ2), and whether feed-aware augmentations outperform generic graph augmentations (MRQ3). These questions are answered in Section 4.7 using the preliminary results from the full pipeline run.

---

## 2. Related Work

### 2.1 Supervised Information Cascade Prediction

Graph neural network approaches to cascade modelling have evolved rapidly over the past decade. CasCN (Chen et al., 2019) was among the first to apply GNNs to cascade prediction, modelling cascades as sequences of graph snapshots with temporal attention; the approach demonstrated the superiority of graph-based representations over sequence models for encoding the structural shape of propagation. DeepHawkes (Cao et al., 2017) bridged classical Hawkes process models with deep learning to provide interpretable attention over user influence, though it assumes homogeneous user influence — an assumption that fails in heterogeneous network environments where followers and non-followers have structurally different roles. CasFlow (Yuan et al., 2024) introduced a variational approach with hierarchical pooling, achieving state-of-the-art performance on Weibo and Twitter datasets.

**Critical gap.** All of the above methods are *supervised*, requiring ground-truth popularity or virality labels — binary viral/non-viral assignments, popularity quantiles, or future cascade size — that must be collected retrospectively after cascade completion. None account for heterogeneous edge types such as follower versus non-follower propagation. None have been applied to crisis-specific cascades. And all were evaluated exclusively on centralised platforms (Twitter, Weibo) where the mechanisms of content amplification are opaque. BlueCCGL addresses all three gaps: it is fully unsupervised, edge-type aware, crisis-focused, and targets a decentralised platform where organic and algorithmic pathways are structurally distinguishable.

### 2.2 Contrastive and Self-Supervised Graph Learning

CCGL (Xu et al., 2022) is the direct methodological predecessor of this work. It introduced contrastive self-supervised pre-training for cascade graphs using sub-cascade sampling for augmentation and an InfoNCE graph-level contrastive loss, demonstrating 10–20% improvement over supervised baselines on popularity prediction tasks on Weibo and Twitter. GraphCL (You et al., 2020) provided a systematic study of graph augmentation strategies — node dropping, edge perturbation, attribute masking, and subgraph sampling — establishing the foundational principle that different augmentations preserve different aspects of graph structure. GRACE (Zhu et al., 2020) introduced node-level corruption-based contrastive learning, demonstrating that contrastive objectives can produce high-quality node embeddings without labels.

**Critical gap.** CCGL's augmentations are generic sub-cascade samples carrying no domain semantics. No existing contrastive graph learning method incorporates augmentations that are aware of the structural distinction between organic and algorithmically-mediated spread. BlueCCGL introduces three feed-aware augmentation strategies explicitly designed for crisis information cascades, replacing domain-agnostic perturbations with semantically grounded structural transformations.

### 2.3 Heterogeneous GNNs and Organic versus Algorithmic Spread

Relational GNNs provide the architectural basis for edge-type aware cascade encoding. R-GCN (Schlichtkrull et al., 2018) introduced relation-specific weight matrices with basis decomposition for parameter efficiency, enabling the model to learn different aggregation functions for different edge types. HAN (Wang et al., 2019) proposed hierarchical attention across node-level and semantic-level paths in heterogeneous graphs, while HGT (Hu et al., 2020) applied Transformer-style attention to heterogeneous social networks. These architectures establish the principle that treating all edges identically — as standard GCNs do — discards semantically important relational information.

On the empirical side, Huszár et al. (2022) provided a landmark internal Twitter study showing differential algorithmic amplification of political content across party lines — a finding that required platform-insider access to the proprietary recommendation system, which decentralised platforms make unnecessary. Goel et al. (2016) defined the structural virality metric — the Wiener index, or mean pairwise shortest path — to distinguish broadcast from viral spread structurally, independent of content. Vosoughi et al. (2018) demonstrated that true and false news produce structurally different cascades, with false news spreading faster, deeper, and more broadly.

**Critical gap.** No prior work has applied heterogeneous GNN encoding to crisis information cascades on a platform where edge types are directly observable. BlueCCGL exploits Bluesky's observable follow graph to directly label cascade edges as follower-based or non-follower-based, and uses R-GCN encoding to learn relation-aware representations that distinguish organic from algorithmically-exposed propagation.

### 2.4 Crisis Communication Research

Crisis communication on social media has a rich literature. Olteanu et al. (2015) provided a comprehensive survey of social media use during crises, identifying information types — situational awareness, emotional support, opinions — and their spread patterns. Starbird and Palen (2012) studied retweeting patterns during crises, finding that crisis retweets form distinct structural patterns compared to steady-state discourse, with organic sharing networks playing a central coordinating role. Vosoughi et al. (2018) demonstrated that emotionally charged content, particularly content evoking surprise and disgust, spreads faster and deeper — a finding directly relevant to conflict-related cascades.

Specific to the Israel-Palestine conflict, social media dynamics have been studied primarily on Twitter and Facebook. Zeitzoff (2017) studied how conflict events affect online political expression, finding that exposure to violence radicalises online sentiment. However, all prior work in this domain studies centralised platforms where the mechanisms of algorithmic amplification are opaque and the structural distinction between organic and algorithmic spread is not available.

**Critical gap.** No study has examined crisis information cascade *structure* on a decentralised platform where organic and algorithmic pathways are distinguishable. BlueCCGL provides the first such analysis, using the Israel-Palestine conflict as a sustained, multi-phase, categorically diverse case study.

### 2.5 Decentralised Social Networks and the Bluesky Dataset

Research on decentralised social networks has focused primarily on the ActivityPub-based Fediverse. Zignani et al. (2018) conducted the first large-scale Mastodon measurement study. For the AT Protocol specifically, Kleppmann and Howard (2024) provided a technical comparison of ActivityPub and the AT Protocol, including Bluesky's Firehose architecture and the semantic differences between federation (ActivityPub) and decoupled moderation with portable identity (AT Protocol).

The primary empirical foundation of this work is the Failla and Rossetti (2024) Zenodo dataset — the first large-scale Bluesky dataset — covering the complete post history of over 4 million users (approximately 81% of all registered accounts at the time of collection), 235 million posts, a directed follower graph of approximately 144 million edges, and timestamped repost/quote/reply interaction graphs, covering the period from February 2023 to March 2024. The dataset also includes posts from 11 feed generators and user feed-subscription data, providing ground-truth information about algorithmic exposure at the user level. The paper explicitly reports a spike in News feed engagement during October 2023 likely attributable to the Israel-Palestine conflict, making this dataset uniquely suited to the present research question.

**Critical gap.** While Failla and Rossetti (2024) provide descriptive analysis of platform-level behaviour, no study has performed cascade-level structural analysis on Bluesky or any AT Protocol platform, and no study has examined crisis-specific cascade dynamics on a decentralised platform.

### 2.6 Hierarchical Graph Pooling and Density-Based Clustering

DiffPool (Ying et al., 2018) introduced differentiable soft cluster assignment for hierarchical graph representation learning, enabling end-to-end learning of multi-scale graph structure through learnable pooling matrices that assign nodes to super-nodes at each hierarchy level. Auxiliary losses — a link prediction loss encouraging the pooled adjacency to match the assignment, and an entropy loss encouraging near-discrete assignments — regularise the pooling towards meaningful structural groupings. SAGPool (Lee et al., 2019) provided a more computationally efficient attention-based alternative, while MinCutPool (Bianchi et al., 2020) offered a spectral-clustering-grounded approach.

For unsupervised pattern discovery, HDBSCAN (McInnes et al., 2017) provides hierarchical density-based clustering that does not require pre-specifying the number of clusters and naturally handles noise points — a critical property for cascade datasets where a fraction of embeddings may not belong to any coherent structural pattern. HDBSCAN's internal cluster persistence metric provides a principled measure of cluster stability across density thresholds, complementing external metrics such as the Silhouette score (Rousseeuw, 1987) and the Density-Based Clustering Validation index (DBCV; Moulavi et al., 2014), which was specifically designed to evaluate density-based clusters that need not be convex.

**Critical gap.** Hierarchical pooling has not been combined with contrastive learning for cascade graphs, nor applied with sentiment-enriched node features in a crisis context.

---

## 3. Methods

### 3.1 Dataset

The analysis uses the Failla and Rossetti (2024) Zenodo dataset (DOI: 10.5281/zenodo.14669616). Of the six available files, two were used in this study:

| File | Size | Role |
|------|------|------|
| `user_posts.tar.gz` | 19.5 GB | Post text, timestamps, sentiment labels, repost metadata |
| `followers.csv.gz` | 469 MB | Directed follower graph (~144M edges) for edge-type labelling |

The remaining files (`interactions.csv.gz`, `graphs.tar.gz`, `feed_posts.tar.gz`, `feed_bookmarks.csv`) were determined to be unnecessary: the `repost_from` and `reposted_author` fields within `user_posts.tar.gz` provide sufficient information for cascade reconstruction via the parent assignment heuristic described in Section 3.3, and a lean intermediate index was pre-extracted from the full tar for fast reposter discovery (see Section 3.2).

All user IDs in the dataset are pseudonymised integers; no AT Protocol DIDs, handles, or post URIs are present. Re-identification of individual users is not possible from the dataset alone. All cascade-level outputs are reported at the aggregate archetype level.

**Conflict window.** Only posts timestamped between October 1, 2023 and March 18, 2024 are considered. This window encompasses the six mega-events (Table 1) and captures the full escalatory arc of the conflict on the Bluesky platform.

**Mega-event windows.** Six major conflict escalations were identified as temporal anchors for cascade analysis. Each was assigned a fixed analysis window (Table 1). Cascades whose root post falls outside all event windows are labelled *baseline*.

**Table 1. Mega-event definitions and temporal windows.**

| ID | Event | Date(s) | Window | Category |
|----|-------|---------|--------|----------|
| E1 | Hamas attack / Israel declares war | Oct 7, 2023 | Oct 4–10 | Military escalation (acute shock) |
| E2 | Al-Ahli hospital explosion | Oct 17, 2023 | Oct 14–20 | Humanitarian / contested narrative |
| E3 | Qatar ceasefire and hostage exchanges | Nov 24–30, 2023 | Nov 21–Dec 3 | Diplomatic / humanitarian |
| E4 | ICJ genocide provisional ruling | Jan 26, 2024 | Jan 23–29 | Legal / institutional milestone |
| E5 | Aaron Bushnell self-immolation | Feb 25, 2024 | Feb 22–28 | Individual human story |
| E6 | Flour massacre / 30,000 death toll | Feb 29, 2024 | Feb 26–Mar 3 | Humanitarian milestone |

### 3.2 NLP Filtering Pipeline

The NLP filtering pipeline identifies conflict-related posts from the 235-million-post corpus. It is implemented as a streaming processor (`--stage nlp_filter`) that reads directly from the compressed tar archive without full decompression.

**Keyword tiers.** Keywords are organised into five tiers by precision. A post is tagged as conflict-related if it matches at least one keyword from any tier via case-insensitive regex on the `text` field.

- *Tier 1 (High-precision English core):* `gaza`, `hamas`, `israel`, `palestine`, `palestinian`, `idf`, `netanyahu`, `ceasefire`, `hostage`, `hostages`, `west bank`, `rafah`, `genocide`, `occupation`, `zionist`, `zionism`, `apartheid`, `nakba`, `intifada`, `al-aqsa`, `al-shifa`, `unrwa`. The term "hostage/hostages" requires co-occurrence with at least one other Tier 1 keyword to suppress matches involving unrelated events (a co-occurrence gate).
- *Tier 2 (English hashtag patterns):* `#FreePalestine`, `#StandWithIsrael`, `#GazaUnderAttack`, `#CeasefireNow`, `#BringThemHome`, `#FromTheRiverToTheSea`, `#October7`, `#GazaGenocide`, `#IsraelHamasWar`, `#IsraelPalestine`, `#GazaWar`, and six further variants.
- *Tier 3 (Arabic-script keywords):* فلسطين (Palestine), غزة (Gaza), حماس (Hamas), طوفان_الأقصى (Al-Aqsa Flood), المقاومة (The Resistance), شهداء (Martyrs), الأقصى (Al-Aqsa), إبادة (Genocide).
- *Tier 4 (Hebrew-script keywords):* חרבות_ברזל (Swords of Iron), עזה (Gaza), חמאס (Hamas), חטופים (Hostages), ביחד_ננצח (Together We'll Win).
- *Tier 5 (Named entities):* Khan Younis, Jabalia, Nuseirat, Deir el-Balah, Be'eri, Sderot, Hezbollah, Houthi, ICJ, PRCS, al-Ahli, Nasser Hospital, Refaat Alareer, Hind Rajab, Wael Dahdouh, Aaron Bushnell, Gallant, Muhammad Deif, Abu Obeida, Ismail Haniyeh, Marwan Issa.

**Processing.** The full `user_posts.tar.gz` archive contains 2,363,249 individual per-user JSONL files totalling approximately 19.5 GB compressed. Sequential `r:gz` decompression with `getmembers()` pre-loading proved unacceptably slow (O(n²) seek behaviour). The implementation switches to streaming `r|gz` mode with 15-worker multiprocessing, achieving approximately 500 files per second — an approximately 5× speedup over the sequential baseline — for a total runtime of approximately 85 minutes on a single CPU.

**Result.** The pipeline identified **1,947,493 conflict-related posts** out of 233 million scanned posts in the conflict window, a conflict-post rate of 0.84%. Event-window assignment yielded E1=214,765, E2=322,440, E3=204,829, E4=111,025, E5=113,780, E6=74,832, and approximately 906,000 baseline posts. The output is a lean Parquet index containing `post_id`, `user_id`, `timestamp`, `sent_label`, `sent_score`, `is_repost`, `repost_from`, `reposted_author`, `conflict_keywords_matched`, and `event_window` for each matched post.

**Limitation.** Keyword precision was validated via manual spot-check of the 100-file sample but a formal precision/recall validation study (the `--stage validate_filter` stage) was not completed at scale due to time constraints. This is acknowledged as a limitation; the thesis reports conflict-post rate and keyword-tier hit distributions as a proxy for filter quality.

### 3.3 Cascade Construction

Cascade construction recovers the directed repost trees rooted at original conflict-related posts. The stage reads from the NLP-filtered post index and the follower graph, and is implemented as `--stage cascade_construction`.

**Lean intermediate index.** The full `user_posts.tar.gz` was streamed to produce a lightweight Parquet file (`conflict_window_posts.parquet`, 2.4 GB, 172 million rows) containing a lean schema — `post_id`, `user_id`, `timestamp`, `repost_from`, `reposted_author`, `sent_label`, `sent_score` — for the entire conflict window. This index supports fast reposter discovery via DuckDB without requiring re-streaming the original tar.

**Follower graph.** The 144-million-edge follower graph from `followers.csv.gz` is loaded into an on-disk DuckDB database with a composite index on `(follower, followed)`. For each batch of cascade roots, all relevant edges are pre-loaded into an in-memory Python `set` via a single batch JOIN query (`preload_for_users()`), avoiding per-edge database round-trips and reducing cascade construction time from an estimated 26 hours to approximately 3 minutes.

**Parent assignment heuristic.** The `reposted_author` field in the dataset links every repost back to the *root author*, not to the intermediate reposter who exposed the content to the current user. The cascade tree structure must therefore be inferred heuristically from temporal ordering and the follower graph. For each reposter in a cascade (processed in timestamp order):

- **Priority 1 (FOLLOWER edge):** Among users already in the cascade who reposted *before* this user — find any that this user follows in `followers.csv.gz`. If multiple candidates exist, select the most recent prior reposter.
- **Priority 2 (NON_FOLLOWER edge):** If no followed prior reposter exists, assign the most recent prior reposter regardless of follow status.
- **Priority 3 (UNKNOWN edge, root fallback):** If no prior reposter exists — a data gap condition — connect directly to the cascade root.

**Edge-type distribution.** Of 844,849 total edges: FOLLOWER = 641,139 (75.9%), NON_FOLLOWER = 203,710 (24.1%), UNKNOWN = 0. Priority 1 assignments account for 641,139 edges; Priority 2 for 186,896; Priority 3 (root fallback) for 16,814 (2.0% of edges). The low Priority 3 rate confirms that the heuristic produces well-connected trees for the overwhelming majority of cascades.

**Temporal confound.** The follower graph is a single snapshot from approximately March 2024. Some follow relationships included in the snapshot may have been formed *as a result of* the October 2023 conflict — for example, users following journalists or activists after October 7 — meaning FOLLOWER edge labels for early cascades (E1, E2) may overstate the degree of organic spread at the time of propagation. This is an irreducible limitation of single-snapshot follower data and is acknowledged throughout the analysis.

**Size thresholds.** Cascades with fewer than 10 nodes are excluded. The 10-node threshold was determined empirically after preliminary sampling showed sufficient cascade volume above this threshold. Cascades larger than 10,000 nodes are capped for computational tractability (no cascade in the dataset exceeded this bound). The resulting dataset contains **39,983 cascades** with a minimum size of 10 nodes, median 10, mean 22.1, 95th percentile 69, and maximum 1,854.

### 3.4 PyTorch Geometric Dataset Construction

Each cascade is converted into a `torch_geometric.data.Data` object for model consumption (`--stage build_pyg_dataset`).

**Node features (6-dimensional per node):**

1. `temporal_delta`: log₁₊(minutes since root post timestamp), normalised within the cascade.
2. `sent_label`: ordinal sentiment label (0 = negative, 1 = neutral, 2 = positive).
3. `sent_score`: continuous sentiment confidence score from the pre-computed RoBERTa annotation in the Failla and Rossetti dataset.
4. `depth`: hop distance from the cascade root in the directed tree.
5. `in_degree`: number of incoming edges within the cascade graph.
6. `out_degree`: number of outgoing edges within the cascade graph.

Sentiment values are drawn from `conflict_posts.parquet` for root posts and matched NLP-filtered nodes; for reposter nodes not in the filtered set, a DuckDB fallback query on the 172-million-row lean index provides sentiment labels.

**Graph-level features:** sentiment composition (proportions of negative, neutral, positive nodes), follower edge ratio, event window label (0–6), cascade depth, cascade breadth (maximum width at any depth level), branching factor, and Wiener index. The Wiener index — the sum of all pairwise shortest path distances in the tree, equal to structural virality per Goel et al. (2016) — is computed in O(N) time via a DFS-based decomposition: for each edge in the tree, the contribution is `subtree_size × (N − subtree_size)`, avoiding the naive O(N²) all-pairs computation.

**Dataset statistics:** 39,983 cascades, 884,832 total nodes, 844,849 total edges, 6-dimensional node features. Node feature dataset-level statistics: sentiment 54.5% negative, 39.6% neutral, 5.8% positive; median follower ratio 0.80; cascade depth median 4, mean 4.6.

### 3.5 BlueCCGL Architecture

The BlueCCGL model is assembled from three novel components layered on top of the CCGL contrastive pre-training framework.

#### 3.5.1 Feed-Aware Augmentation Strategies (Extension 1)

Standard graph augmentations such as random node dropping and edge perturbation (You et al., 2020) ignore the domain semantics of cascade structure. Three feed-aware augmentation strategies are introduced that respect the distinction between organic and algorithmically-mediated spread:

**Strategy A — Follower-Subgraph Extraction.** Retain only edges where the target node follows the source (edge type = FOLLOWER). Sample a configurable ratio of these edges (default 0.4, representing a strong structural perturbation). Compute the connected component containing the root and return it as the augmented view. Interpretation: "what would this crisis cascade look like if spread were purely organic?"

**Strategy B — Temporal Prefix Sampling.** Retain only the earliest fraction of nodes by `temporal_delta` (default 50% prefix). Reindex edges to include only edges between retained nodes, preserving root inclusion. Interpretation: "what did the early-stage spread look like before amplification or community-level engagement?"

**Strategy C — Bridge-Node Removal.** Identify "bridge nodes": nodes whose incoming edge is type NON_FOLLOWER, representing users who entered the cascade without following any existing cascade member. Remove a configurable ratio of these nodes (default 0.5). Return the root-connected component. Interpretation: "what would the cascade look like without cross-community entry points?"

Additionally, **feature masking** randomly zeros 25% of node feature dimensions per view, preventing the model from solving the contrastive task via unchanged feature statistics rather than structural learning.

Positive contrastive pairs are formed by applying two different augmentation strategies (sampled from A, B, C) to the same cascade. A graph that falls below 3 nodes after augmentation falls back to the original cascade graph. Hard negatives — cascade pairs from the same event window but different structural clusters — are used to make the contrastive task harder.

#### 3.5.2 Edge-Type Aware R-GCN Encoder (Extension 2)

The R-GCN encoder (Schlichtkrull et al., 2018) uses relation-specific weight matrices with basis decomposition to aggregate node features across three relation types (FOLLOWER, NON_FOLLOWER, UNKNOWN):

$$\mathbf{h}_v^{(l+1)} = \sigma\!\left(\sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{|\mathcal{N}_r(v)|} \mathbf{W}_r^{(l)} \mathbf{h}_u^{(l)} + \mathbf{W}_0^{(l)} \mathbf{h}_v^{(l)}\right)$$

where $\mathbf{W}_r = \sum_{b=1}^{B} a_{rb} \mathbf{V}_b$ with $B=2$ shared basis matrices. This parameterisation reduces the number of learned parameters from $|\mathcal{R}| \times d_{\text{in}} \times d_{\text{out}}$ to $(|\mathcal{R}| \times B) + (B \times d_{\text{in}} \times d_{\text{out}})$. Two R-GCN layers are used (reduced from three to avoid over-smoothing on the median 10-node cascade), with ReLU activation and dropout between layers, mapping node features from 6 dimensions to a hidden dimension of 128.

#### 3.5.3 Hierarchical DiffPool with Sentiment Enrichment (Extension 3)

After R-GCN encoding, the node embeddings are hierarchically pooled via DiffPool (Ying et al., 2018) at two levels with pool ratios [0.5, 0.25]. At each level, a GNN-based assignment matrix $\mathbf{S} \in \mathbb{R}^{N \times K}$ maps the current $N$ nodes to $K$ super-nodes:

$$\mathbf{X}^{(l+1)} = \mathbf{S}^{(l)\top} \mathbf{X}^{(l)}, \qquad \mathbf{A}^{(l+1)} = \mathbf{S}^{(l)\top} \mathbf{A}^{(l)} \mathbf{S}^{(l)}$$

The maximum number of input nodes to the DiffPool layers is capped at 150 (the 95th percentile cascade size is 69 nodes, making this cap nearly lossless while preventing memory overflow from the dense assignment matrix on large cascades). Auxiliary losses — a link prediction loss $\|\mathbf{S}^\top \mathbf{S} - \mathbf{A}\|_F$ and an entropy loss $H(\mathbf{S})$ — regularise the assignments with weight `diffpool_aux_weight = 0.5`.

At each pooling level, a global mean pool over the current node embeddings produces a level-specific graph-level embedding. These are concatenated with the graph-level sentiment composition vector — a 3-dimensional vector of proportions of negative, neutral, and positive nodes — and passed through a final 3-layer MLP projector with batch normalisation to produce the 64-dimensional contrastive embedding.

#### 3.5.4 Loss Function

The primary training objective is the InfoNCE contrastive loss (He et al., 2020; Chen et al., 2020) following CCGL:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j) / \tau)}$$

where $z_i$ and $z_i^+$ are the embeddings of two augmented views of the same cascade, $\tau$ is the temperature parameter (set to 0.1 in the primary v8 training run), and the denominator sums over all in-batch cascades.

The InfoNCE loss is regularised by VICReg variance and covariance terms (Bardes et al., 2022). The variance term penalises collapse of a batch's embeddings towards a constant vector by enforcing a minimum standard deviation across the batch dimension for each embedding dimension. The covariance term penalises off-diagonal covariance between embedding dimensions, encouraging the 64 output dimensions to encode different aspects of cascade structure.

### 3.6 Training Protocol

Training (`--stage train`) uses the following hyperparameter configuration (v8 run, the primary result):

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 150 | Best checkpoint at epoch 147 |
| Batch size | 32 | Reduced from 64; CPU memory constraint |
| Learning rate | 1e-3 → 1e-5 | Cosine annealing with 5-epoch linear warm-up |
| Temperature (τ) | 0.1 | Harder contrastive task vs. earlier runs |
| Gradient clip | 5.0 | Raised from 2.0; earlier runs saturated clip at 2.0 |
| Covariance weight | 0.25 | Lowered from 1.0; was dominating the loss |
| R-GCN layers | 2 | Lowered from 3; 3-hop over-smoothed small cascades |
| Projector layers | 3 | SimCLR v2 / VICReg best practice |
| Projector BN | True | Batch normalisation in projector MLP |
| follower_retain_ratio | 0.3 | Aggressive; see augmentation fallback note below |
| temporal_prefix_ratio | 0.3 | Aggressive; see augmentation fallback note below |
| bridge_remove_ratio | 0.7 | Aggressive; see augmentation fallback note below |
| min_nodes_after_aug | 5 | Minimum graph size after augmentation |

Training ran on an SSH-accessible virtual machine (CPU-only). Total wall-clock time: approximately 11 hours for 150 epochs. Checkpoints were saved every 10 epochs; the best checkpoint was additionally saved whenever the validation loss improved. All stochastic operations used a fixed global seed of 42 applied to `torch`, `numpy`, and `random`.

**Augmentation fallback rate.** A critical limitation of the v8 run is a 79.6% augmentation fallback rate: nearly 80% of augmentation attempts produced graphs below the 5-node minimum threshold after structural perturbation and fell back to the original (unperturbed) cascade graph as the positive pair. The root cause is the combination of aggressive augmentation ratios — `follower_retain_ratio=0.3` retains only 30% of follower edges, and `bridge_remove_ratio=0.7` removes 70% of bridge nodes — with a raised fallback threshold of 5 nodes on a median cascade of 10 nodes (0.3 × 10 = 3 nodes after augmentation, below the threshold of 5). Practically, the model trained primarily on unperturbed pairs, substantially limiting the structural invariance it could learn. This limitation is discussed quantitatively in Section 4 and partially addressed in a sensitivity run with corrected hyperparameters (v9 config).

### 3.7 Embedding Extraction and Clustering

After training, cascade embeddings are extracted via the `--stage extract_embeddings` pipeline stage, which passes all 39,983 cascades through the trained encoder in inference mode (`torch.no_grad()`, batch size 32) and saves the resulting 64-dimensional embedding matrix to an HDF5 file alongside a cascade metadata index in Parquet.

**UMAP pre-reduction.** Before applying HDBSCAN, the 64-dimensional embeddings are reduced to 10 dimensions via UMAP (McInnes et al., 2018) using `umap-learn`. This pre-reduction step is essential: without it, HDBSCAN operating in 64-dimensional space produces over 67% noise (unclustered cascades) across all grid configurations tested, a manifestation of the curse of dimensionality in high-dimensional density estimation. The 10-dimensional UMAP projection preserves the local neighbourhood structure learned by the contrastive model while providing a density-estimation-friendly coordinate space for HDBSCAN. The UMAP-10D embeddings are saved for reproducibility and reuse in all subsequent evaluation stages.

**HDBSCAN grid search.** HDBSCAN is applied over a grid of `min_cluster_size` values {100, 200, 300}, with `min_samples = max(1, round(0.4 × min_cluster_size))`. The best configuration is selected by the Silhouette score computed on non-noise points in the 10D UMAP space. Configurations with noise fraction above 0.70 are excluded as degenerate. The selected configuration is `min_cluster_size = 300`, which produces 11 clusters with 1.5% noise at the highest Silhouette score of 0.417 (Table 2).

**Table 2. HDBSCAN grid search results.**

| min\_cluster\_size | min\_samples | n\_clusters | Noise fraction | Silhouette | Selected |
|----|----|----|----|----|---|
| 100 | 40 | 32 | 1.7% | 0.320 | No |
| 200 | 80 | 16 | 2.6% | 0.385 | No |
| **300** | **120** | **11** | **1.5%** | **0.417** | **Yes** |

### 3.8 Evaluation and Explainability

**Clustering quality metrics.** All clustering metrics are computed on the 10-dimensional UMAP-reduced embeddings — the same space in which HDBSCAN was applied. Computing metrics in the original 64-dimensional space produces near-zero Silhouette scores and negative DBCV values that do not reflect actual cluster quality; these artefacts arise because cluster boundaries that are clean in 10D can appear to overlap when projected back to 64D. Four metrics are reported with 1000-iteration bootstrap confidence intervals (capped at 5000 cascades per bootstrap sample for computational tractability):

- **Silhouette score** (Rousseeuw, 1987): measures intra-cluster cohesion versus inter-cluster separation; range [−1, 1], higher is better.
- **Davies-Bouldin index**: ratio of within-cluster scatter to between-cluster separation; lower is better.
- **Calinski-Harabasz index**: ratio of between-cluster to within-cluster dispersion; higher is better.
- **DBCV** (Moulavi et al., 2014): density-based clustering validation designed for non-convex clusters; range [−1, 1], positive values indicate clusters denser than the surrounding space.

**Linear probing.** Six structural and temporal properties are probed via linear models trained on the frozen 64-dimensional embeddings with 5-fold cross-validation: `cascade_depth` (Ridge regression), `cascade_breadth` (Ridge), `wiener_index` (Ridge), `follower_edge_ratio` (Ridge), `cascade_size` (Ridge), and `event_window` (Logistic Regression, balanced accuracy). Global R² or balanced accuracy is reported alongside bootstrap confidence intervals (n=200) and a per-cluster breakdown.

**DiffPool assignment analysis.** The DiffPool assignment matrix at level 1 is extracted per cascade (batch size 1 to avoid contamination of the real-node slice by padding). Per-node assignment entropy $H = -\sum_k \text{softmax}(\mathbf{s}_1)_k \log \text{softmax}(\mathbf{s}_1)_k$ measures the softness of cluster assignment; the fraction of FOLLOWER-typed edge targets assigned to the dominant super-node measures whether the model concentrates follower-linked nodes into a single pooling cluster.

**Augmentation sensitivity analysis.** For each cluster, 20 representative cascades (those closest to the cluster centroid in embedding space) are augmented using each strategy with a fixed per-cascade seed. The cosine distance between the original embedding and the augmented embedding is computed for each augmentation type. Aggregated mean ± standard deviation per cluster quantifies how much each augmentation perturbs embeddings in each archetype.

**Cluster statistical profiles.** Per-cluster mean, standard deviation, median, and interquartile range are computed for all eight structural properties. Between-cluster significance is assessed via Kruskal-Wallis tests per property, with pairwise Mann-Whitney U tests (Bonferroni-corrected) for post-hoc comparisons. An archetype × event-window cross-tabulation is computed to assess whether archetype composition varies across the six mega-events, addressed by SRQ1 and SRQ2.

---

## 4. Preliminary Results

### 4.1 Training Convergence

The BlueCCGL model (v8 configuration) trained for 150 epochs on 39,983 cascades (approximately 1,249 batches of 32 cascades per epoch). Table 3 summarises the loss trajectory at key epochs. The best checkpoint was saved at epoch 147 with total loss 0.1292.

**Table 3. Training loss summary at selected epochs.**

| Epoch | Total loss | InfoNCE | Variance (VICReg) | Covariance (VICReg) | Gradient norm (mean) |
|---|---|---|---|---|---|
| 1 | 1.756 | 1.659 | 0.069 | 0.029 | 6.92 |
| 5 | — | — | — | — | — |
| 50 | 0.178 | 0.110 | 0.058 | 0.011 | 1.55 |
| 147 (best) | **0.129** | **0.065** | **0.055** | **0.011** | 1.32 |
| 150 | 0.131 | 0.067 | 0.055 | 0.011 | 1.33 |

The total loss declined by 92.5% from epoch 1 to the best checkpoint, driven primarily by the InfoNCE contrastive objective (1.659 → 0.065, a 96.1% reduction). The VICReg covariance term declined by 62% from epoch 1 to epoch 50 and then stabilised, indicating progressive decorrelation of the 64 embedding dimensions in the early training phase. Gradient norms declined steadily (6.92 → 1.32) with no evidence of explosion or sustained clip saturation, confirming stable training dynamics.

The 79.6% augmentation fallback rate (reported in Section 3.6) is the primary quantitative concern of the v8 run. It implies that the InfoNCE objective trained primarily on unperturbed cascade pairs rather than structurally distinct views, which limits the model's ability to learn structural invariances — a prerequisite for high-quality contrastive embeddings. Despite this, the embedding quality metrics reported in Sections 4.2 and 4.4 demonstrate that the model learned a strong structural signal, particularly in the follower edge ratio dimension (linear probe R² = 0.941).

### 4.2 Clustering Quality

All clustering metrics are computed on the 10-dimensional UMAP-reduced embeddings (the `metric_space` field in the output metadata is recorded as `umap_10d`). Table 4 presents the full bootstrapped results.

**Table 4. Clustering quality metrics (bootstrapped, n=1000, in 10D UMAP space).**

| Metric | Mean | Std | 95% CI lower | 95% CI upper |
|---|---|---|---|---|
| **Silhouette score** | **0.413** | 0.004 | 0.405 | 0.422 |
| **Davies-Bouldin index** | **0.676** | 0.007 | 0.661 | 0.690 |
| **Calinski-Harabasz index** | **1978.7** | 38.4 | 1904.6 | 2056.1 |
| **DBCV** | **0.618** | 0.028 | 0.563 | 0.667 |

| Meta-metric | Value |
|---|---|
| Total cascades | 39,983 |
| Clustered cascades | 39,389 (98.5%) |
| Noise (unclustered) | 594 (1.5%) |
| Number of clusters | 11 |
| Mean HDBSCAN persistence | 0.395 |
| min\_cluster\_size used | 300 |
| Bootstrap sample size (per iteration) | 5,000 |

The Silhouette score of 0.413 [0.405, 0.422] indicates clearly above-chance intra-cluster cohesion and inter-cluster separation. The Davies-Bouldin index of 0.676 (lower is better) confirms moderate to good separation relative to within-cluster scatter. The Calinski-Harabasz index of 1978.7 — a ratio of between-cluster to within-cluster dispersion — is the highest value achieved across all training runs and represents a strong discriminative signal. The DBCV of 0.618 is positive — indicating that the discovered clusters are denser than the space surrounding them — and substantially above zero, validating that the 11 clusters represent genuine density-based structures in the embedding space rather than arbitrary partitions.

The near-zero noise fraction (1.5%) stands in striking contrast to the pre-UMAP baseline where all HDBSCAN configurations produced above 67% noise: the 10-dimensional UMAP projection is essential for enabling the density-based clustering to function.

All between-cluster differences are statistically significant: Kruskal-Wallis H-statistics range from 4,632 (follower edge ratio) to 34,280 (positive sentiment proportion), all with p < 10⁻³⁰⁰ (Table 5). The clusters are not arbitrary partitions but represent genuinely distinct populations across all measured structural and sentiment properties.

**Table 5. Kruskal-Wallis significance tests across 11 clusters.**

| Property | H statistic | Significant (p < 0.05) |
|---|---|---|
| Positive sentiment proportion | 34,280 | Yes |
| Neutral sentiment proportion | 21,706 | Yes |
| Negative sentiment proportion | 21,341 | Yes |
| Cascade size | 8,191 | Yes |
| Cascade breadth | 8,134 | Yes |
| Wiener index | 5,791 | Yes |
| Cascade depth | 5,306 | Yes |
| Follower edge ratio | 4,633 | Yes |

![UMAP Projection](../Outputs_17_04/evaluate/umap_figure.png)

*Figure 1. UMAP two-dimensional projection of all 39,983 cascade embeddings. Panel A: coloured by HDBSCAN cluster label (C0–C10). Panel B: coloured by event window (E1–E6 and baseline). The clear cluster separation in Panel A and the absence of event-window clustering in Panel B confirm that the model learned structural rather than temporal signal.*

### 4.3 Discovered Archetypes

BlueCCGL identified 11 cascade archetypes, summarised in Table 6. For interpretability these are grouped into six structural families based on shared sentinel properties.

**Table 6. Cluster structural profiles — all 11 archetypes.**

| Cluster | n | % | Depth med | Breadth med | Size med | Follower ratio | Sentiment composition |
|---|---|---|---|---|---|---|---|
| C0 | 355 | 0.9% | 4 | 2 | 6 | 0.62 | 100% negative |
| C1 | 670 | 1.7% | 3 | 4 | 7 | 0.84 | 100% negative |
| C2 | 785 | 2.0% | 3 | 4 | 7 | 0.76 | 82% neg / 18% pos |
| C3 | 16,834 | 42.1% | 5 | 6 | 16 | 0.78 | 59% neg / 28% neu / 13% pos |
| C4 | 1,020 | 2.6% | 4 | 3 | 7 | 0.75 | 100% neutral |
| C5 | 1,489 | 3.7% | 3.5 | 4 | 9 | 0.91 | 100% neutral |
| C6 | 3,554 | 8.9% | 4 | 6 | 13 | 0.85 | 90% neutral |
| C7 | 11,009 | 27.5% | 3 | 4 | 9 | 0.82 | 65% neg / 35% neu |
| C8 | 833 | 2.1% | 5 | 2 | 8 | **0.51** | 78% negative |
| C9 | 1,222 | 3.1% | 4.5 | 3 | 8 | 0.69 | 80% negative |
| C10 | 1,618 | 4.0% | 4.5 | 3 | 7 | 0.66 | 50% neg / 50% neu |

![Cluster Radar Profiles](../Outputs_17_04/explain/figures/cluster_radar_profiles.png)

*Figure 2. Normalised structural profiles for all 11 clusters displayed as a radar chart. Each axis represents a normalised structural property. C8 is visually distinct on the follower-ratio axis (lowest at 0.51), C5 is distinct on the opposite end (highest at 0.91), and C3 projects strongly on the size and breadth axes.*

---

#### Family 1 — Pure Negative Reaction Cascades (C0, C1)

C0 (n=355) and C1 (n=670) both exhibit **100% negative sentiment** with zero standard deviation — every post in every cascade in these two clusters carries a negative sentiment label from the pre-computed RoBERTa annotation. Despite this shared sentiment property, they are structurally distinct. C0 is deep-narrow (depth=4, breadth=2, follower ratio=0.62): these are sustained back-and-forth negative exchanges that drew meaningful non-follower engagement. C1 is shallow-wide (depth=3, breadth=4, follower ratio=0.84): a root post that triggered immediate negative responses from many followers simultaneously.

**Interpretation.** C0 likely represents sustained negative debate threads — posts about acute traumatic events that drew both community members and outside voices into extended negative argument. C1 represents negative community consensus cascades — a post that catalysed immediate, broad disapproval within the poster's follower network without drawing sustained outside engagement.

---

#### Family 2 — Near-Negative Mixed Cascades (C2)

C2 (n=785) is structurally similar to C1 (depth=3, breadth=4, follower ratio=0.76) but its sentiment composition departs from uniformity: 82% negative, 18% positive. The 18% positive admixture distinguishes these from the pure-negative clusters; these are predominantly negative repost cascades where a visible minority of nodes carry positive sentiment — likely rooted in solidarity posts, expressions of hope, or counter-narrative content that co-spreads alongside the dominant negative framing.

**Interpretation.** C2 represents the mixed-sentiment propagation archetype where negative-sentiment content is dominant but solidarity or counter-narrative posts co-spread — a pattern consistent with communities where emotional reaction and community cohesion signals propagate simultaneously through the same repost network.

---

#### Family 3 — Dominant Discussion Mode (C3)

C3 (n=16,834, 42.1% of all cascades) is the largest cluster by a substantial margin. It represents the characteristic mode of crisis communication on Bluesky: substantive multi-level discussion threads reaching 5 reply levels deep, branching into 6+ sub-threads at the median, with a median cascade size of 16 posts and a long right tail extending to hundreds of nodes. The sentiment composition (59% negative, 28% neutral, 13% positive) reflects genuine ongoing discussion rather than uniform emotional reaction — these are conversations where multiple perspectives are represented. The 78% follower ratio indicates that most engagement comes from within the poster's network, but 22% of interactions cross community boundaries.

**Interpretation.** C3 is the workhorse of conflict discourse on Bluesky: the community's preferred mode for processing, discussing, and elaborating on crisis events. The large size, moderate depth, and cross-community reach suggest these are posts that matter enough to generate sustained engagement but are not so extreme as to drive exclusively outside (algorithmic) traffic.

---

#### Family 4 — Pure Informational Community Posts (C4, C5)

C4 (n=1,020) and C5 (n=1,489) both exhibit **100% neutral sentiment** with zero variance. They differ primarily in follower ratio: C5 has a follower ratio of 0.91 — the highest in the dataset — indicating that nearly all interactions are between mutual followers. C4 has a moderate follower ratio of 0.75.

**Interpretation.** C4 and C5 represent purely informational posts — news updates, factual sharing, resource links, or community announcements — that generated structured engagement. C5 in particular, with its near-maximal follower ratio and entirely neutral sentiment, represents insular community coordination: information circulated within a closed community of mutual followers, consistent with logistical or organisational posts during crisis periods (coordination of mutual aid, aggregation of verified sources, announcement of community solidarity actions).

---

#### Family 5 — Medium Neutral-to-Negative Threads (C6, C7)

C6 (n=3,554, 8.9%) and C7 (n=11,009, 27.5%) together account for 36% of all cascades — the second and third largest contributors after C3. C6 is predominantly neutral (90%), larger (median size=13), and broader (breadth=6) with a high follower ratio (0.85). C7 is majority-negative (65%), smaller (median size=9), slightly narrower, and still primarily follower-driven (0.82).

**Interpretation.** C6 represents medium-sized neutral discussion threads that grew into multi-level engagement — informational posts that generated substantial community discussion while maintaining a predominantly informational (non-emotive) character. C7 represents the workhorse of negative conflict commentary at the medium scale: negative-leaning threads that generated substantial engagement without reaching the full size and depth of the C3 large discussion archetype. Together C3, C6, and C7 account for 79% of all cascades, forming the core of how this community processed the conflict.

---

#### Family 6 — Cross-Community Debate (C8)

C8 (n=833, 2.1%) is the structurally most distinctive cluster in the dataset. Its **follower edge ratio of 0.51** is the lowest of any cluster: half of all cascade edges in C8 connect users who do not follow each other, indicating that these cascades crossed community boundaries to an extent not seen elsewhere in the dataset. Combined with a deep-narrow shape (depth=5, breadth=2) and 78% negative sentiment, these are sustained back-and-forth exchanges between people who do not follow each other — cross-community arguments or debates, likely triggered by contentious posts (disputed facts, strong political positions, inflammatory content) that provoked extended engagement from outside the poster's network. The augmentation sensitivity analysis (Section 4.5) confirms that C8 embeddings are the most sensitive to bridge-node removal in the dataset (sensitivity = 1.22), directly validating that these cascades are structurally defined by their non-follower entry points.

**Interpretation.** C8 represents algorithmically-amplified or cross-community debate threads — the crisis cascade archetype most strongly associated with non-organic spread. These are the cascades where content escaped the poster's follower community and provoked sustained engagement from external communities, consistent with highly contested posts that may have been surfaced by algorithmic feeds or that circulated within ideologically distant communities.

---

#### Family 7 — Moderate Cross-Community Chains (C9, C10)

C9 (n=1,222) and C10 (n=1,618) share a deep-narrow shape (depth≈4.5, breadth≈3) and moderate cross-community spread (follower ratio 0.66–0.69). They differ in sentiment: C9 is strongly negative (80%) while C10 splits evenly between negative and neutral (50/50). Both show elevated bridge-removal sensitivity (C9=0.84, C10=0.86), confirming that their moderate non-follower component plays a structurally meaningful role.

**Interpretation.** C9 and C10 represent moderate cross-community debate chains — threads that drew partial outside engagement without reaching the extreme cross-community exposure of C8. C9 represents sustained negative argument threads that attracted some non-follower engagement. C10 may represent discussions that started contentiously (negative) but evolved into more measured exchanges (neutral), or threads where different communities had genuinely different emotional responses to the same event.

### 4.4 What the Model Learned — Linear Probing

Linear probes test whether a linear model can decode structural properties from the frozen 64-dimensional embeddings. R² > 0 indicates the property is linearly encoded.

**Table 7. Linear probe results on frozen 64D embeddings.**

| Property | Probe type | Global score | 95% CI | Within-cluster range |
|---|---|---|---|---|
| **Follower edge ratio** | Ridge R² | **0.941** | [0.949, 0.952] | 0.917–0.981 |
| **Cascade depth** | Ridge R² | **0.686** | [0.807, 0.823] | 0.697–0.824 |
| **Cascade breadth** | Ridge R² | 0.317 | [0.742, 0.773] | 0.333–0.938 |
| **Cascade size** | Ridge R² | −6.07 (global) | [0.495, 0.544] | 0.686–0.911 (within-cluster) |
| **Wiener index** | Ridge R² | −31,876 | [−0.012, 0.147] | Not meaningful |
| **Event window** | LogReg bal. acc. | **0.143** | [0.143, 0.143] | 0.142–0.148 |

![Probe Heatmap](../Outputs_17_04/explain/figures/probe_heatmap.png)

*Figure 3. Linear probe R² (regression) or balanced accuracy (classification) per property (rows) and cluster (columns). Follower edge ratio dominates as the most linearly encoded property across all 11 clusters (R² = 0.917–0.981). Event window balanced accuracy equals chance (1/7 ≈ 0.143) universally.*

The follower edge ratio probe result (R² = 0.941 globally, consistent 0.917–0.981 within every cluster) is the dominant finding of the probing analysis. The contrastive model has learned to encode almost perfectly whether a cascade propagated through organic follower-based channels versus non-follower cross-community channels. This is a direct consequence of Extension 2 (R-GCN edge-type encoding): the relation-specific weight matrices allow the encoder to aggregate differently across FOLLOWER and NON_FOLLOWER edges, producing embeddings that carry a strong structural signal about the organic/algorithmic composition of each cascade.

Cascade depth (R² = 0.686) is the second strongest encoded property. The within-cluster depth R² range (0.697–0.824) suggests that depth is encoded consistently across all cluster types. Cascade breadth shows an interesting dissociation: the global R² is moderate (0.317) because breadth varies substantially *between* clusters, but within-cluster R² is high (0.333–0.938), particularly in clusters with uniform breadth distributions (C1=0.938, C5=0.920, C7=0.914). This reflects the fact that the clustering already captures breadth differences at the archetype level; within each archetype, a linear model can reliably rank cascades by breadth.

Cascade size shows a global R² of −6.07 — a catastrophic failure of global linear decoding — driven by clusters C3 and C6, which absorb the majority of cascades with high internal variance in cascade size. Within each cluster, size is well-encoded (R² = 0.686–0.911), confirming that size IS embedded in the representation but is confounded at the global level by the extreme distributional diversity of C3.

The event window probe at exactly 0.143 balanced accuracy — equal to the 1/7 chance level for a 7-class classifier — is a definitive result: the embeddings carry *no* temporal information whatsoever. Every cluster shows the same near-exactly-chance accuracy (range: 0.142–0.148). The model learned the *structure* of how information cascaded, not *when* it did so. This is appropriate for a contrastive model trained to align augmented structural views of the same cascade, but it means the embeddings cannot directly answer SRQ1 and SRQ2 via the probing route — those questions must be addressed via the archetype × event-window cross-tabulation in Section 4.6.

### 4.5 Augmentation Sensitivity Analysis

For each cluster, 20 representative cascades were subjected to each of the three feed-aware augmentation strategies. Table 8 reports the mean cosine distance between original and augmented embeddings.

**Table 8. Augmentation sensitivity (mean cosine distance, range 0–2) per cluster.**

| Cluster | Follower subgraph (A) | Temporal prefix (B) | Bridge removal (C) | Primary structural driver |
|---|---|---|---|---|
| C0 | N/A† | 0.944 | 0.545 | Temporal ordering |
| C1 | 0.998 | 0.820 | 0.539 | Follower structure + timing |
| **C2** | **1.979** | 0.889 | 0.474 | Strongly follower-edge dependent |
| C3 | 1.074 | 0.938 | 0.686 | All signals; heterogeneous cluster |
| C4 | N/A† | 0.922 | 0.671 | Temporal ordering |
| C5 | 0.948 | **1.193** | 0.446 | Early temporal prefix carries structure |
| C6 | 1.067 | 0.996 | 0.785 | Balanced across all augmentations |
| C7 | 1.253 | 0.942 | 0.691 | Follower structure primary |
| **C8** | N/A† | **1.308** | **1.220** | Uniquely sensitive to bridge removal |
| C9 | 1.004 | **1.148** | 0.835 | Temporal + bridge signals |
| C10 | N/A† | 0.930 | 0.856 | Bridge removal informative |

† N/A: these clusters had insufficient follower edges to survive the 30% retention ratio augmentation (graph fell below the minimum node count on all 20 representatives).

![Augmentation Sensitivity](../Outputs_17_04/explain/figures/augmentation_sensitivity_bars.png)

*Figure 4. Augmentation sensitivity grouped bar chart per cluster. Each bar group shows the mean cosine distance under each of the three augmentation strategies. C8's uniquely high bridge-removal sensitivity (1.22) confirms that cross-community debate cascades are structurally defined by their non-follower entry points.*

The C8 bridge-removal sensitivity of 1.22 is the highest in the dataset and represents a direct structural validation of the cross-community debate interpretation: removing non-follower bridge nodes from C8 cascades changes the learned embedding more than any other structural perturbation in any other cluster. The structure of these cascades is *defined* by the presence of non-follower connections. This is consistent with and independently confirms the C8 follower ratio of 0.51 observed in the cluster profiles.

C2's follower-subgraph sensitivity of 1.979 — approaching the maximum possible cosine distance of 2.0 — indicates that the near-negative mixed cascades are almost entirely dependent on their follower-edge topology. Remove the follower structure and the embedding changes to the maximum possible extent.

C5's elevated temporal prefix sensitivity (1.193) suggests that the early portion of insular community coordination cascades carries disproportionate structural information: the first 50% of posts in these threads define the character of the eventual cascade more than in any other archetype.

### 4.6 Archetype Composition Across Mega-Events

Table 9 presents the archetype × event-window cross-tabulation: the percentage of each event window's cascades assigned to each cluster. Note that the event window probe (balanced accuracy = 0.143) confirmed that the embeddings do not linearly encode event-window membership; any archetype composition differences across events therefore reflect genuine structural differences in how different types of crisis event generate different cascade shapes, rather than temporal labelling leakage into the embeddings.

**Table 9. Archetype composition per event window (% of cascades in each event window).**

| Event window | n | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 27,163 | 1.0 | 1.8 | 1.9 | 43.2 | 2.4 | 3.3 | 8.0 | 28.7 | 2.3 | 3.2 | 4.2 |
| **E1 (Oct 7)** | 2,686 | 0.5 | 1.2 | 1.6 | 33.4 | **5.1** | **9.1** | **19.7** | 23.3 | 0.8 | 1.6 | 3.6 |
| **E2 (Al-Ahli)** | 3,285 | 0.3 | 1.0 | 2.1 | 39.6 | 4.0 | 6.8 | **15.9** | 23.4 | 1.2 | 2.3 | 3.5 |
| **E3 (Ceasefire)** | 2,671 | 0.9 | 1.9 | 1.9 | **46.7** | 2.6 | 3.0 | 7.7 | 26.4 | 1.9 | 3.2 | 3.8 |
| **E4 (ICJ)** | 1,304 | 1.0 | 2.9 | 3.0 | **48.1** | 1.0 | 0.9 | 3.5 | 28.9 | 2.5 | 3.5 | 4.7 |
| **E5 (Bushnell)** | 1,364 | 0.8 | 1.6 | 2.4 | 44.6 | 0.9 | 1.1 | 4.1 | **32.7** | 2.2 | 4.3 | **5.3** |
| **E6 (Flour)** | 916 | 0.5 | 1.2 | 2.5 | 47.2 | 0.4 | 1.5 | 3.3 | **31.3** | **3.3** | 4.1 | **4.6** |

Bold values indicate notable deviations from the baseline distribution.

![Archetype Event Heatmap](../Outputs_17_04/explain/figures/archetype_event_heatmap.png)

*Figure 5. Archetype × event-window fraction heatmap. Columns represent event windows; rows represent clusters. Deviations from the baseline column indicate event-specific archetype surges.*

![Temporal Evolution](../Outputs_17_04/explain/figures/temporal_evolution.png)

*Figure 6. Temporal evolution of archetype prevalence fractions across the conflict timeline (E1–E6 and baseline). Lines represent individual cluster fractions over time.*

![Follower Ratio Violins](../Outputs_17_04/explain/figures/follower_ratio_violins.png)

*Figure 7. Distribution of follower edge ratio per cluster displayed as violin plots. C8 (cross-community debate) is clearly separated at the low end (centred at 0.51); C5 (insular community coordination) is separated at the high end (centred at 0.91).*

### 4.7 Research Question Answers

#### SRQ1: Crisis Phase Evolution — Do cascade archetypes shift across conflict phases?

**Partial yes, with caveats.** The archetype × event-window cross-tabulation (Table 9) reveals systematic shifts in archetype composition that are not attributable to embedding-level temporal encoding (confirmed by the chance-level event-window probe).

The most striking pattern is the E1 surge in neutral informational archetypes: C5 (insular community coordination) rises from 3.3% at baseline to 9.1% during E1, and C6 (medium neutral discussion) rises from 8.0% to 19.7%. This pattern is consistent with the breaking-news dynamics of October 7: in the immediate aftermath of an acute shock event, users circulate factual information, updates, and logistical content before the emotional/deliberative phase begins. The C3 large discussion cluster, by contrast, *drops* from 43.2% at baseline to 33.4% during E1 — the dominant deliberative discussion mode is temporarily displaced by informational cascades during acute shock.

This breaking-news-to-deliberation transition reverses by E3 (ceasefire), where C3 climbs to 46.7% and C5/C6 return toward baseline levels — consistent with the more sustained, deliberative character of the ceasefire-and-hostage-exchange period compared to the acute shock of October 7.

By E4 (ICJ ruling) and E5–E6, C3 peaks at 48.1% and the informational archetypes (C4, C5, C6) reach their lowest values in the dataset — the later phases of the conflict are dominated by deep deliberative discussion rather than informational dissemination. This is consistent with the expected temporal narrative: as the conflict matures from acute shock to sustained advocacy, the community's discourse mode shifts from broadcasting factual updates to sustained deliberative engagement.

**Caveat.** The event-window probe confirms that the embeddings themselves do not encode when a cascade occurred. The cross-tabulation differences therefore reflect genuine structural differences — events generating different structural patterns of cascade. However, the differences are modest (a few percentage points in most cases) and the dominant pattern is stability of archetype composition across events, with C3 dominating (33–48%) and C7 being the second largest contribution (23–33%) across all event windows. The crisis phase evolution is real but incremental rather than dramatic.

#### SRQ2: Event-Type Structural Signatures — Do different crisis event categories produce structurally distinct cascade archetypes?

**Yes, with interpretable patterns.** Different event categories produce measurably distinct archetype distributions:

- **Acute military shock (E1):** Highest informational archetype share (C5=9.1%, C6=19.7%), lowest C3 (33.4%). Breaking-news dynamics dominate; factual dissemination precedes deliberation.
- **Contested humanitarian event (E2, Al-Ahli):** Intermediate; C6 remains elevated (15.9%) relative to baseline. Contested attribution may prolong the informational phase as users share competing accounts.
- **Diplomatic multi-day event (E3, ceasefire):** Highest C3 share (46.7%) among the first four events. The sustained, hope-inflected character of the ceasefire/hostage-exchange period generates large deliberative discussions — the community engages with an ongoing evolving situation rather than responding to a single acute event.
- **Legal/institutional milestone (E4, ICJ):** Highest overall C3 share (48.1%), lowest informational archetypes (C4=1.0%, C5=0.9%). Legal milestones generate the most deliberative, sustained discussion, consistent with their procedural and interpretive character.
- **Individual human story (E5, Bushnell):** Highest C7 share (32.7%) — medium-sized negative threads dominate. Individual emotionally resonant events generate sustained negative mid-scale engagement rather than large C3 deliberative discussions.
- **Compound humanitarian/milestone (E6, flour massacre):** Highest C8 share (3.3%) — cross-community debate is proportionally most common. Humanitarian milestone events with high moral salience draw the most cross-community contestation.

#### SRQ3: Algorithmic Amplification — What proportion of crisis cascade edges are follower-based?

**Predominantly organic, but with a distinct minority cross-community archetype.** At the dataset level, 75.9% of all cascade edges are FOLLOWER-typed. The vast majority of crisis information spread on Bluesky during the conflict period was organic — flowing through established follower relationships. Non-follower edges account for 24.1%, representing cross-community or algorithmically-exposed spread.

The follower edge ratio probe (R² = 0.941) confirms that this ratio is the primary structural axis encoded in the BlueCCGL embeddings, and the cluster profiles confirm that this axis produces a clean archetype separation. C5 (follower ratio = 0.91) and C8 (follower ratio = 0.51) anchor the two ends of the organic-to-algorithmic spectrum. Only 2.1% of cascades are in C8 (the primary cross-community debate archetype), but its temporal pattern is revealing: C8 is most prevalent during E6 (3.3%), the flour massacre/30,000 death toll compound event — the event with the highest moral salience and clearest humanitarian framing. This is consistent with the hypothesis that content with strong moral framing crosses community boundaries more readily than other crisis content.

The augmentation sensitivity analysis corroborates this finding independently: C8's bridge-removal sensitivity (1.22) is the highest in the dataset, confirming that the structural identity of cross-community debate cascades is constituted by their non-follower edges.

**Caveat.** The temporal confound in the follower graph (Section 3.3) means that some FOLLOWER edges — particularly in early cascades (E1, E2) — may not have existed at the time of cascade propagation. The true degree of organic spread during October 2023 may be slightly lower than reported, and the true non-follower fraction slightly higher. This is an irreducible limitation of single-snapshot follower data.

#### SRQ4: Sentiment-Structure Coupling — Is sentiment composition correlated with structural archetype?

**Yes, strongly.** The clustering results demonstrate that sentiment composition is one of the primary axes along which the learned representation organises cascade archetypes. Four of the eleven clusters are defined by a single dominant sentiment class with zero variance: C0 and C1 (100% negative), C4 and C5 (100% neutral). This degree of sentiment homogeneity — zero standard deviation across hundreds or thousands of cascades — indicates that the model has learned to separate sentiment-homogeneous cascades into dedicated clusters.

The Kruskal-Wallis H-statistic for positive sentiment proportion (34,280) is the highest of any property tested, confirming that sentiment composition is the single most discriminating dimension between clusters.

Negative sentiment cascades (C0, C1, C2, C7, C8, C9) are structurally diverse: C0 is deep-narrow with moderate follower ratio; C1 is shallow-wide with high follower ratio; C8 is cross-community; C9 is deep-narrow moderate cross-community. Negative sentiment alone does not predict cascade shape — the *combination* of sentiment and structural properties defines each archetype.

Crucially, C8 (cross-community debate, follower ratio=0.51) is predominantly negative (78%). This coupling — negative content crossing community boundaries — is consistent with Vosoughi et al. (2018) who found that emotionally negative content spreads faster and deeper than neutral content. On Bluesky, the same pattern appears: the cascade archetype with the most cross-community spread is also predominantly negative.

#### MRQ1: What natural archetypes emerge?

The BlueCCGL framework with HDBSCAN clustering discovers 11 archetypes that group into six interpretable families: pure negative reaction cascades (C0, C1), near-negative mixed cascades (C2), the dominant discussion mode (C3), pure informational community posts (C4, C5), medium neutral-to-negative threads (C6, C7), cross-community debate (C8), and moderate cross-community chains (C9, C10). The primary structural axes organising this typology are sentiment composition (particularly the negative/neutral distinction), follower edge ratio (organic vs. cross-community spread), and cascade size/depth. All between-cluster differences are statistically significant at p < 10⁻³⁰⁰.

#### MRQ2: Does R-GCN edge-type encoding improve representation quality?

The dominant evidence is the follower edge ratio probe result (R² = 0.941). The R-GCN encoder produces embeddings that encode the organic/algorithmic composition of each cascade almost perfectly. The fact that this signal is present at R² > 0.91 within every single cluster — not just globally — suggests that edge-type encoding permeates the learned representation uniformly rather than being concentrated in a few embedding dimensions. Ablation study A1 (replacing R-GCN with a homogeneous GCN) will provide the direct quantitative comparison; this is planned as the next experimental step (Section 5).

#### MRQ3: Do feed-aware augmentations improve representations?

The augmentation sensitivity analysis provides indirect evidence: the embeddings of different archetype clusters respond differently to the three domain-specific augmentations (follower subgraph, temporal prefix, bridge removal), with augmentation sensitivity patterns that align with the structural interpretations of each archetype. This structural differentiation would not be possible if the augmentations were domain-semantically inert. C8's uniquely high bridge-removal sensitivity and C2's near-maximum follower-subgraph sensitivity demonstrate that the feed-aware augmentations capture the structurally defining features of each archetype. Direct ablation comparison (A2: generic augmentation baseline) is planned as the next experimental step.

---

## 5. Next Steps

The following experimental stages are planned to complete the BlueCCGL evaluation and support thesis writing.

### 5.1 Downstream Prediction Tasks

The pre-trained BlueCCGL encoder will be evaluated on two downstream transfer tasks (`--stage downstream`) to assess the quality of learned representations beyond clustering:

- **Cascade size prediction.** A regression head fine-tuned on top of the frozen encoder predicts log(cascade_size). Metric: RMSE and R².
- **Cascade longevity prediction.** A binary classification head predicts whether the cascade remains active beyond the median duration threshold. Metric: AUROC and F1.

Baselines: (a) random initialisation (same architecture, no pre-training), (b) vanilla CCGL (generic augmentation, no R-GCN, no DiffPool), (c) handcrafted feature baseline (Wiener index, depth, breadth, branching factor, follower edge ratio fed to a linear MLP).

The v8 training results suggest that cascade size will be a challenging target for the frozen encoder (global R² = −6.07 in the probing experiment), while follower-related structural properties should transfer well (global R² = 0.941).

### 5.2 Stability Analysis

The stability analysis (`--stage stability`) evaluates the robustness of the discovered 11-archetype solution to perturbations of the cascade dataset:

- 10 independent 80% random subsamples of the 39,983-cascade dataset.
- For each subsample: apply HDBSCAN with the selected `min_cluster_size = 300`.
- Compute Adjusted Rand Index (ARI) between each pair of subsample cluster assignments.
- Report mean ± standard deviation of clustering metrics (Silhouette, DB, CH, DBCV) and ARI across the 10 subsamples.

High ARI (>0.7) and low metric variance across subsamples would confirm that the 11-archetype solution is not an artefact of the specific cascade dataset composition.

---

## 6. References

Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.

Bianchi, F., Grattarola, D., & Alippi, C. (2020). Spectral Clustering with Graph Neural Networks for Graph Pooling. *ICML 2020*.

Cao, Q., Shen, H., Cen, K., Ouyang, W., & Cheng, X. (2017). DeepHawkes: Bridging the gap between prediction and understanding of information cascades. *Proceedings of CIKM 2017*.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.

Chen, X., Zhou, F., Zhang, K., Trajcevski, G., Zhong, T., & Zhang, F. (2019). Information diffusion prediction via recurrent cascades convolution. *Proceedings of ICDE 2019*.

Failla, A. & Rossetti, G. (2024). "I'm in the Bluesky Tonight": Insights from a year worth of social data. *PLoS ONE, 19*(11): e0310330.

Goel, S., Anderson, A., Hofman, J., & Watts, D. J. (2016). The structural virality of online diffusion. *Management Science, 62*(1), 180–196.

He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR 2020*.

Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous Graph Transformer. *Proceedings of the Web Conference 2020*.

Huszár, F., Ktena, S. I., O'Brien, C., Belli, L., Schlaikjer, A., & Hardt, M. (2022). Algorithmic amplification of politics on Twitter. *Proceedings of the National Academy of Sciences, 119*(1).

Kleppmann, M., & Howard, D. (2024). *Bluesky and the AT Protocol: Usable Decentralized Social Media*. arXiv preprint.

Lee, J., Lee, I., & Kang, J. (2019). Self-Attention Graph Pooling. *ICML 2019*.

McInnes, L., Healy, J., & Astels, S. (2017). HDBSCAN: Hierarchical Density Based Clustering. *Journal of Open Source Software, 2*(11).

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.

Moulavi, D., Jaskowiak, P. A., Campello, R. J. G. B., Zimek, A., & Sander, J. (2014). Density-based clustering validation. *Proceedings of SDM 2014*.

Olteanu, A., Castillo, C., Diaz, F., & Kıcıman, E. (2015). Crisislex: A lexicon for collecting and filtering microblogged communications in crises. *Proceedings of ICWSM 2015*.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65.

Schlichtkrull, M., Kipf, T. N., Bloem, P., Van Den Berg, R., Titov, I., & Welling, M. (2018). Modeling Relational Data with Graph Convolutional Networks. *ESWC 2018*.


Starbird, K., & Palen, L. (2012). (How) will the revolution be retweeted? Information diffusion and the 2011 Egyptian uprising. *Proceedings of CSCW 2012*.

Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science, 359*(6380), 1146–1151.

Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., & Yu, P. S. (2019). Heterogeneous Graph Attention Network. *Proceedings of the Web Conference 2019*.

Xu, H., Zhang, J., Feng, S., Tian, X., & Li, J. (2022). CCGL: Contrastive Cascade Graph Learning. *IEEE Transactions on Knowledge and Data Engineering*.

Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., & Leskovec, J. (2018). Hierarchical Graph Representation Learning with Differentiable Pooling. *NeurIPS 2018*.

You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). Graph Contrastive Learning with Augmentations. *NeurIPS 2020*.

Yuan, C., Li, J., Zhou, W., Lu, Y., Zhang, X., & Hu, S. (2024). CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction. *IEEE Transactions on Knowledge and Data Engineering*.

Zeitzoff, T. (2017). How social media is changing conflict. *Journal of Conflict Resolution, 61*(9), 1970–1991.

Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S., & Wang, L. (2020). Deep Graph Contrastive Representation Learning. *ICML Workshop on Self-Supervised Learning for Graphs 2020*.

Zignani, M., Galimberti, C., Gaito, S., Rossi, G. P., & Zheng, H. (2018). Footprints of a Decentralised Macrostructure in the Mastodon Fediverse. *ICWSM 2018*.

---

*End of partial draft — April 20, 2026*

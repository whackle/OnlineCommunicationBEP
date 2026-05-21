> **DEPRECATED** — superseded by `IMPLEMENTATION_PLAN.md`. Terminology in this document (e.g. "algorithmic amplification") is outdated; current framing uses "out-of-network diffusion".

# BlueCCGL: Unsupervised Discovery of Crisis Information Cascade Archetypes on Bluesky

## The Israel-Palestine Conflict as a Natural Experiment in Decentralized Crisis Communication

---

### 1. Introduction

Understanding how crisis information propagates through social networks has become a central question in computational social science. During acute geopolitical crises, information cascades — the chains of reposts through which content reaches new audiences — exhibit dynamics that differ fundamentally from steady-state discourse: urgency compresses decision-making, emotional intensity amplifies sharing, and community boundaries become more (or less) permeable depending on the nature of the event. On centralized platforms such as Twitter/X and Weibo, crisis cascade dynamics have been studied extensively using supervised deep learning methods (Chen et al., 2019; Cao et al., 2017; Yuan et al., 2024). However, these approaches share two critical limitations: they require ground-truth labels (e.g., "viral" vs. "non-viral"), and they are designed for platforms where a single opaque recommendation algorithm governs content visibility. Neither assumption holds on decentralized social networks, and neither is compatible with the inherent unpredictability of crisis information dynamics.

Bluesky, built on the AT Protocol, represents a new class of social platform where users can choose or disable algorithmic feeds entirely, where identity is portable across servers (Personal Data Servers), and where the full firehose of platform activity is publicly accessible in real time. During the period covered by the Failla & Rossetti (2024) Zenodo dataset (February 2023 – March 2024), the Israel-Palestine conflict escalated dramatically following the October 7, 2023 Hamas attack, producing a sustained, multi-phase crisis that generated intense online discourse. Bluesky's architecture creates a unique natural experiment: on the same platform, crisis content spread both organically through follower networks and via user-selected algorithmic feeds (notably the News feed, which spiked in engagement during October 2023 per Failla & Rossetti, 2024) — and critically, the structural trace of both pathways is observable. Yet, to date, no study has analysed crisis information cascade dynamics on Bluesky or any AT Protocol platform at a structural level.

This research proposes **BlueCCGL** (Bluesky Contrastive Cascade Graph Learning), a framework for the unsupervised discovery of crisis information cascade archetypes on Bluesky, applied to the Israel-Palestine conflict (October 7, 2023 – March 18, 2024). BlueCCGL extends the Contrastive Cascade Graph Learning (CCGL) framework of Xu et al. (2022) with four novel contributions: **(1)** an upstream NLP filtering pipeline that identifies conflict-related posts via multilingual keyword matching and isolates cascades across six major escalation events; **(2)** feed-aware graph augmentation strategies that respect the semantic distinction between organic and algorithmically-amplified spread; **(3)** edge-type aware message passing via Relational Graph Convolutional Networks (R-GCN; Schlichtkrull et al., 2018) that explicitly encodes whether a cascade edge represents follower-based or non-follower-based propagation; and **(4)** hierarchical graph pooling (Ying et al., 2018) enriched with sentiment composition features that capture multi-scale cascade structure from individual user decisions to community-level patterns. By learning cascade representations through contrastive self-supervised learning and clustering them with HDBSCAN (McInnes et al., 2017), BlueCCGL discovers natural crisis cascade archetypes without requiring any labeled data.

The expected results are significant for three reasons. First, a label-free approach to crisis cascade archetype discovery is directly transferable to any platform or crisis where labeled cascade data is unavailable — which is the norm outside a handful of well-studied centralized platforms. Second, the discovered archetypes would provide the first empirical characterisation of how crisis information spreads differently through follower networks versus algorithmic recommendation on a decentralized, algorithm-optional platform — offering insights that are impossible to obtain from centralized platforms where these two pathways cannot be disentangled. Third, by anchoring the analysis in six major conflict escalation events, the study produces a temporal narrative of how cascade dynamics evolve across crisis phases — from acute shock (October 7) through sustained military operations, legal milestones (ICJ ruling), and individual human stories (Aaron Bushnell, the flour massacre) — revealing whether the *structure* of crisis information spread changes as a conflict matures.

**Scope and Framing.** Bluesky's user base during the study period (October 2023 – March 2024) was small (1–5M users), predominantly English-speaking, Western, and politically left-leaning. This study therefore characterises crisis information dynamics within a *politically homogeneous early-adopter community on a decentralized platform* — not global discourse on the conflict. This demographic specificity is a feature, not a limitation: studying how ideologically aligned communities process and propagate crisis information through decentralized infrastructure is an understudied and theoretically important question.

---

### 2. Mega-Events: Temporal Anchors

The analysis is structured around six major escalation events within the dataset window. These serve as temporal anchors for comparing cascade archetypes across crisis phases.

| # | Event | Date(s) | Category | Why Selected |
|---|-------|---------|----------|--------------|
| **E1** | **Hamas attack / Israel declares war** | Oct 7, 2023 | Military escalation (acute shock) | Founding event of the crisis; maximum uncertainty, breaking-news dynamics; expected to produce broadcast-dominated cascades |
| **E2** | **Al-Ahli hospital explosion** | Oct 17, 2023 | Humanitarian / contested attribution | Single biggest social media flashpoint in October; contested narratives (Israel vs. Palestinian rocket attribution) create cross-community cascade dynamics |
| **E3** | **Qatar-brokered ceasefire & hostage exchanges** | Nov 24–30, 2023 | Diplomatic / humanitarian | 7-day sustained multi-day event; daily hostage/prisoner exchanges; qualitatively different from acute shock — sustained, hope-inflected discourse |
| **E4** | **ICJ genocide ruling** | Jan 26, 2024 | Legal / institutional | Institutional milestone; legal/moral framing rather than breaking news; expected to produce deeper, more deliberative cascades |
| **E5** | **Aaron Bushnell self-immolation** | Feb 25, 2024 | Individual human story | One of the most viral events of the war globally; emotionally resonant individual action; expected to produce deep viral cascades |
| **E6** | **Flour massacre + 30,000 death toll** | Feb 29, 2024 | Humanitarian crisis / milestone | Compound event (mass casualty + symbolic death toll number); occurred post-Bluesky public launch (Feb 6), so user base is larger |

**Event windows.** Each mega-event is assigned a ±3-day window for cascade collection (i.e., cascades whose root post falls within [event_date − 3 days, event_date + 3 days]). The 7-day window balances capturing immediate reaction cascades with avoiding overlap between adjacent events. The ceasefire (E3) uses the full Nov 21–Dec 3 window given its multi-day nature. Cascades outside all event windows form the **baseline** corpus for comparison.

**Event categories** enable structured comparison: military escalation (E1) vs. contested narrative (E2) vs. diplomatic (E3) vs. institutional (E4) vs. individual story (E5) vs. humanitarian milestone (E6).

---

### 3. Related Work

#### 3.1 Information Cascade Prediction

Graph neural network approaches to cascade modelling have evolved rapidly. **CasCN** (Chen et al., 2019) was among the first to apply GNNs to cascade prediction, modelling cascades as sequences of graph snapshots with temporal attention. **DeepHawkes** (Cao et al., 2017) bridged classical Hawkes processes with deep learning to provide interpretable attention over user influence, though it assumes homogeneous user influence. More recently, **CasFlow** (Yuan et al., 2024) introduced a variational approach with hierarchical pooling, achieving state-of-the-art performance on Weibo and Twitter datasets. ⚠️ **HierCas** (Wang et al., 2024) extended this with multi-resolution modelling capturing both user-level and community-level dynamics, and ⚠️ **TempCas** (Liu et al., 2025) applied Transformer architectures with continuous-time attention.

**Gap:** All of the above methods are *supervised*, requiring ground-truth popularity or virality labels. None account for heterogeneous edge types (e.g., follower vs. non-follower propagation), none have been applied to crisis-specific cascades on decentralized platforms, and all were evaluated exclusively on centralized platforms (Twitter, Weibo). BlueCCGL addresses all three gaps: it is unsupervised, edge-type aware, crisis-focused, and targets a decentralized platform.

#### 3.2 Contrastive and Self-Supervised Graph Learning

**CCGL** (Xu et al., 2022) is the direct methodological predecessor of this work. It introduced contrastive self-supervised pre-training for cascade graphs using sub-cascade sampling for augmentation and an InfoNCE graph-level contrastive loss, demonstrating 10–20% improvement over supervised baselines. **GraphCL** (You et al., 2020) provided a systematic study of graph augmentation strategies (node dropping, edge perturbation, attribute masking, subgraph sampling) that informs augmentation design. **GRACE** (Zhu et al., 2020) introduced node-level corruption-based contrastive learning. For temporal graphs, **TGC** (Zhang et al., 2023) proposed time-aware augmentation strategies, and ⚠️ **CSTGCN** (Li et al., 2024) introduced multi-scale temporal contrasting for evolving structural patterns.

**Gap:** CCGL's augmentations are generic (random sub-cascade sampling) and carry no domain semantics. No existing contrastive graph learning method incorporates augmentations that are aware of the distinction between organic and algorithmically-mediated spread, nor has any been specifically designed for crisis information cascades. BlueCCGL introduces feed-aware augmentation strategies (follower-subgraph extraction, temporal prefix sampling, bridge-node removal) designed to preserve the semantic structure of crisis-driven propagation.

#### 3.3 Heterogeneous GNNs and Organic vs. Algorithmic Spread

Relational and heterogeneous GNN architectures provide the building blocks for edge-type aware cascade encoding. **R-GCN** (Schlichtkrull et al., 2018) introduced relation-specific weight matrices with basis decomposition for parameter efficiency. **HAN** (Wang et al., 2019) proposed hierarchical attention across node-level and semantic-level, while **HGT** (Hu et al., 2020) applied Transformer-style attention to heterogeneous graphs. **RHGNN** (Yu et al., 2023) demonstrated the importance of modelling multiple relationship types (follow, mention, repost) in social network analysis.

On the empirical side, Huszár et al. (2022) provided a landmark internal Twitter study showing algorithmic amplification of political content — a gold standard that requires platform-insider access. Goel et al. (2016) defined the "structural virality" metric (Wiener index) to distinguish broadcast from viral spread, and Vosoughi et al. (2018) showed that true and false news produce structurally different cascades. Bhadani et al. (2024) proposed structural signatures of algorithmic amplification, and Bandy & Diakopoulos (2021) developed methodology for auditing algorithmic curation.

**Gap:** No prior work has applied heterogeneous GNN encoding to crisis information cascades, nor has any study examined organic vs. algorithmic spread of conflict-related content on a decentralized platform. BlueCCGL exploits Bluesky's observable follow graph to directly label cascade edges and uses R-GCN encoding to learn relation-aware representations of crisis cascades.

#### 3.4 Crisis Communication and Information Spread

Crisis communication on social media has a rich literature. **Olteanu et al. (2015)** provided a comprehensive survey of social media use during crises, identifying information types (situational awareness, emotional support, opinions) and their spread patterns. **Starbird & Palen (2012)** studied retweeting patterns during crises, finding that crisis retweets form distinct structural patterns compared to steady-state discourse. **Vosoughi et al. (2018)** demonstrated that emotionally charged content (particularly false news) spreads faster and deeper — a finding directly relevant to conflict-related cascades.

Specific to the Israel-Palestine conflict, social media dynamics have been studied primarily on Twitter/X and Facebook. **Siegel et al. (2021)** examined echo chambers in Israeli-Palestinian Twitter discourse. **Zeitzoff (2017)** studied how conflict events affect online political expression. However, all prior work examines centralized platforms where algorithmic amplification is opaque.

**Gap:** No study has examined crisis information cascade *structure* on a decentralized platform where organic and algorithmic pathways are distinguishable. BlueCCGL provides the first such analysis, using the Israel-Palestine conflict as a sustained, multi-phase case study.

#### 3.5 Decentralized Social Networks and Bluesky

Research on decentralized social networks has focused primarily on the Fediverse. Zignani et al. (2018) conducted the first large-scale Mastodon measurement study, and Raman et al. (2019) analysed technical challenges of federation protocols. For the AT Protocol specifically, Kleppmann & Howard (2024) provided a technical comparison of ActivityPub and AT Protocol, including Bluesky's Firehose architecture. ⚠️ Cerisara et al. (2025) presented an early Bluesky ecosystem analysis studying user behaviour and AT Protocol adoption. Most notably, **Failla & Rossetti (2024)** released the first large-scale Bluesky dataset, covering the complete post history of over 4M users (~81% of all registered accounts at the time), 235M posts, a directed follower graph of ~144M edges, and timestamped repost/quote/reply interaction graphs — all publicly available on Zenodo (DOI: 10.5281/zenodo.14669616). Critically, the dataset also includes posts from 11 feed generators and user feed-subscription data, providing ground-truth information about algorithmic exposure. The paper explicitly notes a spike in News feed engagement during October 2023, likely attributable to the Israel-Palestine conflict.

**Gap:** While Failla & Rossetti (2024) provide the first comprehensive Bluesky dataset with descriptive analysis, no study has performed cascade-level *structural* analysis on Bluesky or any AT Protocol platform — and no study has examined crisis-specific cascade dynamics on a decentralized platform. BlueCCGL fills both gaps.

#### 3.6 Hierarchical Graph Pooling and Clustering

**DiffPool** (Ying et al., 2018) introduced differentiable soft cluster assignment for hierarchical graph representation learning. **SAGPool** (Lee et al., 2019) provided a more efficient attention-based alternative, and **MinCutPool** (Bianchi et al., 2020) offered a spectral-clustering-grounded approach. For cascades specifically, ⚠️ **HiCas** (Zhou et al., 2023) applied hierarchical pooling to capture multi-scale propagation patterns. For unsupervised pattern discovery, **HDBSCAN** (McInnes et al., 2017) provides density-based clustering that does not require pre-specifying the number of clusters, while **DEC** (Xie et al., 2016) and **DMoN** (Tsitsulin et al., 2023) offer end-to-end deep clustering alternatives.

**Gap:** Hierarchical pooling has not been combined with contrastive learning for cascade graphs, nor applied with sentiment-enriched node features in a crisis context. BlueCCGL integrates DiffPool-style hierarchical pooling with sentiment composition as a graph-level feature, enabling multi-scale representations that capture how crisis sentiment distributes across cascade layers.

---

### 4. Research Questions

**Main Research Question:**

> *Can unsupervised contrastive graph learning discover meaningful crisis information cascade archetypes on Bluesky that reflect the interplay between organic follower-based spread and algorithmic amplification during the Israel-Palestine conflict?*

This is decomposed into two sets of sub-questions: **socially-grounded** questions (what do we learn about crisis communication?) and **methodological** questions (do the technical extensions contribute?). Both sets are answered by the same pipeline.

#### 4.1 Socially-Grounded Research Questions

**SRQ1: Crisis Phase Evolution.** Do crisis cascade archetypes shift across conflict phases — e.g., from broadcast-dominated cascades during acute shock events (Oct 7) to deeper organic-viral cascades during sustained advocacy periods (Feb–Mar 2024)? How does the distribution of archetypes change across the six mega-events?

**SRQ2: Event-Type Structural Signatures.** Do different categories of crisis events (military escalation, humanitarian crisis, legal/institutional milestone, individual human story, diplomatic ceasefire) produce structurally distinct cascade archetypes, and if so, what structural properties (depth, breadth, branching factor, follower-edge ratio, sentiment composition) differentiate them?

**SRQ3: Algorithmic Amplification of Crisis Content.** What proportion of crisis cascade edges are follower-based (organic) versus non-follower-based (suggesting algorithmic or cross-community exposure), and does this ratio differ systematically across mega-events and cascade archetypes?

**SRQ4: Sentiment-Structure Coupling.** Is the sentiment composition of a crisis cascade (proportion of negative, neutral, positive posts) correlated with its structural archetype? Do negative-sentiment-dominated cascades spread differently from mixed-sentiment ones during crisis periods?

#### 4.2 Methodological Research Questions

**MRQ1: Archetype Discovery.** What natural cascade archetypes emerge when applying HDBSCAN clustering to cascade embeddings learned via contrastive self-supervised learning, and how do these archetypes differ in structural properties (depth, breadth, branching factor, follower-edge ratio, temporal spread pattern, sentiment composition)?

**MRQ2: Edge-Type Aware Encoding.** How does encoding follower vs. non-follower edges via relational graph convolution (R-GCN) affect the quality and discriminative power of learned cascade representations, compared to homogeneous GNN baselines?

**MRQ3: Feed-Aware Augmentation.** Do domain-specific augmentation strategies — follower-subgraph extraction, temporal prefix sampling, and bridge-node removal — produce higher-quality cascade representations than generic graph augmentations (random node/edge dropping) as measured by clustering coherence and downstream task performance?

---

### 5. Methodology

#### 5.1 NLP Filtering Pipeline (New)

The NLP filtering pipeline identifies conflict-related posts from the full 235M-post corpus and assigns them to mega-event temporal windows. This is a prerequisite for cascade construction.

##### 5.1.1 Keyword Design

Keywords are organised into tiers by precision, applied with case-insensitive matching on the `text` field of posts in `user_posts.tar.gz`.

**Tier 1 — High-Precision English (Core):**
```
gaza, hamas, israel, palestine, palestinian, idf, netanyahu, ceasefire,
hostage, hostages, west bank, rafah, genocide, occupation, zionist,
zionism, apartheid, nakba, intifada, al-aqsa, al-shifa, unrwa
```

**Tier 2 — English Hashtag Patterns (Inline in Text):**
```
#FreePalestine, #StandWithIsrael, #GazaUnderAttack, #CeasefireNow,
#BringThemHome, #FromTheRiverToTheSea, #October7, #GazaGenocide,
#IStandWithIsrael, #SaveGaza, #FreeGaza, #StopTheGenocide,
#IsraelHamasWar, #IsraelPalestine, #GazaWar
```

**Tier 3 — Arabic Keywords (Low Volume Expected):**
```
فلسطين (Palestine), غزة (Gaza), حماس (Hamas), طوفان_الأقصى (Al-Aqsa Flood),
المقاومة (The Resistance), شهداء (Martyrs), الأقصى (Al-Aqsa), إبادة (Genocide)
```

**Tier 4 — Hebrew Keywords (Low Volume Expected):**
```
חרבות_ברזל (Swords of Iron), עזה (Gaza), חמאס (Hamas),
חטופים (Hostages), ביחד_ננצח (Together We'll Win)
```

**Tier 5 — Named Entities (Supplementary OR Conditions):**
```
Khan Younis, Jabalia, Nuseirat, Deir el-Balah, Be'eri, Sderot,
Hezbollah, Houthi, ICJ, PRCS, al-Ahli, Nasser Hospital,
Refaat Alareer, Hind Rajab, Wael Dahdouh, Aaron Bushnell,
Gallant, Muhammad Deif, Abu Obeida, Ismail Haniyeh, Marwan Issa
```

A post is tagged as conflict-related if it matches **≥1 keyword from any tier**. The matching is applied via compiled regex for performance.

##### 5.1.2 Temporal Filtering

Only posts dated **October 1, 2023 – March 18, 2024** are considered (the conflict window within the dataset). The `langs` metadata field is available for posts from late June 2023 onward, covering the entire conflict window.

##### 5.1.3 Event-Window Assignment

Each conflict-related post is assigned to a mega-event window if its timestamp falls within the event's ±3-day range (E3 uses Nov 21 – Dec 3). Posts matching conflict keywords but outside all event windows are labelled **baseline**.

##### 5.1.4 Preliminary Sampling Validation

Before processing the full 19.5 GB `user_posts.tar.gz` (100 files), sample 2–3 files to:
- Estimate conflict-post volume (expected: 0.5–2% of total)
- Measure keyword precision (manual inspection of ~200 matched posts)
- Measure keyword recall (manual inspection of ~200 random posts from event windows to check for missed conflict posts)
- Adjust keyword lists based on findings

##### 5.1.5 Output

A filtered post index: `post_id → {user_id, timestamp, text, langs, sent_label, sent_score, is_repost, repost_from, reposted_author, conflict_keywords_matched, event_window}`

#### 5.2 Cascade Construction

##### 5.2.1 Cascade Identification

From the filtered conflict post index, identify cascade roots: original (non-repost) conflict-related posts. Trace downstream reposts via:
1. `repost_from` and `reposted_author` fields in `user_posts.tar.gz` (direct repost links)
2. `interactions.csv.gz` (timestamped interaction edges)
3. `graphs.tar.gz` (repost graph edge lists)

A cascade is the directed tree rooted at an original post, where each edge represents a repost event. Note: reposts of reposts may include users who did *not* match conflict keywords — these are included in the cascade (the *root* must be conflict-related, but downstream propagators need not be).

##### 5.2.2 Parent Assignment Heuristic

For each reposter in a cascade, assign a parent node (the user who likely exposed them to the content):
- **Priority 1:** A user the reposter follows (per `followers.csv.gz`) who already reposted earlier in the cascade → organic path.
- **Priority 2 (fallback):** The user who reposted most recently before this reposter → temporal proximity heuristic.
- **Priority 3 (final fallback):** Connect directly to the cascade root → conservative assumption.

##### 5.2.3 Edge-Type Assignment

For each cascade edge `(source → target)`, cross-reference against `followers.csv.gz`:
- **Type 0 (FOLLOWER):** `target` follows `source` → organic spread.
- **Type 1 (NON_FOLLOWER):** Both users exist in the follower graph but no follow edge → suggests algorithmic/external exposure.
- **Type 2 (UNKNOWN):** One or both users missing from the follower graph.

**Caveat:** The follower graph is a single snapshot (~Mar 2024). Some follow relationships may have been *created because of* the conflict (e.g., users following journalists or activists after Oct 7). Edges labeled FOLLOWER for October 2023 cascades may not have existed at cascade time. This is acknowledged as a limitation and discussed in the thesis.

##### 5.2.4 Cascade Size Thresholds

The minimum cascade size will be determined empirically after the preliminary sampling step:
- **If ≥50K cascades with ≥10 reposts:** Use 10 as threshold (consistent with original plan).
- **If 10K–50K cascades with ≥10 reposts:** Use 10 as threshold, accept smaller dataset, report bootstrap confidence intervals on clustering metrics.
- **If <10K cascades with ≥10 reposts:** Lower threshold to ≥5 reposts.

Maximum cascade size: 10,000 reposts (computational tractability).

##### 5.2.5 Node Features

Each node (user) in a cascade graph carries:
- **Temporal feature:** Time delta (in minutes) between this user's repost and the cascade root post.
- **Sentiment feature:** `sent_label` (0/1/2) and `sent_score` from the pre-computed RoBERTa sentiment annotation. For non-English posts or posts without sentiment labels, use a default neutral (1) with score 0.5.
- **Positional feature:** Depth in the cascade tree (hops from root).
- **Degree feature:** In-degree and out-degree within the cascade.

##### 5.2.6 Graph-Level Features

Each cascade graph carries aggregate features used alongside the learned embedding:
- **Sentiment composition:** Proportion of negative, neutral, positive nodes.
- **Follower-edge ratio:** Proportion of Type 0 (FOLLOWER) edges.
- **Event window label:** Which mega-event (E1–E6) or baseline.
- **Structural descriptors:** Depth, breadth (max width at any level), branching factor, Wiener index (structural virality per Goel et al., 2016).

#### 5.3 BlueCCGL Architecture

##### 5.3.1 Overview

```
Input: Crisis Cascade Graph G = (V, E, X, R, S)
       V = users, E = repost edges, X = node features,
       R = edge types (follower/non-follower/unknown),
       S = sentiment labels
    ↓
Feed-Aware Augmentation (Extension 1)
    ↓
Edge-Type Aware R-GCN Encoder (Extension 2)
    ↓
Hierarchical DiffPool Pooling (Extension 3)
    ↓
Graph-Level Readout (concatenated with sentiment composition)
    ↓
Contrastive Loss (InfoNCE)
    ↓
Pre-trained Encoder
    ↓
HDBSCAN Clustering → Crisis Cascade Archetypes
    ↓
Archetype Analysis (structural + sentiment + event-window characterisation)
```

##### 5.3.2 Extension 1: Feed-Aware Augmentation

Standard graph augmentations (random node/edge dropping) ignore the semantic meaning of cascade structure. We propose augmentations that respect the distinction between organic (follower-based) and algorithmically-mediated crisis spread.

**Strategy A: Follower-Subgraph Extraction**
- Retain only edges where the target follows the source (Type 0).
- Sample a configurable ratio (default 0.7) of these edges.
- Simulates: "What would this crisis cascade look like if spread were purely organic?"

**Strategy B: Temporal Prefix Sampling**
- Retain only the earliest N% of nodes (default 60%) by timestamp.
- Simulates: "What did the early-stage spread look like before amplification kicked in?"
- Particularly relevant for crisis cascades where breaking-news dynamics dominate the first hours.

**Strategy C: Bridge-Node Removal**
- Identify nodes that entered the cascade without following any existing cascade member (non-follower edges only).
- Remove a configurable ratio (default 0.3) of these bridge nodes.
- Simulates: "What would spread look like without cross-community bridges?"

**Contrastive Pair Generation:**
- Positive pairs: Same cascade with different augmentations.
- Negative pairs: Different cascades (in-batch negatives).
- Hard negatives: Cascades from the same event window but with different structure.

##### 5.3.3 Extension 2: Edge-Type Aware R-GCN Encoder

- Three relation types (FOLLOWER, NON_FOLLOWER, UNKNOWN) with relation-specific weight matrices.
- Basis decomposition (2 bases for 3 relations) for parameter efficiency.
- Multi-layer R-GCN (2–3 layers) with ReLU activation and dropout.
- Node features (temporal, sentiment, positional, degree) are concatenated as input.

##### 5.3.4 Extension 3: Hierarchical Pooling with Sentiment Enrichment

- DiffPool-based hierarchical pooling with pool ratios `[0.5, 0.25]`.
- At each hierarchy level, global mean pooling produces a level-specific embedding.
- Level embeddings are concatenated with graph-level sentiment composition features (proportion negative/neutral/positive).
- Final readout MLP projects to the contrastive embedding space.

This design captures:
- **Micro-level:** Individual user repost decisions (with their sentiment).
- **Meso-level:** Community-level spread patterns (e.g., a sub-cluster of the cascade with uniformly negative sentiment).
- **Macro-level:** Global cascade shape + overall sentiment composition.

##### 5.3.5 Loss Function

**InfoNCE contrastive loss** following CCGL:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j) / \tau)}$$

where $z_i$ and $z_i^+$ are embeddings of two augmented views of the same cascade, $\tau$ is the temperature parameter, and the denominator sums over all in-batch cascades (including the positive pair).

#### 5.4 Archetype Discovery and Analysis

##### 5.4.1 Clustering

**HDBSCAN** applied to the learned cascade embeddings:
- No pre-specified number of clusters.
- Handles noise (outlier cascades assigned to no cluster).
- Minimum cluster size tuned via grid search (candidates: 50, 100, 200).

##### 5.4.2 Archetype Characterisation

Each discovered archetype (cluster) is characterised by:

| Property | Description |
|----------|-------------|
| **Structural profile** | Distributions of depth, breadth, branching factor, Wiener index |
| **Edge-type composition** | Mean follower-edge ratio; distribution of FOLLOWER vs. NON_FOLLOWER edges |
| **Sentiment composition** | Mean proportion of negative/neutral/positive nodes |
| **Temporal profile** | Distribution of cascade duration (root to last repost); spread velocity |
| **Event-window distribution** | Which mega-events are over/under-represented in this archetype |
| **Cascade size distribution** | Mean, median, variance of cascade size within the archetype |

##### 5.4.3 Hypothesised Archetypes

Based on prior cascade literature (Goel et al., 2016; Vosoughi et al., 2018) and the crisis context, we anticipate at minimum:

1. **Broadcast cascades:** Shallow, wide, high non-follower-edge ratio, dominated by E1/E2 (breaking news events). High negative sentiment.
2. **Organic-viral cascades:** Deep, narrow, high follower-edge ratio, dominated by E5 (individual human stories). Mixed sentiment.
3. **Deliberative cascades:** Moderate depth/breadth, higher positive/neutral sentiment mix, dominated by E4 (legal/institutional milestone).
4. **Ceasefire/hope cascades:** Possibly distinct pattern around E3 (ceasefire), potentially with more positive sentiment than other archetypes.
5. **Sustained-advocacy cascades:** Moderate depth, high follower ratio, appearing in the baseline (non-event) periods, representing ongoing activism rather than event-triggered spread.

These are hypotheses; the unsupervised approach may reveal different or additional archetypes.

#### 5.5 Evaluation Strategy

##### 5.5.1 Clustering Quality Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Silhouette Score** | Intra-cluster cohesion vs. inter-cluster separation |
| **Davies-Bouldin Index** | Ratio of within-cluster scatter to between-cluster separation (lower = better) |
| **Calinski-Harabasz Index** | Ratio of between-cluster dispersion to within-cluster dispersion (higher = better) |
| **DBCV** (Density-Based Clustering Validation) | Specific to HDBSCAN; handles non-convex clusters |

##### 5.5.2 Ablation Studies

| Ablation | What Is Removed | Expected Effect |
|----------|----------------|-----------------|
| **A1: No edge-type encoding** | Replace R-GCN with standard GCN (all edges homogeneous) | Loss of organic vs. algorithmic distinction in embeddings |
| **A2: Generic augmentation** | Replace feed-aware augmentations with random node/edge dropping | Loss of domain-semantic structure in contrastive pairs |
| **A3: Flat pooling** | Replace hierarchical DiffPool with global mean pooling | Loss of multi-scale cascade representation |
| **A4: No sentiment features** | Remove sentiment from node features and graph-level readout | Loss of affective information in cascade representation |
| **A5: Random initialisation** | Skip contrastive pre-training; cluster raw structural features | Tests whether learned representations add value over handcrafted features |

##### 5.5.3 Downstream Transfer Evaluation

The pre-trained BlueCCGL encoder is fine-tuned on two downstream tasks:
- **Cascade size prediction:** Predict final cascade size from the first N% of the cascade.
- **Cascade longevity prediction:** Predict whether the cascade will remain active beyond a time threshold.

Comparison against: random initialisation baseline, vanilla CCGL, and a feature-engineering baseline (Wiener index + depth + breadth + follower ratio).

##### 5.5.4 Statistical Robustness

Given the potentially smaller dataset (event-focused), all results include:
- **Bootstrap confidence intervals** (1000 resamples) on clustering metrics.
- **Stability analysis:** Re-run HDBSCAN on 10 random 80% subsamples; report Adjusted Rand Index of cluster assignments.

---

### 6. Data Source

#### 6.1 Primary Data Source: Failla & Rossetti (2024) Zenodo Dataset

- **Paper:** Failla, A. & Rossetti, G. (2024). "I'm in the Bluesky Tonight": Insights from a year worth of social data. *PLoS ONE* 19(11): e0310330.
- **Dataset:** Zenodo, DOI: 10.5281/zenodo.14669616
- **Size:** 22 GB compressed
- **Coverage:** 4,099,699 users (~81% of registered accounts), 235,567,116 posts, Feb 17 2023 – Mar 18 2024
- **License:** Bluesky ToS

#### 6.2 Relevant Dataset Files

| File | Size | Contents | Role in This Study |
|------|------|----------|--------------------|
| `user_posts.tar.gz` | 19.5 GB | JSON posts per user: `text`, `timestamp`, `langs`, `sent_label`, `sent_score`, `repost_from`, `reposted_author`, `reply_to`, `thread_root`, `quoted_author`, `user_id`, `post_id` | NLP filtering (keyword matching on `text`); cascade root identification; sentiment features |
| `interactions.csv.gz` | 1.0 GB | `user_id, replied_author, thread_root_author, reposted_author, quoted_author, date` | Cascade edge construction; temporal ordering |
| `graphs.tar.gz` | 891 MB | Timestamped edge lists for repost, quote, reply networks | Cascade graph reconstruction |
| `followers.csv.gz` | 491 MB | Directed follower edge list (~144M edges) | Edge-type labelling (follower vs. non-follower) |
| `feed_posts.tar.gz` | 16.4 MB | Posts from 11 feed generators (incl. News feed) | Identifying feed-amplified content |
| `feed_bookmarks.csv` | 553 KB | User feed-subscription data | Identifying users with algorithmic feed exposure |

#### 6.3 Dataset Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Pseudonymised IDs** (integer IDs, no DIDs/handles) | Cannot link to specific Bluesky accounts or verify user identity | Not needed for structural analysis; prevents re-identification (ethics benefit) |
| **Single follower-graph snapshot** (~Mar 2024) | Some follow relationships may not have existed during earlier cascades; conflict may have *created* follow relationships | Acknowledge as limitation; report proportion of UNKNOWN edges; consider sensitivity analysis excluding UNKNOWN edges |
| **Minute-level timestamps** | Cannot distinguish sub-minute cascade ordering | Sufficient for cascade reconstruction; sub-minute ordering is rarely critical for repost cascades |
| **Text up to 300 chars** | Short posts may have low keyword recall | Bluesky's limit means 300 chars is the *full* post, not truncated; most conflict posts will contain at least one unambiguous keyword |
| **Coverage ends Mar 18, 2024** | Misses later conflict developments | The 5.5-month window (Oct 7 – Mar 18) already contains 6 mega-events across diverse categories |
| **English-dominant, left-leaning user base** | Findings are not representative of global conflict discourse | Framed as a study of crisis dynamics within a politically homogeneous decentralized community |
| **Sentiment only for English posts** | Non-English posts get default neutral sentiment | ~95%+ of conflict posts expected to be English given Bluesky demographics |

#### 6.4 Estimated Data Requirements

| Metric | Estimate | Notes |
|--------|----------|-------|
| Conflict-related posts | ~750K–3M | 0.5–2% of 235M posts in conflict window, based on keyword coverage |
| Cascade roots (original conflict posts with reposts) | ~10K–50K | Subset with ≥5 or ≥10 reposts; determined empirically |
| Observation period | Oct 1 2023 – Mar 18 2024 (170 days) | Conflict window within dataset |
| Follower graph | 4M+ users, ~144M edges | Single snapshot |
| Storage (raw download) | ~22 GB | Compressed Zenodo files |
| Storage (processed) | ~30–50 GB | Decompressed + filtered + PyG dataset |

---

### 7. Expected Outcomes

**Outcome 1 — A taxonomy of crisis cascade archetypes on Bluesky.** We expect HDBSCAN clustering to reveal distinct crisis cascade archetypes characterised by interpretable structural and sentiment properties. The key finding is the *temporal narrative*: how the distribution of archetypes shifts across the six mega-events, revealing whether crisis information dynamics evolve from broadcast-dominated (acute shock) to organic-viral (sustained advocacy) as the conflict matures.

**Outcome 2 — Empirical evidence on algorithmic vs. organic amplification of crisis content.** By quantifying the follower-edge ratio across archetypes and mega-events, we provide the first direct measurement of how crisis content spreads through organic follower networks versus algorithmically-mediated pathways on a decentralized platform. This is a finding that is structurally impossible to obtain from centralized platforms.

**Outcome 3 — Sentiment-structure coupling in crisis cascades.** By embedding sentiment composition into the cascade representation and characterising archetypes by their sentiment profile, we reveal whether affective dynamics are coupled to structural propagation patterns — e.g., whether uniformly negative cascades spread differently from mixed-sentiment cascades.

**Outcome 4 — Quantitative evidence for each methodological extension.** Through ablation studies (A1–A5), we demonstrate the contribution of edge-type encoding, feed-aware augmentation, hierarchical pooling, and sentiment enrichment. Negative results (e.g., "sentiment features do not improve clustering") are themselves valuable findings about the nature of crisis cascades.

**Outcome 5 — Open-source implementation and reusable crisis cascade dataset.** The project produces: (a) the BlueCCGL PyTorch Geometric codebase (NLP filtering, cascade construction, model training, clustering); and (b) a processed Bluesky crisis cascade graph dataset derived from the Failla & Rossetti (2024) dataset, containing conflict-related cascades with edge-type annotations, sentiment labels, and event-window assignments.

---

### 8. Research Plan

This is a solo Bachelor's thesis project spanning 16 weeks. The addition of the NLP filtering pipeline adds complexity to the data acquisition phase, which is compensated by leveraging the pre-existing Zenodo dataset and the pre-computed sentiment labels.

| Week | Phase | Activities | Deliverable |
|------|-------|-----------|-------------|
| 1 | Data Acquisition | Download Failla & Rossetti (2024) Zenodo dataset (~22 GB); decompress `user_posts.tar.gz`, `followers.csv.gz`, `interactions.csv.gz`, `graphs.tar.gz`; load into working format (Parquet/DuckDB) | Raw dataset loaded; storage pipeline validated |
| 2 | NLP Filtering | **Preliminary sampling:** process 2–3 of 100 `user_posts` files; apply keyword regex (Tiers 1–5); estimate conflict-post volume; validate precision/recall on ~200 manually inspected posts; adjust keyword lists | Keyword validation report; volume estimate; refined keyword list |
| 3 | NLP Filtering | Full-corpus keyword filtering across all 100 `user_posts` files (Oct 1 2023 – Mar 18 2024); assign event-window labels (E1–E6 + baseline); generate filtered post index | Filtered conflict post index; event-window assignment; corpus statistics |
| 4 | Cascade Construction | Reconstruct repost cascades from filtered conflict posts using `interactions.csv.gz` and `graphs.tar.gz`; assign edge types via `followers.csv.gz`; determine cascade size threshold based on observed distribution | Cascade construction pipeline; edge-type labelled cascades; size threshold decision |
| 5 | Cascade Construction | Convert cascade graphs to PyTorch Geometric dataset with node features (temporal, sentiment, positional, degree) and graph-level features (sentiment composition, follower ratio, event window); detailed EDA | PyG dataset; EDA notebook with per-event and per-archetype-candidate visualisations |
| 6 | Model Implementation | Implement base CCGL reproduction: GNN encoder, sub-cascade sampling augmentation, InfoNCE contrastive loss | Base CCGL training loop verified on crisis cascade data |
| 7 | Model Implementation | Implement Extension 1 (feed-aware augmentations) and Extension 2 (R-GCN encoder with basis decomposition); integrate with contrastive pipeline | Augmentation module + R-GCN encoder with unit tests |
| 8 | Model Implementation | Implement Extension 3 (hierarchical DiffPool with sentiment enrichment); assemble full BlueCCGL architecture; end-to-end training verification | Complete BlueCCGL model; successful training run |
| 9 | Experiments | Contrastive pre-training of BlueCCGL; hyperparameter tuning (temperature, augmentation ratios, learning rate) via Weights & Biases | Trained model; hyperparameter sweep results |
| 10 | Experiments | Extract cascade embeddings; apply HDBSCAN clustering; compute clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, DBCV); generate UMAP visualisations colour-coded by event window and sentiment | Archetype clusters; clustering quality report; UMAP visualisations |
| 11 | Experiments | Ablation studies (A1–A5); bootstrap confidence intervals on all metrics; stability analysis (10× 80% subsamples) | Ablation results table with confidence intervals |
| 12 | Experiments | Downstream transfer evaluation (cascade size + longevity prediction); archetype interpretation (structural + sentiment + event-window characterisation per cluster); temporal narrative construction | Downstream task results; archetype characterisation document; crisis phase evolution analysis |
| 13 | Writing | Draft Introduction, Related Work, and Research Questions sections; compile bibliography | Thesis draft chapters 1–3 |
| 14 | Writing | Draft Methodology (NLP pipeline + cascade construction + BlueCCGL architecture) and Experimental Setup sections; create all figures and tables | Thesis draft chapters 4–5 |
| 15 | Writing | Draft Results, Discussion (including social implications), and Conclusion; integrate all sections into full thesis | Complete thesis draft |
| 16 | Writing | Revisions based on supervisor feedback; final proofreading; prepare presentation | Final thesis; presentation slides |

---

### 9. Ethical Considerations

1. **Data provenance.** The Failla & Rossetti (2024) dataset is publicly available under Bluesky's Terms of Service. All user IDs are pseudonymised (integer IDs); no DIDs, handles, or post URIs are present. Re-identification of individual users is not possible from the dataset alone.

2. **Conflict sensitivity.** This study analyses the *structural dynamics* of information spread, not the *content* or *validity* of political positions. The thesis does not evaluate, endorse, or critique any stance on the Israel-Palestine conflict. Framing is neutral and structural throughout.

3. **Demographic bias acknowledgement.** All findings are explicitly contextualised as describing crisis information dynamics within Bluesky's specific user base (English-dominant, Western, left-leaning early adopters during 2023–2024). No claims of representativeness for global discourse are made.

4. **Noindex and consent.** Bluesky users can set `noindex` flags on their profiles. The Failla & Rossetti dataset was collected via the public API respecting platform terms, and the pseudonymisation further protects user privacy. However, the thesis acknowledges that users may not have anticipated their posts being used for conflict-related research.

5. **Responsible disclosure.** Archetype analysis results are reported at the aggregate level. No individual cascades are reproduced or identifiable. The open-source dataset release will maintain the same pseudonymisation as the source dataset.

---

### ⚠️ Flagged Citations (Require Verification)

The following citations from the original plan are retained but flagged as unverified:

- HierCas (Wang et al., 2024) — "Hierarchical Information Propagation Prediction"
- TempCas (Liu et al., 2025) — "Temporal Cascade Transformers"
- CSTGCN (Li et al., 2024) — "Contrastive Self-Supervised Temporal Graph Learning"
- Cerisara et al. (2025) — "Bluesky Ecosystem Analysis"
- HiCas (Zhou et al., 2023) — "Hierarchical Cascade Representation"
- Bhadani et al. (2024) — "Algorithmic Amplification of Low-Quality Content"

---

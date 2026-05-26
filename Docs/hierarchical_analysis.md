# Hierarchical Cascade Archetype Analysis

## Overview

Having identified 11 fine-grained cascade archetypes via HDBSCAN, we apply Ward agglomerative
clustering to the cluster centroids in the 10-dimensional UMAP embedding space. This reveals a
**two-level taxonomy**: five macro-archetypes that group the 11 clusters by their dominant
organising principle, with structural sub-variation visible within each group.

The central finding is that **sentiment polarity organises cascade archetypes at the macro level,
while propagation structure — depth, breadth, and follower edge ratio — sub-divides them at the
fine-grained level.** This result has direct grounding in the information diffusion literature,
which has long distinguished between emotional and structural drivers of online spreading behaviour.

---

## The Two-Level Taxonomy

![Dendrogram + Heatmap](../Outputs_17_04/hierarchical/figures/dendrogram_heatmap.png)

*Figure 1. Ward dendrogram of the 11 HDBSCAN cluster centroids (left), with each leaf labelled by
its macro assignment [M0–M4]. The heatmap (right) shows rank-normalised medians (0 = lowest across
clusters, 1 = highest) for seven structural and sentiment properties. The dashed red line marks the
natural cut at Ward distance 10.05, yielding five macro-archetypes.*

The dendrogram immediately surfaces the most fundamental division in the taxonomy: **Macro 0
(C0, C1) merges with all other clusters only at the very top of the tree (Ward distance ≈ 20)**,
making it the most structurally isolated group. The heatmap explains why — these are the only
clusters scoring at the maximum on the Neg. column and zero across Neu. and Pos., marking them as
the dataset's sole reservoir of **pure-negative sentiment cascades**.

Reading further down the tree, a secondary split at distance ≈ 17 separates the large
mixed-sentiment cascades (Macro 2, C2/C3) from the neutral-dominant groups (Macros 3/4).
Within each macro, remaining variation is structural: depth, breadth, and follower ratio
differentiate the fine-grained clusters once sentiment is held roughly constant.

---

## Macro-Archetype Profiles

### Sentiment as the Primary Discriminator

![Sentiment Composition](../Outputs_17_04/hierarchical/figures/macro_sentiment.png)

*Figure 2. Median sentiment composition per macro-archetype. Macro 0 and Macro 3 are sentiment-pure
(100% negative and 100% neutral respectively). Macro 2 is the only group containing meaningful
positive sentiment (≈11%), marking it as a contested or counter-narrative spreading pattern.*

The stacked bar confirms the macro-level story cleanly. Three of the five macro-archetypes are
sentiment-dominant:

- **Macro 0** — entirely negative; no neutral or positive content at the median cascade level
- **Macro 3** — entirely neutral; the purest follower-driven informational cascade regime
- **Macro 2** — the only group with positive sentiment; uniquely mixed, suggesting the presence of
  counter-narrative or solidarity content alongside dominant negative framing

Macros 1 and 4 occupy intermediate positions — mixed negative-neutral and predominantly neutral
respectively — forming a gradient between the two extremes.

---

### Structural Profiles (Median + IQR)

![Structural Profiles](../Outputs_17_04/hierarchical/figures/macro_profiles.png)

*Figure 3. Median structural profiles per macro-archetype with interquartile range error bars.
Cascade size shows the clearest macro-level gradient. Follower edge ratio is relatively stable
across macros (0.76–0.88), confirming that structural diversity within the dataset is primarily
expressed through depth, breadth, and size rather than follower composition.*

The structural profiles reveal a clear **size gradient**: Macro 0 cascades are the smallest
(median size 7), while Macro 2 contains the largest (median size 15, driven by C3 at 16). The
follower edge ratio, by contrast, is largely stable — all macros sit between 0.76 and 0.88 —
which is consistent with the overall 75.9% FOLLOWER-typed edge rate in the dataset. This suggests
that follower structure is a baseline property of Bluesky cascade propagation rather than a
macro-level discriminator.

---

### Structural Positioning of Individual Clusters

![Bubble Chart](../Outputs_17_04/hierarchical/figures/bubble_chart.png)

*Figure 4. Structural positioning of all 11 fine-grained clusters. x-axis: follower edge ratio
(median); y-axis: cascade depth (median); bubble size: cascade size (median). Colour encodes
macro-archetype membership.*

Two clusters stand out as structural outliers:

- **C3** (green, Macro 2) — the largest bubble in the chart, confirming its role as the dominant
  large-cascade archetype. It is also the deepest cascade in Macro 2.
- **C8** (orange, Macro 1) — the leftmost point, with the lowest follower ratio across all 11
  clusters (0.52) and the greatest depth (5). This cluster represents **out-of-network diffusion
  via weak ties** — the only archetype in the dataset where the majority of cascade spread occurs
  across non-follower ties.

The congestion of most other clusters around follower ratio 0.65–0.85 and depth 3–4 reinforces
the finding from Figure 3: follower ratio and depth alone do not cleanly separate the majority of
archetypes. It is the GNN embeddings, which integrate edge-type, structural, and sentiment
information jointly, that produce the separable clusters — the bubble chart inadvertently
demonstrates the value of the learned representation.

---

### Spatial Distribution in Embedding Space

![UMAP](../Outputs_17_04/hierarchical/figures/umap_macro.png)

*Figure 5. Two-level structure visualised in 2D UMAP embedding space. Left panel: macro-archetype
membership. Right panel: fine-grained HDBSCAN cluster labels. Macro 1 (orange) occupies a broad
central region; Macro 2 (green) forms a compact distinct blob; Macro 0 (blue) appears as a small
isolated scatter — consistent with its structural isolation in the dendrogram.*

---

### Cascade Distribution Across Macro-Archetypes

![Counts](../Outputs_17_04/hierarchical/figures/macro_counts.png)

*Figure 6. Cascade counts per macro-archetype. The distribution is inherently imbalanced: Macros 1
and 2 together account for 82% of all cascades, reflecting that the majority of conflict-related
spreading on Bluesky follows common baseline patterns. The smaller macro-archetypes (0, 3, 4)
represent structurally distinctive minority modes.*

The count distribution is itself a finding. It is consistent with the broader diffusion literature:
Goel et al. (2016) showed that across 1 billion Twitter diffusion events, the vast majority of
cascades are structurally simple and small, with structurally complex viral diffusion being the
rare exception. Our Macro 0 (n=1,025) and Macro 3 (n=2,509) represent precisely these rare but
qualitatively distinct modes, while Macros 1 and 2 represent the dominant baseline of conflict
discourse propagation.

---

## The Five Macro-Archetypes: Descriptions and Literary Grounding

### Macro 0 — Pure-Negative Micro-Cascades (C0, C1 · n=1,025)

**Profile:** Smallest cascades in the dataset (median size 7), 100% negative sentiment at the
median cascade level, moderate follower ratio (0.76). C0 sub-variant is deeper and narrower
(depth=4, breadth=2, follower=0.60); C1 sub-variant is shallower and wider (depth=3, breadth=4,
follower=0.83).

**Fine-grained clusters:**

| Cluster | Size | Depth | Breadth | Follower ratio | Sentiment | Character |
|---------|------|-------|---------|----------------|-----------|-----------|
| C0 | 6 | 4 | 2 | 0.60 | 100% neg | Deep narrow chain; lower follower ratio suggests non-follower bridging even within this negative regime |
| C1 | 7 | 3 | 4 | 0.83 | 100% neg | Shallower and wider; high follower ratio indicates fan-out within an established follower community |

The within-macro distinction maps directly onto Goel et al.'s (2016) broadcast vs. viral axis:
C0 is the viral mode (narrow chain, each node infecting few others), C1 is the broadcast mode
(wide fan-out from a central node). Both are emotionally identical — purely negative — but
structurally opposite propagation mechanisms.

**Interpretation:** These cascades represent intense, emotionally concentrated bursts of negative
content that remain small. Their isolation at the top of the dendrogram suggests they are
qualitatively different from all other cascade types — not simply small versions of the mixed
discourse cascades in Macro 1, but a structurally distinct propagation regime.

**Literary grounding:** Brady et al. (2017) demonstrated that each additional moral-emotional word
in social media posts is associated with a 24% increase in retweet rate within politically
homogeneous networks — but critically, this amplification is *network-local*. Moral outrage
content circulates intensely within in-group communities rather than spreading broadly across
the social graph. This matches the profile of Macro 0: high emotional intensity, constrained reach.
The C0 sub-variant (depth=4, breadth=2, follower=0.60) further aligns with Goel et al.'s (2016)
"viral" diffusion mode — a narrow chain where each node infects few others — while C1 (breadth=4,
follower=0.83) resembles a "broadcast" event within a follower community.

Stieglitz & Dang-Xuan (2013) additionally found that negative sentiment messages spread 25% faster
than neutral ones but do not necessarily achieve larger audience reach — consistent with the micro
scale of Macro 0 despite its emotional intensity.

---

### Macro 1 — Mixed Negative Discourse (C7, C8, C9, C10 · n=14,682)

**Profile:** The largest macro-archetype by cascade count. Medium cascades (median size 7–9),
predominantly negative (65%) with a substantial neutral component (33%). Follower ratio moderate
(0.67–0.83). C8 is the structural outlier: depth=5, breadth=2, follower=0.52 — the only deep
chain with majority non-follower propagation in the dataset.

**Fine-grained clusters:**

| Cluster | Size | Depth | Breadth | Follower ratio | Sentiment | Character |
|---------|------|-------|---------|----------------|-----------|-----------|
| C7 | 9 | 3 | 4 | 0.83 | 67% neg / 33% neu | Balanced shape; moderate follower ratio; the most "typical" conflict cascade |
| C8 | 8 | 5 | 2 | 0.52 | 78% neg / 22% neu | **Structural outlier**: deepest chain, lowest follower ratio in the dataset — out-of-network diffusion via weak ties |
| C9 | 8 | 4 | 3 | 0.70 | 83% neg / 17% neu | Moderate depth and breadth; higher negative fraction than C7 |
| C10 | 7 | 4 | 2 | 0.67 | 50% neg / 50% neu | Smallest and narrowest; most balanced sentiment split in this macro |

C8 is the most structurally distinctive cluster in the entire dataset. Its follower ratio of 0.52
means the majority of its edges cross non-follower ties — an unusual pattern consistent with
out-of-network diffusion: content reaching users outside the poster's immediate follower community
through structural bridges rather than within established networks.

**Interpretation:** This is the dominant "everyday conflict discourse" regime — the baseline
pattern through which most conflict-related information about the Gaza war, ceasefire negotiations,
and protest events propagated on Bluesky. C8's structural distinctiveness (deep chain, low follower
ratio) identifies a sub-archetype of information spreading through non-follower bridges, consistent
with Granovetter's (1973) "strength of weak ties" — content traversing structurally distant users
rather than tight follower communities.

**Literary grounding:** The prevalence of Macro 1 (37% of all cascades) aligns with the finding
of Ferrara & Yang (2015) that negative-sentiment messages achieve the highest reposting volume
(20–60% above neutral), explaining their dominance in a conflict-adjacent dataset. The neutral
component (33%) likely captures factual reporting and contextual commentary co-propagating
alongside emotional reactions — consistent with Lerman & Ghosh (2010) who found that news-adjacent
content consistently accompanies emotional content in crisis diffusion events.

The deep-chain C8 sub-archetype is directly predicted by Goel et al. (2016)'s structural virality
framework: while most diffusion is broadcast-dominated, a minority of events grow through sustained
multi-generational chains. C8 represents exactly this mode within the conflict discourse regime.

---

### Macro 2 — Large Mixed-Sentiment Cascades (C2, C3 · n=17,619)

**Profile:** The largest cascades in the dataset (C3: median size 16, depth 5, breadth 6). The
only macro-archetype with meaningful positive sentiment (median 11%). C2 is a smaller sub-variant
(size 7) that shares C3's distinctive positive sentiment signal (17%). Follower ratio similar to
Macro 1 (0.78–0.80).

**Fine-grained clusters:**

| Cluster | Size | Depth | Breadth | Follower ratio | Sentiment | Character |
|---------|------|-------|---------|----------------|-----------|-----------|
| C2 | 7 | 3 | 3 | 0.80 | 83% neg / 17% pos | Small but carries positive sentiment; likely counter-narrative or solidarity content embedded in negative discourse |
| C3 | 16 | 5 | 6 | 0.78 | 63% neg / 25% neu / 10% pos | Largest and most structurally complex cascade archetype; highest internal heterogeneity (size std=68) |

C2's small size (7) combined with positive sentiment is notable — it suggests short but
emotionally mixed cascades, possibly early-stage counter-narrative bursts that do not achieve
sustained spread. C3 by contrast is the dataset's most viral archetype: broad, deep, and
structurally complex, representing sustained multi-generational spreading.

**Interpretation:** The positive sentiment component is the defining feature of this group — and
in a conflict dataset, positive sentiment likely does not indicate positive *framing* of the
conflict, but rather counter-narratives, solidarity expressions, humanitarian appeals, or
ironic/satirical content. C3's large size and structural complexity (high depth and breadth) marks
it as the most "virally" spreading archetype in the dataset, closest to what Goel et al. (2016)
call structurally viral diffusion events.

The heterogeneity within Macro 2 is high (cascade size std=68 for C3), which reflects the
catch-all nature of the group: it captures all cascades large enough and mixed enough in sentiment
to occupy a distinct region of the embedding space, without a single uniform propagation mechanism.

**Literary grounding:** Vosoughi et al. (2018) showed that false and contested news spreads
deeper, faster, and more broadly than true news on Twitter — attributing this in part to the
novelty and emotional complexity of contested content. Macro 2 cascades are candidates for
precisely this dynamic: their structural size and the presence of positive sentiment alongside
dominant negative framing suggests contested or counter-narrative content that attracts engagement
from multiple communities across the repost network.

The convergence behaviour literature (Starbird & Palen 2010; Stab et al. 2019) identifies
"helpers" and "counter-narrative" actors as distinct convergence roles in crisis communication on
social media — both of which could generate the mixed-sentiment, large-cascade structure observed
in Macro 2.

---

### Macro 3 — Follower-Driven Neutral Cascades (C4, C5 · n=2,509)

**Profile:** 100% neutral sentiment at the median cascade level. The highest follower edge ratio
in the dataset (C5 median: 0.909 — the single highest across all 11 clusters). Small cascades
(size 7–9), moderate depth (3–4), moderate breadth (3–4).

**Fine-grained clusters:**

| Cluster | Size | Depth | Breadth | Follower ratio | Sentiment | Character |
|---------|------|-------|---------|----------------|-----------|-----------|
| C4 | 7 | 4 | 3 | 0.80 | 100% neu | Slightly deeper and narrower; more chain-like neutral spread |
| C5 | 9 | 3 | 4 | 0.91 | 100% neu | **Highest follower ratio in the dataset**; wider and shallower — broadcast-style within a tight follower community |

The two clusters are structural mirrors of C0/C1 in Macro 0 — the same broadcast vs. chain
distinction, but in the neutral sentiment regime. C5's 0.91 follower ratio is the most extreme
follower-dominance in the entire dataset.

**Interpretation:** These cascades spread almost exclusively through follower networks with purely
neutral content. In a conflict dataset, this likely corresponds to **factual news diffusion and
institutional reporting** — content that is informational rather than reactive, circulating within
established follower communities without bridging to the broader public. C5, with its 0.909
follower ratio, represents the purest expression of within-community information circulation in
the dataset.

**Literary grounding:** Cinelli et al. (2021) demonstrated in a landmark PNAS study that
information on social media preferentially circulates within homophilic communities — users are
far more likely to engage with and reshare content from accounts they already follow, producing
tight intra-community cascades that do not bridge to other communities. Macro 3 is the empirical
manifestation of this dynamic in Bluesky conflict data: follower-dominant, sentiment-neutral,
structurally contained.

This also connects to echo chamber formation research (Bail et al. 2018; Cinelli et al. 2021):
high follower-ratio cascades represent information circulating within ideologically or
interest-homogeneous communities, reinforcing existing perspectives without cross-community
exposure.

---

### Macro 4 — Neutral Follower Spreading (C6 · n=3,554)

**Profile:** Singleton macro-archetype. Predominantly neutral (90%), high follower ratio (0.875),
medium-large cascades (median size 13, breadth 6). Structurally intermediate between Macro 3
(smaller, purer neutral) and Macro 2 (larger, mixed sentiment).

**Fine-grained clusters:**

| Cluster | Size | Depth | Breadth | Follower ratio | Sentiment | Character |
|---------|------|-------|---------|----------------|-----------|-----------|
| C6 | 13 | 4 | 6 | 0.875 | 90% neu / 10% neg | Singleton; larger and broader than Macro 3; slight negative undercurrent distinguishes it from pure-neutral C4/C5 |

As a singleton, C6 cannot be sub-divided. Its dendrogram position — merging with C4/C5 before
joining C2/C3 — places it as a structural bridge between the follower-driven neutral regime
and the large mixed-sentiment regime. The 10% negative sentiment and notably wider breadth (6 vs
3–4 for Macro 3) are the key differentiators.

**Interpretation:** C6 sits at a structural inflection point in the dendrogram — it merges with
C4/C5 at distance ≈12 and with C2/C3 at ≈14.5, reflecting a genuinely transitional character.
These are **scaled-up neutral follower cascades**: content that spreads primarily through follower
networks but achieves greater reach and structural breadth than the Macro 3 archetypes. This could
represent institutional accounts or verified journalists whose content propagates through
established follower networks but reaches larger audiences due to higher follower counts.

The 10% negative sentiment distinguishes C6 from the pure-neutral Macro 3 — suggesting these
cascades are not entirely factual/neutral in tone but carry a slight negative undercurrent,
potentially corresponding to analytical or critical reporting on conflict events (factual framing
with implicit negative valence).

**Literary grounding:** The distinction between Macro 3 and Macro 4 mirrors the distinction
identified in Bakshy et al. (2012) between "local" information diffusion (tight follower
communities, small reach) and "global" information diffusion (follower networks with high-degree
hubs, broader reach). Macro 4's larger cascade size at comparable follower ratio suggests these
cascades originate from higher-degree nodes — accounts with more followers — producing wider
reach without requiring non-follower bridging.

---

## Summary Table

| Macro | Name | Clusters | n | Dominant sentiment | Structural character | Key reference |
|-------|------|----------|---|-------------------|---------------------|---------------|
| 0 | Pure-negative micro-cascades | C0, C1 | 1,025 | 100% negative | Small, emotionally intense, localised | Brady et al. (2017) |
| 1 | Mixed negative discourse | C7–C10 | 14,682 | 65% neg / 33% neu | Dominant baseline; C8 deep-chain outlier | Goel et al. (2016); Ferrara & Yang (2015) |
| 2 | Large mixed-sentiment cascades | C2, C3 | 17,619 | 64% neg / 11% pos | Largest, most structurally complex | Vosoughi et al. (2018) |
| 3 | Follower-driven neutral cascades | C4, C5 | 2,509 | 100% neutral | Highest follower ratio, intra-community | Cinelli et al. (2021) |
| 4 | Neutral follower spreading | C6 | 3,554 | 90% neutral | Scaled-up neutral, transitional | Bakshy et al. (2012) |

---

## References

- Bakshy, E., Rosenn, I., Marlow, C., & Adamic, L. (2012). The role of social networks in
  information diffusion. *WWW '12*. [https://dl.acm.org/doi/10.1145/2187836.2187907](https://dl.acm.org/doi/10.1145/2187836.2187907)
- Brady, W. J., Wills, J. A., Jost, J. T., Tucker, J. A., & Van Bavel, J. J. (2017). Emotion
  shapes the diffusion of moralized content in social networks. *PNAS*, 114(28), 7313–7318.
  [https://www.researchgate.net/publication/317947723_Emotion_shapes_the_diffusion_of_moralized_content_in_social_networks](https://www.researchgate.net/publication/317947723_Emotion_shapes_the_diffusion_of_moralized_content_in_social_networks)
- Cinelli, M., De Francisci Morales, G., Galeazzi, A., Quattrociocchi, W., & Starnini, M. (2021).
  The echo chamber effect on social media. *PNAS*, 118(9), e2023301118.
  [https://www.pnas.org/doi/10.1073/pnas.2023301118](https://www.pnas.org/doi/10.1073/pnas.2023301118)
- Ferrara, E., & Yang, Z. (2015). Quantifying the effect of sentiment on information diffusion in
  social media. *PeerJ Computer Science*, 1, e26.
  [https://peerj.com/articles/cs-26/](https://peerj.com/articles/cs-26/)
- Goel, S., Anderson, A., Hofman, J., & Watts, D. J. (2016). The structural virality of online
  diffusion. *Management Science*, 62(1), 180–196.
  [https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2158](https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2158)
- Stieglitz, S., & Dang-Xuan, L. (2013). Emotions and information diffusion in social media —
  sentiment of microblogs and sharing behavior. *Journal of Management Information Systems*,
  29(4), 217–248.
  [https://www.researchgate.net/publication/259704042_Emotions_and_Information_Diffusion_in_Social_Media_-_Sentiment_of_Microblogs_and_Sharing_Behavior](https://www.researchgate.net/publication/259704042_Emotions_and_Information_Diffusion_in_Social_Media_-_Sentiment_of_Microblogs_and_Sharing_Behavior)
- Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science*,
  359(6380), 1146–1151.
  [https://www.science.org/doi/10.1126/science.aap9559](https://www.science.org/doi/10.1126/science.aap9559)
- Starbird, K., & Palen, L. (2010). Pass it on? Retweeting in mass emergency. *ISCRAM 2010*.
  [https://www.researchgate.net/publication/228920742_Pass_It_On_Retweeting_in_Mass_Emergency](https://www.researchgate.net/publication/228920742_Pass_It_On_Retweeting_in_Mass_Emergency)

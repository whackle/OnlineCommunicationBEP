# BlueCCGL — Presentation Outline
**5–10 min · Technical DS audience · No prior knowledge of cascades/clustering assumed**

---

## Slide 1 — Title

### BlueCCGL: How Does a Crisis Spread Online?
**Unsupervised Discovery of Information Cascade Archetypes on Bluesky**

*Rayan Mahmoud — Bachelor's Project, April 2026*

---
> **Narration:**
> "Today I want to show you what happens structurally to information when a crisis hits a social platform — who shares it, how far it travels, and whether it crosses community lines. The platform I'm studying is Bluesky, and the crisis is the Israel-Palestine conflict from October 2023 to March 2024. I'll walk you through the data, the model, and what we found."

---

## Slide 2 — Why Bluesky? The Unique Advantage

- **Most platforms hide how content spreads** — you can't tell if someone saw a post because they follow the author, or because an algorithm showed it to them
- **Bluesky makes this observable** — its open follower graph lets us label every single repost as *organic* (follower) or *cross-community* (non-follower)
- **The Israel-Palestine conflict** gave us 6 distinct crisis phases to compare — a natural experiment in how different event types shape information flow

---
> **Bullet 1 narration:**
> "On Twitter or Instagram, you have no idea whether a user saw a post because they follow the author or because the platform's recommendation algorithm surfaced it. That distinction is completely invisible. It means any cascade analysis on those platforms is measuring a mix of organic and algorithmic spread that you can't disentangle."

> **Bullet 2 narration:**
> "Bluesky is built on an open protocol — the follower graph is publicly accessible. That means for every repost in a cascade, we can check: did this person follow the previous reposter? If yes, it's organic spread. If no, some algorithmic or cross-community mechanism was involved. This is architecturally unique and it's the core methodological advantage of this project."

> **Bullet 3 narration:**
> "The Israel-Palestine conflict gave us an unusually rich dataset — six qualitatively different event types in five months: a military shock, a contested humanitarian event, a diplomatic ceasefire, a legal ruling, an individual human story, and a compound humanitarian milestone. That diversity lets us ask: do different kinds of crisis produce structurally different patterns of information spread? Which brings me to the research questions."

---

## Slide 3 — Research Questions

**Socially-grounded questions (SRQ):**

- **SRQ1:** Do cascade dynamics differ across conflict phases — breaking news vs. sustained advocacy?
- **SRQ2:** Do different *types* of crisis event (military shock vs. legal ruling vs. individual story) produce structurally distinct cascade signatures?
- **SRQ3:** Is crisis information spread primarily organic (follower-based) or cross-community (algorithmically exposed)?
- **SRQ4:** Does sentiment composition couple with cascade structure — do negative cascades spread differently?

**Methodological questions (MRQ):**

- **MRQ1:** What natural cascade archetypes emerge from contrastive learning on crisis graphs?
- **MRQ2:** Does edge-type aware encoding (R-GCN) improve representation quality over a homogeneous GCN?
- **MRQ3:** Do feed-aware augmentations outperform generic graph augmentations?

---
> **SRQ narration:**
> "We have four socially-grounded questions. SRQ1 asks whether the structure of how people spread crisis information evolves as a conflict progresses from acute shock to sustained advocacy — does a breaking-news cascade look different from a retrospective analysis cascade? SRQ2 asks whether different event categories — a military attack, a legal ruling, an individual human story — produce structurally distinct cascade patterns. SRQ3 asks how much spread is organic versus cross-community, and whether there are distinct archetype clusters associated with each mode. SRQ4 asks whether what a cascade is *about* emotionally — the sentiment composition — couples with *how* it spreads structurally."

> **MRQ narration:**
> "We also have three methodological questions about the model itself. MRQ1 is the core unsupervised discovery question: what archetypes does the model find? MRQ2 and MRQ3 ask whether our novel extensions — the R-GCN edge-type encoder and the feed-aware augmentations — actually improve the learned representations compared to simpler baselines. Those ablations are pending, but we'll see indirect evidence for both today in the probing results."

---

## Slide 4 — The Data

- **235 million posts**, 4 million users, Feb 2023 – Mar 2024
- NLP filter across **5 keyword tiers** (English, hashtags, Arabic, Hebrew, named entities) → **1.95 million conflict posts** (0.84% of all posts)
- **6 mega-events** labelled as temporal anchors

| ID | Event | Window |
|----|-------|--------|
| E1 | Hamas attack / Israel declares war | Oct 4–10 |
| E2 | Al-Ahli hospital explosion | Oct 14–20 |
| E3 | Qatar ceasefire & hostage exchanges | Nov 21–Dec 3 |
| E4 | ICJ genocide provisional ruling | Jan 23–29 |
| E5 | Aaron Bushnell self-immolation | Feb 22–28 |
| E6 | Flour massacre / 30,000 death toll | Feb 26–Mar 3 |

---
> **Bullet 1 narration:**
> "The raw dataset is 235 million posts from 4 million users — about 81% of all Bluesky accounts at the time. It covers about 19.5 gigabytes compressed. We obviously can't analyse all of that. The first step is filtering it down to the content we care about."

> **Bullet 2 narration:**
> "We built a 5-tier NLP keyword filter — English high-precision terms like 'Gaza', 'ceasefire', 'occupation', then hashtag patterns, then Arabic-script keywords, Hebrew-script keywords, and finally named entities like 'Aaron Bushnell' or 'al-Ahli'. A post matches if it hits any tier. That gave us 1.95 million conflict-related posts out of 233 million scanned — 0.84%. This ran in about 85 minutes using streaming decompression and 15-worker multiprocessing."

> **Bullet 3 narration:**
> "We then label each post with one of six event windows, or 'baseline' if it falls outside all of them. These six events were chosen to span different crisis categories: a military shock, a contested narrative, a diplomatic event, a legal milestone, an individual emotional story, and a compound humanitarian crisis. This categorical diversity is what makes the dataset a natural experiment — it lets us directly test SRQ1 and SRQ2."

---

## Slide 5 — What is an Information Cascade?

- A **cascade** = the tree of reposts spreading from one original post
- Each node is a user; each edge is a repost connection
- We label edges: **FOLLOWER** (organic) vs **NON_FOLLOWER** (cross-community)
- Result: **39,983 cascades**, 884,832 nodes, 844,849 edges — **75.9% FOLLOWER-typed**

```
            [Root Post]
           /     |      \
    [Follower] [Follower] [Non-Follower ← algorithmic/cross-community]
       |
  [Follower]
```

---
> **Bullet 1 narration:**
> "An information cascade is just the repost tree rooted at an original post. When someone reposts it, they become a child node. When someone reposts the repost, they become a grandchild node. The shape of that tree — how deep it goes, how wide it gets, how many branches it has — tells you something about how the information spread. A deep, narrow tree suggests a chain of sequential reposts; a shallow, wide tree suggests a single post that many people responded to simultaneously."

> **Bullet 2 narration:**
> "We construct these trees from the dataset by processing reposters in timestamp order and applying a parent assignment heuristic. For each new reposter, we ask: is there already someone in the cascade who this person follows? If yes, that's their likely parent — they probably saw it through organic means. If not, we assign the most recent prior reposter as parent and label the edge NON_FOLLOWER. We also compute structural properties for each cascade: depth, breadth, size, the Wiener index — which is the sum of pairwise shortest paths in the tree and serves as a measure of how 'viral' vs 'broadcast' the spread was — and the follower edge ratio."

> **Bullet 3 narration:**
> "After filtering for cascades with at least 10 nodes — to get meaningful structure — we end up with 39,983 cascades. 75.9% of edges are FOLLOWER-typed, meaning crisis information on Bluesky spread predominantly organically. This is a first-pass answer to SRQ3. But the 24.1% non-follower component is where things get interesting — and as we'll see, the model will discover that a small but structurally distinct subset of cascades is defined almost entirely by that non-follower component."

---

## Slide 6 — The Model: BlueCCGL

- **Goal:** learn a vector representation (embedding) of each cascade — capturing its structural fingerprint — *without any labels*
- **How:** contrastive self-supervised learning — show the model two distorted views of the same cascade, teach it they should look similar; different cascades should look different
- **Three novel components** on top of the base CCGL framework:
  1. **Feed-aware augmentations** — distortions that respect organic vs. algorithmic semantics
  2. **R-GCN encoder** — edge-type aware message passing (FOLLOWER and NON_FOLLOWER edges processed separately)
  3. **DiffPool + sentiment** — hierarchical pooling that compresses the tree into a 64-dimensional vector, enriched with sentiment composition

---
> **Bullet 1 narration:**
> "The goal of the model is to produce a single 64-dimensional vector — an embedding — for each of the 39,983 cascades. Think of it like a fingerprint: cascades with similar structural patterns should end up close together in this 64-dimensional space. We want to do this without any labels, because we don't have ground-truth categories for crisis cascades — no one has hand-labelled 40,000 repost trees."

> **Bullet 2 narration:**
> "We use contrastive self-supervised learning via the InfoNCE loss. InfoNCE works like a softmax cross-entropy over cascade identity: for each cascade, the model is given its embedding and the embeddings of all other cascades in the batch, and it must identify which one is its augmented positive pair. Getting this right means the model has learned a representation where structurally similar cascades cluster together. We also add VICReg regularisation — variance and covariance terms — to prevent the model from collapsing all embeddings to the same vector, which is a known failure mode in contrastive learning."

> **Bullet 3 narration:**
> "We extend the base CCGL framework with three contributions. First, the augmentations are domain-specific: instead of random node dropping, we have follower-subgraph extraction — keep only the organic edges — temporal prefix sampling — keep only the earliest nodes — and bridge-node removal — remove the cross-community entry points. Each augmentation has a crisis-communication interpretation and directly tests structural invariances relevant to our research questions. Second, the encoder is a Relational GCN, which uses separate weight matrices for follower and non-follower edges, so the model explicitly learns that these two types of connection mean different things. Third, we use hierarchical DiffPool pooling to compress the tree into an embedding, and concatenate the sentiment composition — proportions of negative, neutral, positive posts — into the final vector."

---

## Slide 7 — Training Results

- Trained for **150 epochs** on 39,983 cascades (CPU-only, ~11 hours)
- Total loss: **1.756 → 0.129** (92.5% reduction); best checkpoint epoch 147
- **Critical limitation:** 79.6% augmentation fallback rate — aggressive perturbation ratios left too little of the cascade, so most pairs trained on unperturbed graphs
- Despite this, the learned embeddings show strong structural signal (coming up)

---
> **Bullet 1 narration:**
> "Training ran on a CPU-only virtual machine for 150 epochs — about 11 hours total. The loss combines the InfoNCE contrastive objective with VICReg regularisation terms. Best checkpoint was epoch 147, selected by held-out loss."

> **Bullet 2 narration:**
> "The total loss dropped 92.5% over training, with the InfoNCE component dropping 96%. Gradient norms declined steadily from 6.92 to 1.32 with no explosion or sustained clipping — the model trained stably."

> **Bullet 3 narration:**
> "One important limitation: 79.6% of augmentation attempts produced a graph with fewer than 5 nodes after perturbation, so they fell back to the unperturbed original. The root cause is that the median cascade is only 10 nodes, and retaining 30% of follower edges gives you 3 nodes — below the threshold. In practice the model trained mostly on unperturbed pairs, which limits the structural invariances it could learn. A sensitivity run with corrected ratios brought this down to 42% fallback, but that run produced weaker embeddings — follower R² dropped from 0.941 to 0.714 — so we kept v8 as the primary result and documented the fallback as a limitation."

---

## Slide 8 — From Embeddings to Archetypes

![UMAP projection of all 39,983 cascade embeddings](../Outputs_17_04/evaluate/umap_figure.png)

- **Left panel:** coloured by HDBSCAN cluster — clear separation into 11 groups
- **Right panel:** coloured by event window — *no* temporal clustering
- Clustering quality (10D UMAP space): **Silhouette = 0.413**, **DBCV = +0.618**, noise = 1.5%
- The model learned **structure**, not **time**

---
> **UMAP/HDBSCAN narration:**
> "After training, we extract the 64-dimensional embedding for every cascade. Before clustering we reduce to 10 dimensions with UMAP — Uniform Manifold Approximation and Projection. UMAP is a non-linear dimensionality reduction technique that tries to preserve the local neighbourhood structure of the data. We need this step because HDBSCAN — the density-based clustering algorithm — struggles to find meaningful density in 64 dimensions due to the curse of dimensionality: without UMAP, over 67% of cascades get labelled as noise. HDBSCAN itself is a hierarchical clustering method that finds clusters as regions of high density separated by regions of low density, without needing you to specify the number of clusters in advance."

> **Left panel narration:**
> "What you see on the left is 39,983 cascades projected further to 2D for visualisation only, coloured by which of 11 clusters HDBSCAN assigned them to. The separation is clean. HDBSCAN found this structure by looking at density in 10D space — no labels, no number of clusters specified."

> **Clustering metrics narration:**
> "How do we know the 11 clusters are real and not arbitrary? We report four validated clustering quality metrics, all computed in the 10D UMAP space where HDBSCAN actually ran. The **Silhouette score** measures for each cascade how similar it is to its own cluster relative to the nearest other cluster — it ranges from -1 to +1, where 1 is perfect separation and 0 means a point is on the cluster boundary. Our score of 0.413 with a 95% bootstrap confidence interval of [0.405, 0.422] is well above chance. The **DBCV** — Density-Based Clustering Validation — is specifically designed for non-convex clusters like HDBSCAN's; unlike silhouette, it validates based on density rather than centroid distance. It also ranges -1 to +1, with positive values meaning clusters are denser than the surrounding space. Our DBCV of +0.618 confirms the clusters are genuine density structures. Only 1.5% of cascades were left unclustered as noise."

> **Right panel narration:**
> "The right panel colours the same points by event window. There is no pattern — October 7th cascades, ICJ ruling cascades, flour massacre cascades all mixed together across the structural clusters. This directly answers SRQ1: the model learned the *shape* of how information spread, not *when*. Temporal structure and cascade structure are genuinely decoupled."

---

## Slide 9 — 11 Archetypes: Six Families

![Radar chart of normalised structural profiles per cluster](../Outputs_17_04/explain/figures/cluster_radar_profiles.png)

- **Pure negative reaction** (C0, C1) — 100% negative sentiment, structurally distinct (deep-narrow vs shallow-wide)
- **Dominant discussion mode** (C3, 42%) — the workhorse: large, deep, broad, mixed sentiment
- **Pure informational** (C4, C5) — 100% neutral; C5 has the *highest* follower ratio (0.91)
- **Cross-community debate** (C8) — *lowest* follower ratio (0.51), deep-narrow, 78% negative

---
> **Radar chart narration:**
> "The radar chart shows normalised structural profiles for all 11 clusters across six axes: depth, breadth, size, follower ratio, negative sentiment proportion, and neutral sentiment proportion. Each axis is normalised so the maximum across all clusters is 1. C8 is visually isolated on the follower-ratio axis at the bottom left — distinctly lower than everything else. C5 is at the top of the follower-ratio axis. C3 projects strongly on size and breadth. The radar chart is a good summary of what makes each archetype distinctive."

> **Pure negative narration:**
> "C0 and C1 are the only clusters where every single post in every cascade carries a negative sentiment label — zero standard deviation on sentiment. Despite this shared property, they look structurally different. C0 is deep and narrow: sustained back-and-forth exchanges that drew outside voices. C1 is shallow and wide: a root post that triggered immediate broad disapproval from followers. Same emotional tone, different social mechanics — this is why unsupervised structural clustering finds things that sentiment analysis alone would miss."

> **C3 narration:**
> "C3 is the largest cluster at 42% of all cascades. This is what normal crisis discourse looks like on Bluesky: substantive multi-level discussion threads, mixed sentiment (59% negative, 28% neutral, 13% positive), moderate cross-community reach. If you imagine how people talk about a geopolitical crisis online — reading, reposting, replying, debating — that's C3."

> **C4/C5 narration:**
> "C4 and C5 are 100% neutral — purely informational posts. C5 in particular has a follower ratio of 0.91, the highest in the dataset. Almost every interaction is between mutual followers. These are insular community coordination posts: logistics, resources, news updates circulated within a closed network. They surge dramatically during E1, the acute shock of October 7th — which we'll see shortly."

> **C8 narration:**
> "C8 is the most structurally unusual cluster. Its follower ratio is 0.51 — half of all edges connect users who don't follow each other. These are genuinely cross-community cascades: deep and narrow, sustained back-and-forth between strangers, 78% negative. C8 is the cross-community debate archetype. It's the model's structural answer to SRQ3."

---

## Slide 10 — What the Model Actually Learned

![Linear probe R² per property and cluster](../Outputs_17_04/explain/figures/probe_heatmap.png)

- **Follower edge ratio: R² = 0.941** — the dominant encoded signal; consistent 0.917–0.981 across every cluster
- **Cascade depth: R² = 0.686** — second strongest
- **Event window: balanced accuracy = 0.143** — exactly 1/7 chance level, no temporal signal

---
> **Setup narration:**
> "To understand what the embeddings actually encode, we run linear probing: we take the frozen 64-dimensional embeddings and train simple linear models on top of them to predict structural and temporal properties. This is a standard interpretability technique in representation learning. If a linear model can predict property X from the frozen embedding, it means X is linearly encoded in the representation — the model learned it. We use **R²** — the coefficient of determination — for regression tasks. R² ranges from minus-infinity to 1: a value of 1 means perfect prediction, 0 means the model does no better than always predicting the mean, and negative values mean it's worse than that baseline. For the classification task, we use **balanced accuracy** — the average recall across all classes. For 7 event windows, the chance level is 1 divided by 7, which is 0.143."

> **Follower ratio narration:**
> "The standout result is follower edge ratio at R² = 0.941 globally, and between 0.917 and 0.981 within every single cluster. This means a linear model trained on the embeddings can predict with 94.1% explained variance what fraction of a cascade's edges were follower-based. The R-GCN edge-type encoding is working: the model has learned to encode the organic/algorithmic composition of each cascade almost perfectly from structure alone. This is the direct methodological evidence for MRQ2 — edge-type aware encoding is encoding what it's supposed to encode."

> **Depth narration:**
> "Cascade depth is the second strongest signal at R² = 0.686. The model knows how deep a cascade went. Cascade breadth shows an interesting dissociation: globally moderate R² at 0.317, because breadth varies substantially between clusters, but within each cluster the within-cluster R² is high — the model has learned the breadth within each structural family. Cascade size globally R² = -6.07, a failure of global linear decoding, because large heterogeneous clusters like C3 have enormous internal variance in size. But within clusters, size is well-encoded at R² 0.686 to 0.911."

> **Event window narration:**
> "Event window at exactly 0.143 balanced accuracy — that's precisely the 1/7 chance level. The model has zero temporal information. This is the most important negative result and the definitive answer to SRQ1's 'are archetypes a timing artefact?' question: they are not. The 11 archetypes are purely structural categories. They reflect genuine differences in how information propagated, completely decoupled from when."

---

## Slide 11 — The Cross-Community Signature

![Follower edge ratio distribution by cluster (violin plots)](../Outputs_17_04/explain/figures/follower_ratio_violins.png)

![Augmentation sensitivity per cluster](../Outputs_17_04/explain/figures/augmentation_sensitivity_bars.png)

- **C8 follower ratio centred at 0.51** — separated far below all other clusters
- **C8 bridge-removal sensitivity = 1.22** — highest in the dataset; removing non-follower entry points fundamentally changes the embedding
- C8 peaks during **E6 (flour massacre / 30,000 deaths)** — highest moral salience event

---
> **Violin narration:**
> "The violin plots show the full distribution of follower ratio within each cluster — not just the mean, but the shape of the distribution. C8 is visually isolated at the bottom, centred around 0.51. C5 is isolated at the top, centred around 0.91. These aren't subtle statistical differences — they're cleanly separated distributions. This is the structural answer to SRQ3: the embedding space is primarily organised along the organic/cross-community axis."

> **Augmentation sensitivity narration:**
> "The augmentation sensitivity analysis gives us a mechanistic, model-internal validation of the archetype interpretations. For each cluster, we take 20 representative cascades and apply each of the three augmentations, then measure how much the embedding changes using **cosine distance** — a measure of how differently two vectors point in embedding space. Cosine distance ranges from 0 (identical direction) to 2 (perfectly opposite). If removing bridge nodes — the non-follower entry points — causes a large cosine distance, it means the embedding was primarily capturing those bridge node connections. C8's bridge-removal sensitivity of 1.22 is the highest in the dataset. Removing the cross-community entry points fundamentally changes what C8 cascades look like to the model. That's a direct structural validation of the cross-community debate interpretation — the model isn't just labelling C8 differently, it's genuinely representing these cascades via their non-follower topology."

> **E6 narration:**
> "C8 peaks at E6 — the flour massacre and 30,000 death toll milestone, the highest compound moral salience event in the dataset. This partially answers SRQ2: compound humanitarian milestones are the event category most associated with cross-community spillover. Content with strong moral framing crosses community boundaries more readily than diplomatic or legal content."

---

## Slide 12 — How Event Type Shapes Cascade Structure

![Archetype × event heatmap](../Outputs_17_04/explain/figures/archetype_event_heatmap.png)

- **E1 (Oct 7, acute shock):** C5 + C6 surge — breaking-news dynamics; pure informational cascades dominate before deliberation begins
- **E4 (ICJ ruling):** C3 peaks at 48.1% — legal milestones produce the most deliberative, sustained discussion
- **E5–E6 (Bushnell, flour massacre):** C7 + C8 elevated — individual human stories and compound humanitarian events drive negative and cross-community engagement

---
> **Heatmap narration:**
> "This heatmap directly addresses SRQ2 — do different event types produce structurally distinct cascade signatures? Each column is an event window; each row is a cluster; colour encodes what percentage of that event window's cascades belong to that cluster. Deviations from the baseline column are the signal."

> **E1 narration:**
> "During E1 — the October 7th attack — the informational archetypes surge. C5 rises from 3.3% baseline to 9.1%, and C6 rises from 8% to 19.7%. Simultaneously C3 drops from 43% to 33%. This is the breaking-news phase: in the immediate aftermath of an acute shock, the community switches from discussion mode to broadcast mode — circulating facts and updates before the deliberative phase begins. This is consistent with crisis communication theory: information-seeking and sharing precedes sense-making."

> **E4 narration:**
> "By E4 — the ICJ genocide provisional ruling — the pattern reverses completely. C3 peaks at 48.1%, its highest value in the entire dataset. The informational archetypes drop to their lowest values. A legal institutional milestone generates sustained deliberative engagement — people processing implications, forming arguments, debating precedent. The temporal arc from E1 to E4 is itself a structural narrative: acute shock → information broadcast → sustained deliberation."

> **E5/E6 narration:**
> "E5 and E6 tell a different story. Aaron Bushnell's self-immolation drives C7 — the negative mid-scale thread — to its peak at 32.7%. The flour massacre drives C8 — cross-community debate — to its peak at 3.3%. Individual emotionally resonant events generate sustained negative engagement within communities; compound humanitarian milestones generate the cross-community spillover. These are structurally distinct event signatures — a direct answer to SRQ2 — and they map interpretably onto what we'd expect from crisis communication research."

---

## Slide 13 — Summary & What's Next

- **11 structural archetypes** discovered with no labels — from pure negative community reaction (C0/C1) to cross-community debate (C8) to the dominant deliberative mode (C3)
- **The primary structural axis:** follower edge ratio (organic vs. cross-community spread) — R² = 0.941
- **Crisis structure ≠ crisis timing** — archetypes are purely structural; event type shifts archetype *composition*, not which archetypes exist
- **Next:** downstream prediction tasks (cascade size/longevity), stability analysis (10× subsampling **ARI**)

---
> **Archetypes narration:**
> "To summarise: from 39,983 unlabelled crisis cascades, BlueCCGL discovers 11 interpretable structural archetypes, grouping naturally into six families. This is fully unsupervised — no labels were used at any point. The archetypes cover the full range from insular organic community coordination, through dominant mixed-sentiment deliberation, to cross-community debate with significant non-follower spread. All 11 cluster differences are statistically significant at p < 10⁻³⁰⁰ on Kruskal-Wallis tests across every structural property."

> **Primary axis narration:**
> "The single most important thing the model learned is the organic/cross-community axis — follower edge ratio at R² = 0.941. This is the direct methodological result for MRQ2: the R-GCN edge-type encoding worked. The model has learned that FOLLOWER and NON_FOLLOWER edges carry fundamentally different structural meaning, and that difference is the primary organising principle of the entire cascade embedding space."

> **Structure ≠ timing narration:**
> "The cleanest theoretical result is the decoupling of structure and time. The 11 archetypes are genuine structural categories — they don't correspond to event windows, crisis phases, or conflict escalation stages. What changes across events is the *mixture* of archetypes: acute shocks shift the community towards informational cascades, legal milestones shift it towards deliberative ones, compound humanitarian events elevate cross-community debate. The archetypes themselves are stable — this is a strong empirical finding."

> **Next steps narration:**
> "Two analyses are pending. First, downstream prediction: can the frozen embeddings transfer to predicting cascade size or longevity? This will tell us whether the representations generalise beyond clustering. Second, stability analysis — do the same 11 archetypes appear if we subsample 80% of the dataset 10 times and re-cluster? We measure this with **ARI — Adjusted Rand Index** — which compares two clusterings and asks how often they agree on whether any two cascades belong to the same cluster, adjusted for chance. ARI of 1 means perfect agreement, 0 means no better than random. High ARI across 10 subsamples would confirm the 11-archetype solution is a robust structural feature of crisis information spread on Bluesky — not an artefact of this particular draw of cascades. Thank you."

---

*Total slides: 13 · Estimated time: 8–10 minutes at natural narration pace*

---

### Metric Reference (quick cheat-sheet)

| Metric | Range | Better | What it measures |
|--------|-------|--------|-----------------|
| **Silhouette** | −1 to +1 | Higher | Intra-cluster cohesion vs inter-cluster separation |
| **DBCV** | −1 to +1 | Higher | Density-based cluster validity (non-convex clusters) |
| **Davies-Bouldin** | 0 to ∞ | Lower | Within-cluster scatter relative to between-cluster separation |
| **Calinski-Harabasz** | 0 to ∞ | Higher | Between-cluster / within-cluster dispersion ratio |
| **R²** | −∞ to 1 | Closer to 1 | Variance in target explained by linear model on embeddings |
| **Balanced accuracy** | 0 to 1 | Higher (chance = 1/n_classes) | Average per-class recall |
| **Cosine distance** | 0 to 2 | — | Angular difference between two embedding vectors |
| **ARI** | −1 to 1 | Higher (chance = 0) | Agreement between two clusterings, adjusted for chance |

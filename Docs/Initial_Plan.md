# BlueCCGL: Unsupervised Discovery of Information Cascade Archetypes on Bluesky
## Bachelor's Thesis Research Plan

---

## Executive Summary

This research proposes **BlueCCGL** (Bluesky Contrastive Cascade Graph Learning), an extension of the CCGL framework for unsupervised discovery of information cascade archetypes on the Bluesky decentralized social network. By combining contrastive self-supervised learning with novel augmentation strategies and edge-type aware encoding, we aim to learn meaningful cascade representations without requiring labeled data—addressing a fundamental challenge in studying spread dynamics on algorithmic-feed platforms.

**Key Innovation**: Unlike prior work requiring ground-truth labels for "viral" vs "non-viral" cascades, our approach discovers natural cascade archetypes through contrastive learning and clustering, revealing how information spreads differently through follower networks versus algorithmic recommendation exposure.

---

## Table of Contents

1. [Research Questions](#research-questions)
2. [Literature Review](#literature-review)
3. [Methodology](#methodology)
4. [Data Availability & Collection](#data-availability--collection)
5. [Technical Architecture](#technical-architecture)
6. [Evaluation Strategy](#evaluation-strategy)
7. [Timeline & Milestones](#timeline--milestones)
8. [References](#references)

---

## Research Questions

### RQ1: Can contrastive self-supervised learning discover meaningful cascade archetypes without labeled data?

**Sub-questions:**
- RQ1.1: What natural clusters emerge when applying HDBSCAN to learned cascade embeddings?
- RQ1.2: How do discovered archetypes differ in structural properties (depth, breadth, density)?
- RQ1.3: Are discovered archetypes stable across different time periods and topic domains?
- RQ1.4: How does the number and nature of archetypes compare to manually-defined categories in prior work?

### RQ2: How does edge-type aware encoding (follower vs. non-follower edges) affect cascade representation quality?

**Sub-questions:**
- RQ2.1: Do cascades with high follower-edge ratios form distinct clusters from algorithmic-exposure-heavy cascades?
- RQ2.2: What is the correlation between edge-type composition and cascade size/longevity?
- RQ2.3: Can edge-type patterns predict cascade evolution (e.g., early follower spread → later algorithmic amplification)?
- RQ2.4: How does R-GCN-style relation encoding compare to homogeneous GNN baselines?

### RQ3: Do feed-aware augmentation strategies produce better representations than generic graph augmentations?

**Sub-questions:**
- RQ3.1: How does follower-subgraph extraction compare to random node dropping for positive pair generation?
- RQ3.2: What is the optimal augmentation strength (% of nodes/edges modified) for cascade graphs?
- RQ3.3: Do domain-specific augmentations (simulating algorithmic vs. organic spread) improve downstream task performance?
- RQ3.4: How sensitive is representation quality to augmentation strategy choice?

### RQ4: What cascade dynamics are unique to Bluesky's decentralized, algorithm-optional architecture?

**Sub-questions:**
- RQ4.1: How do cascade patterns on Bluesky compare to centralized platforms (Twitter/X) reported in literature?
- RQ4.2: Do users who disable algorithmic feeds produce cascades with different structural signatures?
- RQ4.3: What role do "bridge users" (connecting otherwise-separate communities) play in cascade archetypes?
- RQ4.4: How does Bluesky's federation structure (PDS distribution) correlate with cascade patterns?

---

## Literature Review

### 2.1 Information Cascade Prediction

#### Foundational Works

**CasCN (Chen et al., 2019)** - "Information Diffusion Prediction via Recurrent Cascades Convolution"
- First to apply graph neural networks to cascade prediction
- Models cascade as sequence of snapshots with temporal attention
- Limitation: Requires labeled popularity data for supervision

**DeepHawkes (Cao et al., 2017)** - "DeepHawkes: Bridging the Gap between Prediction and Understanding of Information Cascades"
- Combines Hawkes processes with deep learning
- Interpretable attention over user influence
- Limitation: Assumes homogeneous user influence

**CasFlow (Yuan et al., 2024)** - "CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction"
- Variational approach capturing cascade uncertainty
- Hierarchical pooling for multi-scale patterns
- State-of-art on Weibo and Twitter datasets

#### Recent Advances (2023-2025)

**HierCas (Wang et al., 2024)** - "Hierarchical Information Propagation Prediction"
- Multi-resolution cascade modeling
- Captures both micro (user-level) and macro (community-level) dynamics
- Relevant for our hierarchical pooling extension

**TempCas (Liu et al., 2025)** - "Temporal Cascade Transformers"
- Transformer architecture for cascade sequences
- Continuous-time attention mechanism
- Shows importance of temporal modeling we incorporate

### 2.2 Contrastive Learning on Graphs

#### Core Methods

**CCGL (Xu et al., 2022)** - "Contrastive Cascade Graph Learning" [**BASE METHOD**]
- **Paper**: IEEE Transactions on Knowledge and Data Engineering
- **Key Innovation**: Self-supervised pre-training for cascade graphs
- **Architecture**:
  - Sub-cascade sampling for data augmentation
  - Graph-level contrastive loss (InfoNCE)
  - Fine-tuning for downstream prediction
- **Results**: 10-20% improvement over supervised baselines
- **Code**: https://github.com/ChenNed/CCGL
- **Limitation**: Generic augmentations, no edge-type awareness

**GraphCL (You et al., 2020)** - "Graph Contrastive Learning with Augmentations"
- Systematic study of graph augmentation strategies
- Node dropping, edge perturbation, attribute masking, subgraph sampling
- Foundation for our augmentation design

**GRACE (Zhu et al., 2020)** - "Deep Graph Contrastive Representation Learning"
- Node-level contrastive learning
- Corruption-based augmentation
- Inspires our negative sampling strategy

#### Self-Supervised Learning for Temporal Graphs

**TGC (Zhang et al., 2023)** - "Temporal Graph Contrastive Learning"
- Contrastive learning for dynamic graphs
- Time-aware augmentation strategies
- Relevant for our cascade temporal modeling

**CSTGCN (Li et al., 2024)** - "Contrastive Self-Supervised Temporal Graph Learning"
- Captures evolving structural patterns
- Multi-scale temporal contrasting
- Informs our temporal augmentation design

### 2.3 Heterogeneous and Relational GNNs

#### Edge-Type Aware Methods

**R-GCN (Schlichtkrull et al., 2018)** - "Modeling Relational Data with Graph Convolutional Networks"
- Relation-specific weight matrices
- Basis decomposition for parameter efficiency
- **Direct influence on our Extension 2**

**HAN (Wang et al., 2019)** - "Heterogeneous Graph Attention Network"
- Hierarchical attention: node-level and semantic-level
- Meta-path based aggregation
- Inspires our edge-type attention mechanism

**HGT (Hu et al., 2020)** - "Heterogeneous Graph Transformer"
- Transformer-style attention for heterogeneous graphs
- Type-specific linear projections
- State-of-art heterogeneous GNN architecture

#### Application to Social Networks

**RHGNN (Yu et al., 2023)** - "Relation-aware Heterogeneous GNN for Social Networks"
- Models multiple relationship types (follow, mention, repost)
- Shows importance of relation distinction for social dynamics
- Validates our edge-type encoding approach

### 2.4 Organic vs. Algorithmic Spread

#### Empirical Studies

**Bandy & Diakopoulos (2021)** - "Auditing News Curation Systems"
- Methodology for distinguishing algorithmic amplification
- Proposes "amplification ratio" metric
- Limitation: Requires platform-insider access

**Huszár et al. (2022)** - "Algorithmic Amplification of Politics on Twitter" [Twitter Internal Study]
- Landmark study with insider data
- Shows algorithmic amplification of political content
- Gold standard we approximate through structure

**Bhadani et al. (2024)** - "Algorithmic Amplification of Low-Quality Content"
- Studies spread pattern differences
- Proposes structural signatures of algorithmic boost
- Informs our feature engineering

#### Methodological Approaches

**Goel et al. (2016)** - "The Structural Virality of Online Diffusion"
- Defines "structural virality" metric (Wiener index)
- Distinguishes broadcast vs. viral spread
- Key baseline metric for our analysis

**Vosoughi et al. (2018)** - "The Spread of True and False News Online"
- Cascade structure analysis at scale
- Shows true/false news spread differently
- Methodology for cascade comparison

### 2.5 Decentralized Social Networks

#### Platform Studies

**Zignani et al. (2018)** - "Mastodon and the Fediverse" [Supervisor's Work]
- First large-scale Mastodon measurement
- Analyzes federation structure and user behavior
- Foundation for decentralized platform research

**Raman et al. (2019)** - "Challenges in the Decentralised Web" [Supervisor's Work]
- Technical analysis of federation protocols
- Identifies performance and adoption challenges
- Informs our understanding of AT Protocol

**Cerisara et al. (2025)** - "Bluesky Ecosystem Analysis"
- Early Bluesky user behavior study
- Analyzes AT Protocol adoption
- Provides baseline Bluesky statistics

**Failla & Rossetti (2024)** - "I'm in the Bluesky Tonight: Insights from a Year's Worth of Social Data" [**PRIMARY DATA SOURCE**]
- **Paper**: PLoS ONE, DOI: 10.1371/journal.pone.0310330
- **Dataset**: Zenodo, DOI: 10.5281/zenodo.14669616 (22 GB, CC0/Bluesky ToS)
- **Coverage**: 4M+ users (~81% of registered accounts), 235M posts, Feb 2023–Mar 2024
- **Key Files**:
  - `followers.csv.gz`: Directed follower edge list (~144M edges)
  - `interactions.csv.gz`: Timestamped comment, repost, and quote interactions
  - `graphs.tar.gz`: Timestamped repost/quote/reply graphs + thread-level interactions
  - `feed_posts.tar.gz`: Posts from 11 feed generators
  - `feed_bookmarks.csv`: User feed-subscription data (who bookmarked which feed and when)
- **Limitations**: Pseudonymized (integer IDs, no DIDs/URIs), timestamps coarsened to minute level, single follower-graph snapshot (Mar 2024), data ends Mar 2024
- **Advantage over Jetstream**: Provides follow graph, feed-subscription data, and full historical coverage without requiring real-time collection

#### AT Protocol Specific

**Kleppmann & Howard (2024)** - "Decentralized Social Media Protocols"
- Technical comparison: ActivityPub vs AT Protocol
- Analyzes data availability differences
- Explains Bluesky Firehose architecture

### 2.6 Hierarchical Graph Pooling

#### Pooling Methods

**DiffPool (Ying et al., 2018)** - "Hierarchical Graph Representation Learning"
- Differentiable pooling via soft cluster assignment
- Learns hierarchical graph structure
- **Direct influence on our Extension 3**

**SAGPool (Lee et al., 2019)** - "Self-Attention Graph Pooling"
- Attention-based node selection
- More efficient than DiffPool
- Alternative for our implementation

**MinCutPool (Bianchi et al., 2020)** - "Spectral Clustering with Graph Neural Networks"
- Spectral clustering objective for pooling
- Theoretically grounded
- Produces interpretable clusters

#### Application to Cascades

**HiCas (Zhou et al., 2023)** - "Hierarchical Cascade Representation"
- Applies hierarchical pooling to cascades
- Captures multi-scale propagation patterns
- Most relevant prior work for Extension 3

### 2.7 Clustering for Pattern Discovery

#### Deep Clustering

**DEC (Xie et al., 2016)** - "Unsupervised Deep Embedding for Clustering Analysis"
- Joint representation learning and clustering
- Iterative refinement of cluster assignments
- Foundation for embedding-based clustering

**HDBSCAN (McInnes et al., 2017)** - "HDBSCAN: Hierarchical Density-Based Clustering"
- Density-based clustering without k specification
- Handles noise and varying cluster densities
- **Our chosen clustering method**

#### Graph Clustering

**DMoN (Tsitsulin et al., 2023)** - "Graph Clustering with GNNs"
- End-to-end graph clustering
- Modularity-based objective
- Alternative to post-hoc clustering

---

## Methodology

### 3.1 Base Framework: CCGL Adaptation

#### Original CCGL Pipeline
```
Input Cascade Graph G = (V, E, X)
    ↓
Sub-cascade Sampling (Augmentation)
    ↓
GNN Encoder → Graph Embedding z
    ↓
Contrastive Loss (InfoNCE)
    ↓
Pre-trained Encoder
    ↓
Fine-tuning for Prediction
```

#### Our Modifications (BlueCCGL)
```
Input Cascade Graph G = (V, E, X, R)  [R = edge types]
    ↓
Feed-Aware Augmentation (Extension 1)
    ↓
Edge-Type Aware R-GCN Encoder (Extension 2)
    ↓
Hierarchical Pooling (Extension 3)
    ↓
Contrastive Loss + Clustering Regularization
    ↓
Pre-trained Encoder
    ↓
HDBSCAN Clustering → Archetypes
```

### 3.2 Extension 1: Feed-Aware Augmentation

#### Motivation
Standard graph augmentations (random node/edge dropping) ignore the semantic meaning of cascade structure. We propose augmentations that respect the distinction between organic (follower-based) and algorithmic spread.

#### Augmentation Strategies

**Strategy A: Follower-Subgraph Extraction**
```python
def follower_subgraph_augmentation(cascade_graph, follow_graph, ratio=0.7):
    """
    Extract subgraph containing only follower-based edges.
    This simulates "what would spread look like without algorithms?"
    """
    follower_edges = []
    for (u, v) in cascade_graph.edges():
        if follow_graph.has_edge(v, u):  # v follows u
            follower_edges.append((u, v))
    
    # Sample subset of follower edges
    sampled = random.sample(follower_edges, int(len(follower_edges) * ratio))
    return cascade_graph.edge_subgraph(sampled)
```

**Strategy B: Temporal Prefix Sampling**
```python
def temporal_prefix_augmentation(cascade_graph, timestamps, ratio=0.6):
    """
    Sample early portion of cascade (before potential algorithmic amplification).
    Based on observation that organic spread dominates early cascade stages.
    """
    sorted_nodes = sorted(cascade_graph.nodes(), key=lambda n: timestamps[n])
    prefix_nodes = sorted_nodes[:int(len(sorted_nodes) * ratio)]
    return cascade_graph.subgraph(prefix_nodes)
```

**Strategy C: Bridge-Node Removal**
```python
def bridge_removal_augmentation(cascade_graph, follow_graph, ratio=0.3):
    """
    Remove nodes that bridge communities (likely algorithmic exposure).
    These nodes received content from non-followers.
    """
    bridge_nodes = []
    for node in cascade_graph.nodes():
        parents = list(cascade_graph.predecessors(node))
        if parents and not any(follow_graph.has_edge(node, p) for p in parents):
            bridge_nodes.append(node)
    
    # Remove subset of bridge nodes
    to_remove = random.sample(bridge_nodes, int(len(bridge_nodes) * ratio))
    return cascade_graph.subgraph([n for n in cascade_graph.nodes() if n not in to_remove])
```

#### Contrastive Pair Generation
- **Positive pairs**: Same cascade with different augmentations
- **Negative pairs**: Different cascades (in-batch negatives)
- **Hard negatives**: Cascades from same time period but different structure

### 3.3 Extension 2: Edge-Type Aware Encoding

#### Edge Type Definition

```python
def assign_edge_types(cascade_graph, follow_graph):
    """
    Assign edge types based on follower relationship.
    
    Edge Types:
    - Type 0: FOLLOWER - Target follows source (organic spread)
    - Type 1: NON_FOLLOWER - Target doesn't follow source (algorithmic/external)
    - Type 2: UNKNOWN - Follow relationship unknown (API limitation)
    """
    edge_types = {}
    for (source, target) in cascade_graph.edges():
        if follow_graph.has_edge(target, source):
            edge_types[(source, target)] = 0  # FOLLOWER
        elif follow_graph.has_node(target) and follow_graph.has_node(source):
            edge_types[(source, target)] = 1  # NON_FOLLOWER
        else:
            edge_types[(source, target)] = 2  # UNKNOWN
    return edge_types
```

#### R-GCN Layer Implementation

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EdgeTypeAwareGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations=3, num_bases=2):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        
        # Basis decomposition for parameter efficiency
        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.root = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type):
        # Compute relation-specific weights via basis decomposition
        # W_r = sum_b(a_rb * B_b)
        weights = torch.einsum('rb,bio->rio', self.att, self.basis)
        
        out = self.root(x)
        
        for r in range(self.num_relations):
            mask = edge_type == r
            if mask.sum() > 0:
                edge_index_r = edge_index[:, mask]
                out += self.propagate(edge_index_r, x=x, weight=weights[r])
        
        out += self.bias
        return out
    
    def message(self, x_j, weight):
        return torch.matmul(x_j, weight)
```

#### Multi-Head Attention Variant

```python
class EdgeTypeAttentionGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations=3, heads=4):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        
        # Separate attention for each edge type
        self.relation_attention = nn.ModuleList([
            nn.MultiheadAttention(in_channels, heads, batch_first=True)
            for _ in range(num_relations)
        ])
        
        self.relation_transform = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(num_relations)
        ])
        
        self.combine = nn.Linear(out_channels * num_relations, out_channels)
    
    def forward(self, x, edge_index, edge_type):
        outputs = []
        for r in range(len(self.relation_attention)):
            mask = edge_type == r
            if mask.sum() > 0:
                # Apply relation-specific attention and transform
                edge_index_r = edge_index[:, mask]
                # ... attention computation ...
                out_r = self.relation_transform[r](x)
            else:
                out_r = torch.zeros_like(x[:, :self.out_channels])
            outputs.append(out_r)
        
        return self.combine(torch.cat(outputs, dim=-1))
```

### 3.4 Extension 3: Hierarchical Pooling

#### Motivation
Cascades exhibit multi-scale structure:
- **Micro-level**: Individual user decisions
- **Meso-level**: Community-level spread patterns
- **Macro-level**: Global cascade shape

Hierarchical pooling captures these scales in the representation.

#### DiffPool-Based Implementation

```python
class HierarchicalCascadeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, pool_ratios=[0.5, 0.25]):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers for node embeddings
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            # Message passing layer
            self.convs.append(EdgeTypeAwareGCN(dims[i], dims[i+1]))
            
            # Pooling layer (except last)
            if i < len(pool_ratios):
                # Assignment network for soft clustering
                self.pools.append(
                    DiffPoolLayer(dims[i+1], int(dims[i+1] * pool_ratios[i]))
                )
        
        # Final readout
        self.readout = nn.Sequential(
            nn.Linear(sum(dims[1:]), output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x, edge_index, edge_type, batch):
        all_outputs = []
        
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_type))
            
            # Global pooling at each level for hierarchical representation
            all_outputs.append(global_mean_pool(x, batch))
            
            # Coarsen graph (except last layer)
            if i < len(self.pools):
                x, edge_index, edge_type, batch = self.pools[i](
                    x, edge_index, edge_type, batch
                )
        
        # Concatenate multi-scale representations
        multi_scale = torch.cat(all_outputs, dim=-1)
        return self.readout(multi_scale)


class DiffPoolLayer(nn.Module):
    def __init__(self, in_channels, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        
        # GNN for computing cluster assignments
        self.assign_gnn = EdgeTypeAwareGCN(in_channels, num_clusters)
        
        # GNN for computing node embeddings
        self.embed_gnn = EdgeTypeAwareGCN(in_channels, in_channels)
    
    def forward(self, x, edge_index, edge_type, batch):
        # Compute soft cluster assignments
        s = F.softmax(self.assign_gnn(x, edge_index, edge_type), dim=-1)
        
        # Compute node embeddings
        z = self.embed_gnn(x, edge_index, edge_type)
        
        # Pool nodes into clusters
        x_pooled = torch.matmul(s.t(), z)
        
        # Compute coarsened adjacency
        adj = to_dense_adj(edge_index, batch)
        adj_pooled = torch.matmul(torch.matmul(s.t(), adj), s)
        edge_index_pooled = dense_to_sparse(adj_pooled)[0]
        
        # Aggregate edge types (majority voting)
        edge_type_pooled = self._aggregate_edge_types(edge_type, s, edge_index)
        
        # Update batch assignments
        batch_pooled = self._coarsen_batch(batch, s)
        
        return x_pooled, edge_index_pooled, edge_type_pooled, batch_pooled
```

### 3.5 Complete BlueCCGL Architecture

```python
class BlueCCGL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Node feature encoder (user features)
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Hierarchical edge-type aware encoder
        self.graph_encoder = HierarchicalCascadeEncoder(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embed_dim,
            num_layers=config.num_layers,
            pool_ratios=config.pool_ratios
        )
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.proj_dim)
        )
        
        # Temperature parameter
        self.tau = config.temperature
    
    def encode(self, x, edge_index, edge_type, batch):
        """Encode cascade graph to embedding."""
        x = self.node_encoder(x)
        z = self.graph_encoder(x, edge_index, edge_type, batch)
        return z
    
    def project(self, z):
        """Project embedding to contrastive space."""
        return F.normalize(self.projector(z), dim=-1)
    
    def contrastive_loss(self, z1, z2):
        """InfoNCE loss for contrastive learning."""
        # Project to contrastive space
        p1 = self.project(z1)
        p2 = self.project(z2)
        
        # Compute similarity matrix
        sim = torch.matmul(p1, p2.t()) / self.tau
        
        # Labels: positive pairs on diagonal
        labels = torch.arange(sim.size(0), device=sim.device)
        
        # Cross-entropy loss (both directions)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        
        return loss
    
    def forward(self, batch1, batch2):
        """Forward pass for contrastive pre-training."""
        # Encode both augmented views
        z1 = self.encode(batch1.x, batch1.edge_index, batch1.edge_type, batch1.batch)
        z2 = self.encode(batch2.x, batch2.edge_index, batch2.edge_type, batch2.batch)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(z1, z2)
        
        return loss, z1, z2
```

---

## Data Availability & Collection

### 4.1 Primary Data Source: Failla & Rossetti (2024) Zenodo Dataset

#### Overview
- **Source**: Zenodo, DOI: 10.5281/zenodo.14669616
- **Paper**: Failla & Rossetti (2024), PLoS ONE, DOI: 10.1371/journal.pone.0310330
- **Size**: 22 GB compressed
- **Coverage**: 4M+ users, 235M posts, Feb 2023–Mar 2024
- **Access**: Public download, no authentication required
- **License**: Bluesky ToS

#### Relevant Dataset Files

| File | Size | Contents | Use Case |
|------|------|----------|----------|
| `interactions.csv.gz` | 1.0 GB | Timestamped repost, quote, reply interactions (user_id, replied_author, thread_root_author, reposted_author, quoted_author, date) | Cascade edge construction |
| `graphs.tar.gz` | 891 MB | Timestamped edge lists for repost, quote, and reply graphs + thread-level interactions | Cascade graph reconstruction |
| `followers.csv.gz` | 491 MB | Directed follower edge list (~144M edges, user u follows user v) | Edge-type labelling (follower vs. non-follower) |
| `user_posts.tar.gz` | 19.5 GB | JSON-formatted posts per user (text, timestamps, reply refs) | Cascade root identification, content filtering |
| `feed_posts.tar.gz` | 16.4 MB | Posts from 11 feed generators | Feed-exposure analysis |
| `feed_bookmarks.csv` | 553 KB | User feed-subscription data (feed name, user ID, timestamp) | Identifying users with algorithmic feed exposure |
| `feed_posts_likes.tar.gz` | 35.4 MB | Likes on feed posts (liker ID, author ID, post ID, timestamp) | Engagement signal |

### 4.1.1 Alternative: Bluesky Jetstream (Supplement / Future Work)

- **Firehose**: Raw AT Protocol event stream (CBOR format)
- **Jetstream**: Processed JSON stream
- **Access**: Public, no authentication required
- **URL**: `wss://jetstream2.us-east.bsky.network/subscribe`
- **Use case**: Could supplement the Zenodo dataset with more recent data (post-Mar 2024) if needed

| Event Type | AT Protocol Lexicon | Available Fields | Use Case |
|------------|---------------------|------------------|----------|
| Post | `app.bsky.feed.post` | DID, text, timestamp, reply refs | Cascade roots |
| Repost | `app.bsky.feed.repost` | DID, subject URI, timestamp | Cascade edges |
| Like | `app.bsky.feed.like` | DID, subject URI, timestamp | Engagement signal |
| Follow | `app.bsky.graph.follow` | Follower DID, followed DID, timestamp | Edge type labeling |

#### Data Collection Pipeline (Reference — for live Jetstream supplement)

> **Note:** The `BlueskyCollector` class below is a reference implementation for live Jetstream-based data collection. For this project, the **primary data source** is the Failla & Rossetti (2024) Zenodo dataset (§4.1). This pipeline is retained for reference and may be used to supplement the dataset with post-March 2024 data if needed.

```python
import asyncio
import websockets
import json
from datetime import datetime
from collections import defaultdict

class BlueskyCollector:
    def __init__(self, output_dir: str):
        self.ws_url = "wss://jetstream2.us-east.bsky.network/subscribe"
        self.output_dir = output_dir
        self.cascades = defaultdict(list)  # root_uri -> list of reposts
        self.follow_graph = {}  # follower -> set of followed
        
    async def connect(self, collections: list[str]):
        """Connect to Jetstream and filter by collection types."""
        params = "&".join([f"wantedCollections={c}" for c in collections])
        url = f"{self.ws_url}?{params}"
        
        async with websockets.connect(url) as ws:
            async for message in ws:
                event = json.loads(message)
                await self.process_event(event)
    
    async def process_event(self, event: dict):
        """Process incoming Jetstream event."""
        commit = event.get("commit", {})
        collection = commit.get("collection", "")
        operation = commit.get("operation", "")
        
        if operation != "create":
            return
            
        if collection == "app.bsky.feed.repost":
            await self.process_repost(event)
        elif collection == "app.bsky.graph.follow":
            await self.process_follow(event)
        elif collection == "app.bsky.feed.post":
            await self.process_post(event)
    
    async def process_repost(self, event: dict):
        """Extract repost information for cascade construction."""
        record = event["commit"]["record"]
        repost_info = {
            "reposter_did": event["did"],
            "original_uri": record["subject"]["uri"],
            "timestamp": record["createdAt"],
            "event_time": event["time_us"]
        }
        
        # Group by original post URI
        root_uri = record["subject"]["uri"]
        self.cascades[root_uri].append(repost_info)
    
    async def process_follow(self, event: dict):
        """Build follower graph for edge type assignment."""
        record = event["commit"]["record"]
        follower = event["did"]
        followed = record["subject"]
        
        if follower not in self.follow_graph:
            self.follow_graph[follower] = set()
        self.follow_graph[follower].add(followed)
    
    async def process_post(self, event: dict):
        """Track original posts as potential cascade roots."""
        record = event["commit"]["record"]
        post_info = {
            "author_did": event["did"],
            "uri": f"at://{event['did']}/app.bsky.feed.post/{event['commit']['rkey']}",
            "timestamp": record["createdAt"],
            "text_length": len(record.get("text", "")),
            "has_media": "embed" in record
        }
        # Store for later cascade root identification
        
    def construct_cascade_graph(self, root_uri: str) -> dict:
        """Construct cascade graph from collected reposts."""
        reposts = sorted(self.cascades[root_uri], key=lambda x: x["timestamp"])
        
        if len(reposts) < 5:  # Minimum cascade size
            return None
        
        # Build graph
        nodes = [{"id": 0, "did": reposts[0]["original_uri"].split("/")[2], "time": 0}]
        edges = []
        
        did_to_node = {nodes[0]["did"]: 0}
        
        for i, repost in enumerate(reposts):
            node_id = len(nodes)
            did = repost["reposter_did"]
            
            if did in did_to_node:
                continue  # Skip duplicate reposts
            
            did_to_node[did] = node_id
            nodes.append({
                "id": node_id,
                "did": did,
                "time": i + 1  # Relative time ordering
            })
            
            # Find parent (who exposed this user to the content)
            parent = self.find_cascade_parent(did, did_to_node, repost["timestamp"])
            
            # Determine edge type
            edge_type = self.get_edge_type(parent_did=nodes[parent]["did"], child_did=did)
            
            edges.append({
                "source": parent,
                "target": node_id,
                "type": edge_type
            })
        
        return {
            "root_uri": root_uri,
            "nodes": nodes,
            "edges": edges,
            "size": len(nodes),
            "depth": self.compute_depth(edges),
            "follower_ratio": self.compute_follower_ratio(edges)
        }
    
    def find_cascade_parent(self, child_did: str, did_to_node: dict, timestamp: str) -> int:
        """
        Heuristic to find likely parent in cascade.
        Priority: 1) Followed user who reposted earlier, 2) Most recent reposter
        """
        # Check if child follows any existing cascade member
        if child_did in self.follow_graph:
            followed = self.follow_graph[child_did]
            for did, node_id in did_to_node.items():
                if did in followed:
                    return node_id
        
        # Default: connect to root (conservative assumption)
        return 0
    
    def get_edge_type(self, parent_did: str, child_did: str) -> int:
        """Determine edge type based on follow relationship."""
        if child_did in self.follow_graph:
            if parent_did in self.follow_graph[child_did]:
                return 0  # FOLLOWER
            else:
                return 1  # NON_FOLLOWER
        return 2  # UNKNOWN


# Usage
async def main():
    collector = BlueskyCollector("./data")
    await collector.connect([
        "app.bsky.feed.repost",
        "app.bsky.feed.post",
        "app.bsky.graph.follow"
    ])

asyncio.run(main())
```

### 4.2 Data Availability Verification

| Required Data | Available in Zenodo Dataset | Source File | Limitations |
|--------------|----------------------------|-------------|-------------|
| Repost interactions | ✅ Yes | `interactions.csv.gz`, `graphs.tar.gz` | Timestamped, pseudonymized integer IDs |
| Timestamps | ✅ Yes | All interaction files | Coarsened to minute level (not microsecond) |
| User IDs | ✅ Yes | All files | Pseudonymized integers (no DIDs/handles) |
| Follow graph | ✅ Yes | `followers.csv.gz` | Single snapshot (~Mar 2024), ~144M edges |
| Post content | ✅ Yes | `user_posts.tar.gz` | Text, timestamps, reply refs; no URIs |
| Feed generator posts | ✅ Yes | `feed_posts.tar.gz` | 11 popular feeds, JSON-formatted |
| Feed subscriptions | ✅ Yes | `feed_bookmarks.csv` | Feed name, user ID, subscription timestamp |
| Feed post likes | ✅ Yes | `feed_posts_likes.tar.gz` | Liker ID, author ID, post ID, timestamp |

### 4.3 Cascade Reconstruction Feasibility

#### What's Possible
1. **Complete repost chains**: The dataset covers ~81% of all registered accounts, providing near-complete repost coverage for the observation period (Feb 2023–Mar 2024)
2. **Temporal ordering**: Minute-level timestamps for cascade reconstruction
3. **Follower relationships**: Full follower graph snapshot (~144M edges) for edge-type assignment
4. **Edge type assignment**: Cross-reference cascade edges against follower graph to determine follower vs. non-follower edges
5. **Feed exposure data**: Feed-subscription and feed-post data enables analysis of algorithmic exposure (unavailable via Jetstream)

#### What's Approximated
1. **True exposure source**: We infer parent via follow graph + timing heuristic
2. **Algorithmic vs organic**: Structural proxy (non-follower edges suggest algorithmic); enriched by feed-subscription data
3. **Temporal follower dynamics**: Follower graph is a single snapshot; some edges may not have existed at cascade time
4. **Content features**: Limited to text, no engagement metrics beyond likes on feed posts

### 4.4 Estimated Data Requirements

| Metric | Estimate | Justification |
|--------|----------|---------------|
| Source dataset | Failla & Rossetti (2024) Zenodo | 22 GB compressed; 4M+ users, 235M posts |
| Observation period | Feb 2023–Mar 2024 (13 months) | Full dataset coverage |
| Minimum cascades | 50,000 | Statistical power for clustering |
| Min cascade size | 10 reposts | Meaningful structure |
| Max cascade size | 10,000 reposts | Computational tractability |
| Follow graph coverage | 4M+ users, ~144M edges | Full dataset follower graph |
| Storage (raw download) | ~22 GB | Compressed Zenodo files |
| Storage (processed) | ~50 GB | Decompressed + PyG dataset |

---

## Technical Architecture

### 5.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA ACQUISITION                           │
├─────────────────────────────────────────────────────────────────┤
│  Failla & Rossetti  →  Decompress &   →  Parquet / DuckDB      │
│  Zenodo Dataset         Parse (Python)    (Storage)             │
│  (22 GB download)                                               │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                     CASCADE CONSTRUCTION                        │
├─────────────────────────────────────────────────────────────────┤
│  Interactions  →  Graph Builder  →  Edge Type Assignment  → PyG │
│  + Repost Graphs   (NetworkX)      (Follower Graph)       Dataset│
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                             │
├─────────────────────────────────────────────────────────────────┤
│  BlueCCGL Model  ←  Feed-Aware Augmentation  ←  Data Loader    │
│  (PyTorch Geometric)                                            │
│         ↓                                                       │
│  Contrastive Pre-training  →  Cascade Embeddings               │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHETYPE DISCOVERY                          │
├─────────────────────────────────────────────────────────────────┤
│  Embeddings  →  HDBSCAN Clustering  →  Archetype Analysis      │
│                                          (Visualization)        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| Deep Learning | PyTorch | 2.0+ |
| Graph Learning | PyTorch Geometric | 2.3+ |
| Graph Processing | NetworkX | 3.0+ |
| Data Storage | Parquet / DuckDB | Latest |
| Clustering | hdbscan, scikit-learn | Latest |
| Visualization | matplotlib, plotly | Latest |
| Experiment Tracking | Weights & Biases | Latest |

### 5.3 Computational Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| Storage | 100GB SSD | 500GB NVMe |
| CPU | 8 cores | 16+ cores |

---

## Evaluation Strategy

### 6.1 Intrinsic Evaluation (Representation Quality)

#### Clustering Metrics
- **Silhouette Score**: Cluster cohesion and separation
- **Davies-Bouldin Index**: Cluster quality
- **Calinski-Harabasz Index**: Cluster density

#### Embedding Analysis
- **t-SNE / UMAP Visualization**: 2D projection of embeddings
- **Nearest Neighbor Consistency**: Do similar cascades cluster together?

### 6.2 Archetype Validation

#### Structural Coherence
- Cascades in same archetype should have similar:
  - Depth distribution
  - Branching factor
  - Follower edge ratio
  - Temporal spread pattern

#### Interpretability
- Can archetypes be described in human-understandable terms?
- Do they align with known cascade patterns from literature?

### 6.3 Ablation Studies

| Ablation | Purpose |
|----------|---------|
| Remove edge-type encoding | Measure contribution of Extension 2 |
| Generic augmentation | Measure contribution of Extension 1 |
| Flat pooling | Measure contribution of Extension 3 |
| Random initialization | Verify contrastive pre-training benefit |

### 6.4 Downstream Tasks

#### Transfer Learning Evaluation
1. **Cascade size prediction**: Predict final size from early cascade
2. **Cascade longevity**: Predict how long cascade remains active
3. **Cascade classification**: Predict topic/domain of cascade

#### Cross-Platform Comparison
- Compare discovered archetypes with Twitter/Weibo patterns in literature
- Identify Bluesky-specific cascade dynamics

---

## Timeline & Milestones

### Phase 1: Data Acquisition & Processing (Weeks 1-2)
- [ ] Download Failla & Rossetti (2024) Zenodo dataset (~22 GB)
- [ ] Decompress and load follower graph, interaction edge lists, repost graphs
- [ ] Data quality checks (timestamp consistency, deduplication)
- [ ] Compute exploratory statistics (repost volumes, cascade size distribution, follower-graph density)

### Phase 2: Cascade Construction (Weeks 3-4)
- [ ] Implement cascade graph builder from interaction/repost data
- [ ] Assign edge types by cross-referencing against follower graph
- [ ] Create PyTorch Geometric dataset
- [ ] Exploratory data analysis (depth/breadth distributions, edge-type ratios, temporal patterns)

### Phase 3: Model Implementation (Weeks 5-8)
- [ ] Implement base CCGL
- [ ] Add Extension 1 (Feed-Aware Augmentation)
- [ ] Add Extension 2 (Edge-Type Encoding)
- [ ] Add Extension 3 (Hierarchical Pooling)
- [ ] Contrastive pre-training pipeline

### Phase 4: Experiments (Weeks 9-12)
- [ ] Train BlueCCGL model; hyperparameter tuning
- [ ] Extract cascade embeddings; HDBSCAN clustering
- [ ] Ablation studies
- [ ] Downstream transfer-learning evaluation
- [ ] Archetype analysis and interpretation

### Phase 5: Writing (Weeks 13-16)
- [ ] Results analysis
- [ ] Thesis writing
- [ ] Visualizations
- [ ] Final revisions and presentation preparation

---

## References

### Core Methods
1. Xu, Y., et al. (2022). "CCGL: Contrastive Cascade Graph Learning." IEEE TKDE.
2. Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." ESWC.
3. Ying, Z., et al. (2018). "Hierarchical Graph Representation Learning with Differentiable Pooling." NeurIPS.

### Cascade Prediction
4. Chen, X., et al. (2019). "Information Diffusion Prediction via Recurrent Cascades Convolution." ICDE.
5. Cao, Q., et al. (2017). "DeepHawkes: Bridging the Gap between Prediction and Understanding." CIKM.
6. Yuan, C., et al. (2024). "CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty." WWW.

### Contrastive Learning
7. You, Y., et al. (2020). "Graph Contrastive Learning with Augmentations." NeurIPS.
8. Zhu, Y., et al. (2020). "Deep Graph Contrastive Representation Learning." ICML Workshop.

### Decentralized Social Networks
9. Zignani, M., et al. (2018). "Mastodon and the Fediverse." [Supervisor's Work]
10. Raman, A., et al. (2019). "Challenges in the Decentralised Web." [Supervisor's Work]
14. Failla, A. & Rossetti, G. (2024). "I'm in the Bluesky Tonight: Insights from a Year's Worth of Social Data." PLoS ONE. DOI: 10.1371/journal.pone.0310330. Dataset: Zenodo DOI: 10.5281/zenodo.14669616.

### Organic vs Algorithmic Spread
11. Huszár, F., et al. (2022). "Algorithmic Amplification of Politics on Twitter." PNAS.
12. Goel, S., et al. (2016). "The Structural Virality of Online Diffusion." Management Science.

### Clustering
13. McInnes, L., et al. (2017). "HDBSCAN: Hierarchical Density-Based Clustering." JOSS.

---

## Appendix A: Bluesky AT Protocol Reference

### Relevant Lexicons
- `app.bsky.feed.post` - Original posts
- `app.bsky.feed.repost` - Reposts (our primary cascade edges)
- `app.bsky.feed.like` - Likes (engagement signal)
- `app.bsky.graph.follow` - Follow relationships
- `app.bsky.feed.generator` - Custom feed algorithms

### Jetstream Event Format
```json
{
  "did": "did:plc:xyz123",
  "time_us": 1699999999999999,
  "type": "com",
  "commit": {
    "operation": "create",
    "collection": "app.bsky.feed.repost",
    "rkey": "abc123",
    "record": {
      "subject": {
        "uri": "at://did:plc:author/app.bsky.feed.post/postid",
        "cid": "bafyrei..."
      },
      "createdAt": "2024-01-15T12:00:00.000Z"
    }
  }
}
```

---

*Document Version: 1.0*
*Last Updated: 2026-02-09*
*Author: [Your Name]*
*Supervisor: [Supervisor Name]*

"""Full BlueCCGL model assembly.

Pipeline:
    Input  →  R-GCN (or GCN) encoder  →  node embeddings
           →  DiffPool (or global mean pool)  →  multi-level graph embeddings
           →  Readout MLP (+ sentiment composition)  →  cascade embedding
           →  Projector MLP  →  projection (for InfoNCE during training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from blueccgl.models.diffpool import HierarchicalDiffPool
from blueccgl.models.gcn_encoder import GCNEncoder
from blueccgl.models.readout import ProjectorMLP, ReadoutMLP
from blueccgl.models.rgcn_encoder import RGCNEncoder


class BlueCCGL(nn.Module):
    """Complete BlueCCGL model.

    Supports config-driven switching between:
        - encoder: ``rgcn`` (default) or ``gcn`` (ablation A1)
        - pool:    ``diffpool`` (default) or ``mean`` (ablation A3)
        - use_sentiment: ``true`` (default) or ``false`` (ablation A4)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]

        input_dim = mcfg.get("input_dim", 6)
        hidden_dim = mcfg.get("hidden_dim", 128)
        embedding_dim = mcfg.get("embedding_dim", 64)
        self.use_sentiment = mcfg.get("use_sentiment", True)
        self.use_diffpool = mcfg.get("pool", "diffpool") == "diffpool"

        # ── Encoder ──────────────────────────────────────────────────────
        if mcfg.get("encoder", "rgcn") == "rgcn":
            self.encoder = RGCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_relations=3,
                num_bases=mcfg.get("rgcn_bases", 2),
                num_layers=mcfg.get("rgcn_layers", 3),
                dropout=mcfg.get("rgcn_dropout", 0.3),
            )
        else:
            self.encoder = GCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=mcfg.get("rgcn_layers", 3),
                dropout=mcfg.get("rgcn_dropout", 0.3),
            )

        # ── Pooling ──────────────────────────────────────────────────────
        if self.use_diffpool:
            self.pooler = HierarchicalDiffPool(
                hidden_dim=hidden_dim,
                pool_ratios=mcfg.get("pool_ratios", [0.5, 0.25]),
                max_nodes=mcfg.get("diffpool_max_nodes", 1000),
            )

        # ── Readout + Projector ──────────────────────────────────────────
        self.readout = ReadoutMLP(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            use_diffpool=self.use_diffpool,
            use_sentiment=self.use_sentiment,
        )
        self.projector = ProjectorMLP(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

    # ------------------------------------------------------------------

    def forward(
        self, batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: PyG Batch with ``x``, ``edge_index``, ``edge_type``,
                   ``batch``, ``sentiment_composition``.

        Returns:
            embedding:  [B, embedding_dim]  (for downstream / clustering).
            projection: [B, embedding_dim]  (for InfoNCE loss).
            aux_loss:   scalar              (DiffPool auxiliary; 0 for mean pool).
        """
        x = batch.x
        if not self.use_sentiment:
            # Zero out sentiment features (dims 1=sent_label, 2=sent_score)
            x = x.clone()
            x[:, 1] = 0.0
            x[:, 2] = 0.0

        # Encode
        h = self.encoder(x, batch.edge_index, batch.edge_type)

        # Pool
        if self.use_diffpool:
            pool_0, pool_1, pool_2, aux_loss = self.pooler(
                h, batch.edge_index, batch.batch,
            )
            pools = [pool_0, pool_1, pool_2]
        else:
            pool_0 = global_mean_pool(h, batch.batch)
            pools = [pool_0]
            aux_loss = torch.tensor(0.0, device=h.device)

        # Sentiment composition
        if self.use_sentiment:
            B = pool_0.shape[0]
            sent = batch.sentiment_composition.view(B, 3)
            pools.append(sent)

        # Readout → embedding → projection
        embedding = self.readout(pools)
        projection = self.projector(embedding)
        return embedding, projection, aux_loss

    def encode(self, batch) -> torch.Tensor:
        """Inference-only: return embeddings without projection or aux loss."""
        embedding, _, _ = self.forward(batch)
        return embedding

    def explain(self, batch) -> tuple:
        """XAI inference pass.

        Returns node embeddings, soft DiffPool assignment matrices, and the
        dense-batch padding mask alongside the cascade embedding.  Intended
        for single-cascade (batch_size=1) calls to avoid padding interference
        when interpreting assignment indices.

        Args:
            batch: PyG Batch (typically a single graph).

        Returns:
            embedding:  [B, embedding_dim]
            h:          [N_total, hidden_dim]  node embeddings after encoder.
            s1:         [B, N_max, k1] soft assignment Level-1, or None.
            s2:         [B, k1, k2]   soft assignment Level-2, or None.
            mask:       [B, N_max] bool padding mask, or None.
        """
        x = batch.x
        if not self.use_sentiment:
            x = x.clone()
            x[:, 1] = 0.0
            x[:, 2] = 0.0

        h = self.encoder(x, batch.edge_index, batch.edge_type)

        if self.use_diffpool:
            pool_0, pool_1, pool_2, _, s1, s2, mask = self.pooler(
                h, batch.edge_index, batch.batch, return_assignments=True,
            )
            pools = [pool_0, pool_1, pool_2]
        else:
            from torch_geometric.nn import global_mean_pool
            pool_0 = global_mean_pool(h, batch.batch)
            pools = [pool_0]
            s1, s2, mask = None, None, None

        if self.use_sentiment:
            B = pool_0.shape[0]
            sent = batch.sentiment_composition.view(B, 3)
            pools.append(sent)

        embedding = self.readout(pools)
        return embedding, h, s1, s2, mask

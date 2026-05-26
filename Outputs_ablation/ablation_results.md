# BlueCCGL Ablation Results — 2³ Factorial Grid

Baseline: **rgcn_feed_sent** (R-GCN + feed-aware augmentation + sentiment ON) = `Outputs_17_04`

Metrics: Silhouette (↑ better), Davies-Bouldin (↓ better), Calinski-Harabasz (↑ better), DBCV (↑ better).
Δ columns show change vs. baseline; ↑/↓ indicates direction of improvement.

| Variant | Encoder | Aug | Sent | Silhouette ↑ | Δ Sil | DB ↓ | Δ DB | CH ↑ | DBCV ↑ | Clusters | Noise % | Best Loss | Best Epoch | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **rgcn_feed_sent** | R-GCN | feed-aware | ✓ | 0.4133 | baseline | 0.6757 | baseline | 1978.7 | 0.6181 | 11 | 1.49 | 0.12921 | 147 | done |
| gcn_feed_sent | GCN | feed-aware | ✓ | — | — | — | — | — | — | — | — | — | — | missing |
| rgcn_rand_sent | R-GCN | random | ✓ | — | — | — | — | — | — | — | — | — | — | missing |
| rgcn_feed_nosent | R-GCN | feed-aware | ✗ | — | — | — | — | — | — | — | — | — | — | missing |
| gcn_rand_sent | GCN | random | ✓ | — | — | — | — | — | — | — | — | — | — | missing |
| gcn_feed_nosent | GCN | feed-aware | ✗ | — | — | — | — | — | — | — | — | — | — | missing |
| rgcn_rand_nosent | R-GCN | random | ✗ | — | — | — | — | — | — | — | — | — | — | missing |
| gcn_rand_nosent | GCN | random | ✗ | — | — | — | — | — | — | — | — | — | — | missing |

## Notes

- **Δ Sil** and **Δ DB** are computed relative to the full CCGL baseline.
- Davies-Bouldin: lower is better — a negative Δ (↑) means improvement.
- Silhouette: higher is better — a positive Δ (↑) means improvement.
- `aug_fallback_rate` in the CSV records the fraction of batches where
  augmentation fell back to the original graph (min-nodes constraint).
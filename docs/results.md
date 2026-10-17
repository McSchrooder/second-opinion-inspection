# Results

recovery@e = share of good items recovered when the threshold lets through a
fraction e of true defects. Test sets are the datasets' own splits; no reject-lane
simulation yet, so these numbers are optimistic upper bounds.

| dataset | category | model | good / defect | AUROC | recovery@0 | recovery@1% | recovery@5% | fit | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| VisA | candle | PatchCore (default) | 100 / 100 | 0.976 | 0.51 | 0.81 | 0.91 | 12 min | 36 |

Hardware: Radeon RX 7900 GRE, ROCm 6.4.2, torch 2.6, anomalib 2.7.0. Fit time is
dominated by coreset selection over 92k patches; untuned.

Reading: an AUROC of 0.976 recovers only half the good items if we allow zero
escapes. The gap between AUROC and recovery@0 is the whole problem.

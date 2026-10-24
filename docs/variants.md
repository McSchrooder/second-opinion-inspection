# PatchCore variants: resolution, training subset, backbone

Generated from `results/*.json` (recovery of goods at 1% escape, all test goods). Baseline is
wide_resnet50_2 at 256 px on the full training split. `n150`/`n300` = random 150/300 training
images (seeded). `r18` = resnet18 backbone at 256 px. The 256 n150 column is the control that
separates the cost of fewer training images from the effect of resolution.

| dataset | category | 256 all | 256 n150 | 320 n150 | 384 n150 | 320 n300 | 256 r18 | best |
|---|---|---|---|---|---|---|---|---|
| mvtec | bottle | 1.00 |  |  |  | 1.00 | 1.00 | 320 n300 (1.00) |
| mvtec | hazelnut | 1.00 |  |  |  | 1.00 | 0.95 | 320 n300 (1.00) |
| visa | candle | 0.66 | 0.69 | 0.75 | 0.67 | 0.68 | 0.67 | 320 n150 (0.75) |
| visa | capsules | 0.03 | 0.02 | 0.08 | 0.12 | 0.27 | 0.05 | 320 n300 (0.27) |
| visa | cashew | 0.66 | 0.56 | 0.78 | 0.66 | 0.82 | 0.64 | 320 n300 (0.82) |
| visa | chewinggum | 0.84 |  |  |  | 0.78 | 0.76 | 256 all (0.84) |
| visa | fryum | 0.56 | 0.28 | 0.36 | 0.54 | 0.36 | 0.32 | 256 all (0.56) |
| visa | macaroni1 | 0.24 | 0.24 | 0.33 | 0.39 | 0.23 | 0.53 | 256 r18 (0.53) |
| visa | macaroni2 | 0.11 | 0.05 | 0.06 | 0.07 | 0.12 | 0.23 | 256 r18 (0.23) |
| visa | pipe_fryum | 0.94 |  |  |  | 0.90 | 0.62 | 256 all (0.94) |

## Cost per variant (mean over categories)

| variant | ms/img | fit s | peak VRAM GiB | train imgs |
|---|---|---|---|---|
| 256 all | 27 | 313 |  | nan |
| 256 n150 | 13 | 23 | 1.85 | 150 |
| 320 n150 | 21 | 51 | 2.84 | 150 |
| 384 n150 | 38 | 108 | 4.05 | 150 |
| 320 n300 | 33 | 176 | 5.42 | 291 |
| 256 r18 | 14 | 311 | 2.32 | 564 |

## Reading

- **Resolution helps where the defect is small relative to the item**: cashew 0.66 -> 0.82 (320 n300),
  capsules 0.03 -> 0.27 (320 n300), macaroni1 0.24 -> 0.39 (384 n150), candle 0.66 -> 0.75 (320 n150).
  It does not rescue the multi-instance trays: macaroni2 and capsules stay below 0.3 at every setting.
- **Fewer training images cost real recovery on textured items**: the 256 n150 control drops cashew
  0.66 -> 0.56 and fryum 0.56 -> 0.28. Fryum's apparent loss at 320 px is this subset cost, not resolution.
  For a line this means: collect several hundred goods before trusting the thresholds.
- **Backbone is a per-category choice**: resnet18 doubles macaroni1 (0.24 -> 0.53) and macaroni2
  (0.11 -> 0.23) at half the latency (14 vs 27 ms/img), but loses on pipe_fryum (0.94 -> 0.62),
  chewinggum and hazelnut. There is no single best feature extractor. The per-line pilot picks it.
- **Memory**: anomalib keeps every training patch on the GPU and grows the bank with torch.cat, so peak
  VRAM is about twice the bank (train imgs x patches x 1536 x 4 B). 384 px with 300 images already
  exceeds 9.5 GiB. Keep resolution x images in budget or subsample patches, not images.
- Tiers hold: rigid objects (bottle, hazelnut) are solved at any setting. Textured single items land
  0.4-0.9 and move with resolution and data. Multi-instance trays need a different approach
  (per-instance crops or detection first), not a bigger PatchCore.

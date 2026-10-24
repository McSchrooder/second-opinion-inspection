# Second-opinion inspection: technical overview

What we are building, what we built it with, what we ran, and what came out.

## 1. The product in one paragraph

Food and beverage lines already have a primary vision camera. It is tuned to
never miss a defect, so it over-rejects: a sizeable share of what lands in the
reject bin is fine. We add a second opinion on that bin only. An anomaly model
scores every rejected item, hands back the ones it is sure are good, confirms
the ones it is sure are bad, shows a heatmap for the rest and sends them to a
technician. The product promise is one number pair per line: how many good
items we recover, at an escape rate (defects slipping through) we can defend
statistically.

## 2. Tooling

| Piece | Choice | Why |
|---|---|---|
| Library | anomalib 2.7.0 (Apache-2.0) | Reference implementations of 35 anomaly models behind one API, dataset loaders, export to ONNX/OpenVINO. We do not reimplement models. |
| Framework | PyTorch 2.6 + Lightning (anomalib's `Engine`) | Comes with anomalib. |
| Backbones | torchvision / timm ImageNet weights | Frozen feature extractors for the memory-bank models. |
| Metrics | scikit-learn + our own `recovery_at_escape` | anomalib's built-in AUROC sigmoid-saturates raw scores. Its `PGn` metric is the same quantity as ours but cannot run on the lane subset or 5-fold. |
| Data | VisA (CC BY 4.0), MVTec AD (non-commercial, research use) | Public, labelled, food-adjacent categories. |
| Hardware | one consumer AMD GPU, 16 GB, ROCm 6.4.2 under WSL2 | What we had. See caveats. |

Anomalib's design, in short: every model is a `LightningModule` with four
plug-in parts (pre-processor for resize/normalise, the torch model, a
post-processor that thresholds and normalises, an evaluator with metrics).
We keep the first two, switch off the post-processor (raw scores, our own
operating point) and the evaluator (our own metric on our own subset).

## 3. Every model anomalib 2.7 ships, and whether it applies to us

Legend. **Bank**: stores features of good images, no gradient training, minutes
to fit, stable on our GPU. **Trained**: gradient training for epochs or
thousands of steps, needs a stable GPU or rented time. **Zero-shot**: no training
images at all, language or foundation-model prompts.

### Applicable now (bank models, ran or can run on our box)

| Model | Idea | Status |
|---|---|---|
| PaDiM | One Gaussian per patch position over ImageNet features, Mahalanobis distance. | Ran, 10 categories. |
| PatchCore | Memory bank of all good patches, coreset-subsampled, nearest-neighbour distance. | Ran, 10 categories, 6 variants. Our engine. |
| MH-PatchCore | PatchCore with per-head PCA whitening and covariance shrinkage. | Candidate, same cost class. |
| DFKDE | Kernel density on PCA-reduced global features. Image-level only, no heatmap. | Low value: no heatmap for the technician. |
| DFM | PCA subspace of layer-3 features, feature reconstruction error. | Candidate, cheap. |
| RAD | Retrieval against a DINOv3 patch bank with positional bank. | Candidate, foundation features. |
| SuperADD | PatchCore on a DINOv3 ViT-H+ backbone at 448 px tiles. | Candidate, heavy backbone, likely best on textured items. |
| AnomalyDINO | PatchCore-style on DINOv2 patch features, few-shot capable. | Candidate, first to try next. |
| FoundAD | DINOv2 features plus a small trained predictor head, few-shot. | Candidate, light training. |

### Applicable with real training (rented GPU or a stable box)

| Model | Idea | Loss | Status |
|---|---|---|---|
| Reverse Distillation | Frozen teacher encoder, trained student decoder reconstructs teacher features from a bottleneck. | 1 minus cosine similarity between teacher and student feature maps, three scales. | Ran, 10 epochs, 10 categories. |
| EfficientAD | Small student copies a distilled teacher, autoencoder catches global defects. | Hard-mined MSE student vs teacher, MSE autoencoder vs teacher, MSE student vs autoencoder. | Started twice, GPU hung both times at about 40 min. Not evaluated. |
| STFPM | Student-teacher feature pyramid matching. | MSE on normalised features per layer. | Candidate, lighter than EfficientAD. |
| FastFlow | Normalising flow on backbone features. | Negative log-likelihood. | Candidate. |
| CFlow, CS-Flow, U-Flow, PatchFlow | Flow variants (conditional, cross-scale, U-shaped, patch). | Negative log-likelihood. | Candidates, slower. |
| DRAEM | Synthetic defects pasted from a texture dataset, reconstruction plus segmentation nets. | MSE + SSIM + focal segmentation loss. | Candidate, needs DTD download. |
| DSR | Quantised reconstruction with synthetic anomalies in latent space. | Two-stage reconstruction and focal losses. | Candidate, complex. |
| GLASS | Gradient-ascent synthetic anomalies in feature space plus a discriminator. | Discriminator BCE. | Candidate, strong on benchmarks. |
| SuperSimpleNet | Feature adaptor plus segmentation/classification heads on Perlin-noise anomalies. | Focal + truncated L1. | Candidate, fast inference. |
| Dinomaly | DINOv2 encoder, trained decoder reconstructs features. | Cosine distance, hard-mined. | Candidate, current benchmark leader among trained models. |
| INP-Former | Dinomaly-style with learned normal prototypes. | Cosine plus prototype coherence. | Candidate. |
| UniNet | Student-teacher with contrastive temperature. | UniNet contrastive loss. | Candidate. |
| GeneralAD | DINOv2 large features plus trained discriminator on perturbed features. | Discriminator loss. | Candidate, heavy. |
| CFA | Coupled-hypersphere feature adaptation on a memory bank. | CFA hypersphere loss. | Candidate. |
| FRE | Shallow autoencoder on layer-3 features. | MSE. | Candidate, cheap. |
| L2BT | Learn-to-blur-and-sharpen teacher/student. | Reconstruction. | Candidate, niche. |

### Zero-shot and language

| Model | Idea | Status |
|---|---|---|
| WinCLIP | CLIP patches against "normal / damaged" text prompts, windowed, optional few-shot. | Candidate: puts a number on "VLM as scorer". |
| AnomalyVFM | Vision foundation model turned zero-shot detector. | Candidate. |
| VlmAd | Ask a vision-language model (LLaVA via Ollama, GPT-4V) per image. | Only as explanation layer, not a scorer: near chance on scoring, slow. |

### Not applicable

| Model | Why not |
|---|---|
| AI-VAD, FUVAS | Video anomaly detection. |
| CFM | Needs RGB plus 3D point clouds. |
| GANomaly | 2018 GAN reconstruction, weak and unstable. |

## 4. What we ran

### Data

| Dataset | Categories | Train goods | Test goods / defects |
|---|---|---|---|
| MVTec AD | bottle, hazelnut | 209, 391 | 20/63, 40/70 |
| VisA | candle, cashew, chewinggum, fryum, pipe_fryum, macaroni1, macaroni2, capsules | 900 to 1000 (VisA 1cls split) | 50 to 100 / 100 each |

### Scripts

| Script | Does |
|---|---|
| `download.py` | Fetches VisA and MVTec (HF mirror fallback for MVTec). |
| `run.py` | One model on one category. Raw scores, per-image CSV with anomaly-map statistics, JSON with AUROC, recovery at 0/1/5% escape, latency, peak VRAM. Variants: `--size`, `--backbone`, `--train-n`, `--coreset`. |
| `standin.py` | The two fake primary cameras: golden-template difference and PCA reconstruction error, 128 px grayscale. |
| `lane.py` | Builds the simulated reject lane, computes recovery at fixed escape on it, the 5-fold calibrated two-threshold band, the Clopper-Pearson escape bound. Writes `results.md` and base figures. |
| `classify.py` | Second-stage gradient-boosted trees on score plus map statistics, 5-fold, versus the raw score. |
| `grid.sh` | Runs everything with a GPU probe before each job, a VRAM cap and a wall-clock cap. Stops on GPU failure. |
| `figures.py` | Heatmap, resolution effect, speed vs recovery. |
| `export_items.py` | Per-item score files in the shared evaluation layout. |

### Settings

- Input 256 px (variants 320 and 384), batch 16, ImageNet normalisation, no augmentation.
- PatchCore: wide_resnet50_2 layers 2+3, 10% coreset, 9 neighbours. Variant: resnet18.
- PaDiM: resnet18 layers 1 to 3, 100 random channels.
- Reverse Distillation: wide_resnet50_2, 10 epochs, Adam, `MIOPEN_FIND_MODE=2` on ROCm.
- EfficientAD-S: batch 1, 5000 steps cap, ImageNette as the penalty set. Did not finish.
- Lane: stand-in threshold at the lowest defect score, so it catches every defect. It then flags 72 to 100% of goods.
- Operating point: `t_recover` lets at most 1% of calibration defects through. `t_confirm` keeps at most 2% of calibration goods above it. Calibrated on 4 folds, evaluated on the 5th, seed 0.
- Escape bound: 95% Clopper-Pearson upper limit with zero observed escapes on the calibration defects, roughly 3/n.

### Losses, plainly

PaDiM and PatchCore have no loss: nothing is optimised, features are stored.
Reverse Distillation minimises one minus the cosine similarity between the
teacher's and the student's feature maps at three scales. On good images the
student learns to copy, on defects it cannot. EfficientAD minimises three
squared errors: student against teacher on the hardest 0.1% of positions,
autoencoder against teacher, and student against autoencoder, with an ImageNet
penalty so the student does not generalise beyond the product.

## 5. Results

Recovery of good items at 1% escape, on the simulated reject lane (template
stand-in), thresholds held out. Full tables in `results.md`, variants in
`variants.md`, figures in `figures/`.

| Tier | Category | PaDiM | PatchCore | Rev. Dist. | PatchCore 320 | PatchCore r18 |
|---|---|---|---|---|---|---|
| rigid object | bottle | 0.95 | 1.00 | 0.95 | 1.00 | 1.00 |
| | hazelnut | 0.16 | 1.00 | 1.00 | 1.00 | 0.95 |
| textured single item | pipe_fryum | 0.38 | 0.94 | 0.86 | 0.90 | 0.62 |
| | chewinggum | 0.54 | 0.84 | 0.72 | 0.78 | 0.76 |
| | cashew | 0.06 | 0.61 | 0.33 | 0.75 | 0.58 |
| | candle | 0.55 | 0.65 | 0.65 | 0.66 | 0.66 |
| | fryum | 0.33 | 0.54 | 0.26 | 0.33 | 0.30 |
| multi-instance tray | macaroni1 | 0.27 | 0.25 | 0.47 | 0.24 | 0.52 |
| | capsules | 0.02 | 0.04 | 0.11 | 0.27 | 0.05 |
| | macaroni2 | 0.13 | 0.11 | 0.02 | 0.11 | 0.22 |
| mean | | 0.34 | 0.60 | 0.54 | 0.60 | 0.57 |
| ms per image | | 10 | 27 | 127 | 33 | 14 |

Findings.

1. Three tiers, set by the product, not the model. Rigid single objects are
   solved at any setting and could run autonomously. Textured single items land
   0.3 to 0.94 and move with model, resolution and data. Multi-instance trays
   stay below 0.52 for every model: one score per tray hides one bad piece among
   many good ones. They need per-piece crops or a detector in front.
2. PatchCore is the engine: best or tied on 7 of 10 categories at 27 ms.
   Reverse Distillation only wins on macaroni1 and costs 5x the latency.
3. Resolution helps where the defect is small on the item (cashew 0.61 to
   0.75, capsules 0.04 to 0.27). It does not fix trays.
4. Training-set size matters more than expected: 150 instead of all good images
   drops cashew 0.61 to 0.50 and fryum 0.54 to 0.24. A line needs several
   hundred goods before thresholds are trusted.
5. Backbone is a per-line choice: resnet18 doubles macaroni1 and halves latency
   but loses pipe_fryum (0.94 to 0.62).
6. Held-out escape at the 1% target lands at 1 to 3%. The 95% upper bound with
   our defect counts is 4 to 6%. Under 1% is not claimable from these datasets.
7. A second-stage classifier on anomaly-map statistics does not beat the raw
   score with 40 to 90 lane goods per category. Revisit with real line labels.
8. Stand-ins agree: template and PCA lanes give the same recovery within 0.03.

## 6. Caveats

- The reject lane is simulated. The stand-ins are deliberately dumb. A real
  primary camera may reject differently and the lane composition is what sets
  recovery. Real line data replaces this first.
- Public datasets are not the line. VisA and MVTec defects are curated and
  centred. Lighting and handling on a line vary more.
- Escape is bounded, not zero. With 60 to 100 test defects we can show "under
  about 5% with 95% confidence", not "under 1%". Technician confirmation of
  every recovery stays mandatory until a line delivers hundreds of defects.
- Thresholds in the plain `all@e` and `lane@e` columns are set on the test
  defects themselves (optimistic by construction). The calibrated band table
  is the held-out version and is the one to quote.
- anomalib's built-in AUROC and F1 threshold are unsuitable on raw scores
  (sigmoid saturation, symmetric cost). We bypass both.
- PatchCore keeps every training patch in VRAM and peaks at twice the bank
  size. 384 px with 300 images already exceeds 9.5 GB. Resolution times images
  is a budget.
- Our GPU box hangs under hours of continuous training (pre-existing fault,
  worsened by WSL). EfficientAD and anything in the trained list runs on a
  rented GPU or after the box is fixed.
- AUROC is reported for reference only. It does not predict recovery at low
  escape (see `figures/auroc_vs_recovery.png`).

## 7. Next

- Load the pilot line's own images (anomalib `Folder` loader) and rerun the
  same pipeline. That replaces the simulated lane with the real bin.
- AnomalyDINO and WinCLIP on the ten categories: foundation features vs ResNet,
  and a number for the language-model route.
- Per-piece cropping for tray categories before any further model work there.
- EfficientAD at 2000 steps on a stable GPU for the edge-latency argument.

# Models and data for a reject-lane second-opinion inspector

Research snapshot, 2026-10-17. Scope: anomaly-detection models, VLMs, metrics, benchmark construction, datasets, edge deployment and heatmap explainability for re-checking items that an inline machine-vision system has already rejected. Every number carries a source tag `[Sx]` that resolves in the Sources table (section 9). "UNVERIFIED" marks claims that could not be confirmed from a primary source. Blank table cells mean no published number was found; nothing was guessed.

---

## 1. anomalib (open-edge-platform/anomalib)

### 1.1 Version, license, packaging

| Item | Value | Source |
|---|---|---|
| Current stable | `lib/v2.7.0`, published 2026-10-06 | [S1] |
| Recent cadence | v2.4.0 (2026-04-24), v2.5.0 (2026-05-29), v2.5.1 (2026-07-17), v2.6.0 (2026-07-25), v2.6.1 (2026-09-04), v2.6.2 (2026-09-11), v2.7.0 (2026-10-06) | [S1] |
| v2.0.0 | Released 19 Mar (2025 per release sequence; year not printed on page) | [S3] |
| License (code) | Apache-2.0 | [S2] |
| Minimum PyTorch | 2.10 for CPU/CUDA/XPU extras, 2.13 for ROCm (since v2.7.0) | [S1] |
| Extras | `[openvino]`, `[clip]` (WinCLIP), `cpu`, `cu130`, `xpu` | [S2] |
| Studio app | "Anomalib Studio" low/no-code web app (USB/IP camera or folder in; MQTT/ROS out; Docker or standalone). Marked pre-release. | [S2] |

### 1.2 Model zoo (image models in `src/anomalib/models/image`, 33 directories, plus video `ai_vad`, `fuvas`)

Speed column is relative and qualitative, from the papers cited in section 2 and section 7 where available; "?" means no published latency found. "Good-only" = trains on defect-free images only.

| Model | Family | Good-only | Zero/few-shot | Relative speed |
|---|---|---|---|---|
| AnomalyDINO | Memory bank on DINOv2 patches | Yes (few normals) | Few-shot (1-16 normal shots) | Medium (~60 ms/img A40 at 448 px [S20]) |
| AnomalyVFM | Foundation model + LoRA trained on FLUX-synthesised anomalies | n/a (no target data) | Zero-shot (anomalib impl.); paper also reports 1/2/4-shot | Fast for its size (20.5 ms A100 [S15]) |
| CFA | Memory bank / hypersphere feature adaptation | Yes | No | Medium |
| CFM | Cross-modal (RGB + 3D) feature mapping | Yes | No | Needs 3D input |
| C-Flow (CFLOW-AD) | Conditional normalizing flow | Yes | No | Medium (27 fps, 256 px, GTX 1080 [S11]) |
| CS-Flow | Normalizing flow, cross-scale | Yes | No | Medium |
| DFKDE | Deep-feature KDE (shallow) | Yes | No | Fast |
| DFM | Deep-feature PCA/Gaussian (shallow) | Yes | No | Fast |
| Dinomaly | Reconstruction of ViT (DINOv2) features, multi-class | Yes | No | Medium (58 img/s ViT-B, RTX 3090, bs 16 [S9]) |
| DRAEM | Reconstruction + discriminative segmentation, synthetic anomalies | Yes (synthesises anomalies) | No | Medium |
| DSR | Dual subspace re-projection | Yes | No | Fast (17 ms A6000 [S8]) |
| EfficientAD | Student-teacher + autoencoder | Yes | No | Fastest (2.2-4.5 ms A6000 [S8]) |
| FastFlow | 2D normalizing flow | Yes | No | Fast (17 ms A6000 [S8]) |
| FoundAD | Foundation encoder few-shot detector (new in v2.7.0) | Yes (few normals) | Few-shot | ? |
| FRE | Feature reconstruction error (shallow) | Yes | No | Fast |
| GANomaly | GAN reconstruction | Yes | No | Fast, weak |
| GeneralAD | Cross-domain, attends to distorted features | Yes | No | ? |
| GLASS | Gradient-ascent anomaly synthesis | Yes | No | ? |
| INP-Former | Intrinsic normal prototypes from a single image | Yes | Few-shot-friendly | ? |
| L2BT | Transformer feature regression | Yes | No | ? |
| MH-PatchCore | Covariance-aware, streaming PatchCore variant (new in v2.7.0) | Yes | No | ? |
| PaDiM | Per-patch Gaussian (memory/statistics) | Yes | No | Fast to train; 0.19 s/img GPU [S7] |
| PatchCore | Memory bank, kNN patch distance | Yes | Few-shot capable (degrades) | Medium (32 ms A6000 [S8]; 0.17-0.22 s V100-class [S7]) |
| PatchFlow | Flow on patch features | Yes | No | ? |
| RAD | Retrieval-based, no task-specific training (new in v2.7.0) | Yes (reference images) | Few-shot / training-free | ? |
| Reverse Distillation | Teacher-student (reverse) distillation | Yes | No | Medium |
| STFPM | Student-teacher feature pyramid | Yes | No | Fast |
| SuperADD | Token-level ViT AD (VAND 4.0 winner, v2.6.0) | Yes | No | ? |
| SuperSimpleNet | Feature-noise discriminator; unsupervised to supervised | Yes, can also use labelled defects | No | Fast |
| U-Flow | U-shaped flow, unsupervised threshold | Yes | No | Medium |
| UniNet | Contrastive student-teacher, feature selection (CVPR 2025) | Yes (supervised mode optional) | No | ? |
| VLM-AD | Prompts a VLM (Ollama default, OpenAI, Hugging Face); output is a yes/no label parsed from the first letter plus `batch.explanation` text; no score, no anomaly map, export skipped [S34] | n/a | Zero-shot, or k normal reference images | Slow (VLM call per image) |
| WinCLIP | CLIP window-text similarity | n/a | Zero- and few-shot | Medium |

Source for the list: [S4] (directory listing), descriptions from [S5]; new-model notes from [S1]. RKDE was removed in v2.0.0 [S3].

### 1.3 Datamodules (`src/anomalib/data/datamodules/image`)

AutoVI, BMAD, BTech (BTAD), Datumaro (import from Intel Geti / Datumaro exports), Folder (custom data), Kaputt, Kolektor (KSDD), MPDD, MVTec LOCO, MVTecAD, MVTecAD2, Real-IAD, Tabular (csv/json/parquet listings), VAD (Valeo), VisA [S6]. Depth/3D (`depth/`): ADAM-3D, Folder3D, MVTec 3D-AD. Video (`video/`): Avenue, ShanghaiTech, UCSD Ped [S6].

### 1.4 Export, inference, benchmarking

| Capability | Status | Source |
|---|---|---|
| Export types | `ExportType`: `onnx`, `openvino`, `torch` (TorchScript/torch export) | [S10] |
| Compression | OpenVINO only: `fp16`, `int8`, `int8_ptq`, `int8_acq` (NNCF-based PTQ and accuracy-control quantisation) | [S10] |
| ONNX path | Legacy `dynamo=False` exporter removed in v2.7.0; dynamo-based export is the only path | [S1][S2] |
| TensorRT | No native exporter. Path is ONNX export then `trtexec` / TensorRT builder outside anomalib | [S10] (absence) |
| Inferencers | `TorchInferencer`, `OpenVINOInferencer` (accept NumPy, PIL, tensors since v2.0) | [S12][S3] |
| Inference scripts | `tools/inference/{torch,lightning,openvino,gradio}_inference.py` | [S13] |
| Benchmarking | `anomalib benchmark --config tools/experimental/benchmarking/sample.yaml`: grid over models x datasets x categories, one parallel job per CUDA device, CSV/W&B/TensorBoard logging of metrics and throughput | [S14] |
| Metrics shipped | AUROC, AUPR, AUPRO, PRO, F1Score, F1Max, AUPIMO (pimo/), anomaly-score distribution, PGn/PBn (`pg_pb.py`), F1AdaptiveThreshold, ManualThreshold | [S16] |

### 1.5 v1 to v2 gotchas

- `task` argument removed (v2.0.0, PR #2450); models produce image score and anomaly map together. Old configs with `task: segmentation` fail [S3].
- `AnomalyModule` renamed `AnomalibModule`; `BaseThreshold` renamed `Threshold`; `trainer.model` is now `trainer.lightning_module` [S3].
- Pre/post-processing moved into the model: `PreProcessor` (resize, normalise transforms) and `PostProcessor` (normalisation + thresholding) are model components; Engine-level threshold/normalisation args from v1 are gone [S3][S17].
- Metrics are configured through `Evaluator(val_metrics=..., test_metrics=...)` with explicit `fields=["pred_score","gt_label"]`; `F1Score` cannot be a validation metric because `pred_label` only exists after thresholds are computed, use `F1Max` [S18].
- Outputs are dataclasses (`ImageBatch`, `InferenceBatch`) with fields `pred_score`, `pred_label`, `anomaly_map`, `pred_mask` [S17].
- `imgaug` replaced by torchvision v2 transforms [S3].
- v2.5.1: DINOv2-based models moved to timm feature extraction; custom ViT removed; `cu118` extra dropped [S1].
- v2.7.0: checkpoints load with `weights_only=True`; dataset metadata paths must resolve under `root`; pre-timm ViT checkpoint migration removed. Old checkpoints may refuse to load [S1].
- Threshold leakage default: `MVTecAD(val_split_mode=SAME_AS_TEST, val_split_ratio=0.5)` [S19]. The adaptive threshold and min-max normalisation are fitted on test data unless changed. For our metric this must be overridden (section 4).
- Normalised scores: PostProcessor maps the learned threshold to 0.5 and clamps to [0,1] using min/max seen on validation [S17]. Clamping destroys ordering above max and below min, so compute operating points on raw `pred_score` (`enable_normalization=False`) or on unclamped values.

---

## 2. Model map

Conventions: I-AUROC = image AUROC; P-AUROC = pixel AUROC; AU-PRO at FPR 0.30 unless noted; MVTec AD 2 uses AU-PRO at FPR 0.05 on `test_private` / `test_private_mixed` (no image AUROC is published for AD 2). "Multi-class" = one model for all categories; otherwise one model per category. Few-shot/zero-shot numbers are not comparable to full-shot numbers.

### 2.1 Memory bank

| Model | MVTec AD I-AUROC / P-AUROC / AU-PRO | VisA I-AUROC / AU-PRO | MVTec AD 2 AU-PRO0.05 priv / mixed | Params | Published latency | Few-shot | In anomalib | Paper |
|---|---|---|---|---|---|---|---|---|
| PatchCore (WRN50, 25% coreset) | 99.1 / 98.1 / 93.4 [S7] | 94.3 [S8]; 94.7 / 91.8 [S9]; 90.3 (realistic protocol) [S21] | 28.8 / 26.0 [S22] | Backbone WRN50-2 (layers 2-3) + memory bank | 32 ms bs1, RTX A6000 [S8]; 0.17-0.22 s/img GPU [S7]; ~2 s/img at full AD 2 resolution [S22] | 1-shot I-AUROC 84.1, 50-shot 97.7 [S7] | Yes | arXiv 2106.08265 |
| PaDiM (WRN50) | 97.9 / 97.5 / 92.1 (as cited in [S11]) | 87.8 (cls AUROC), seg AUROC 93.1 [S23] | | Model 3.8-5.2 GB (WRN50 covariances) [S11] | 0.19 s/img GPU [S7]; 1.1 fps on i7 CPU (cited in [S11]) | 1-shot 76.1 [S7] | Yes | arXiv 2011.08785 |
| AnomalyDINO-S (few-shot) | 1-shot 96.5 / PRO 91.7; 16-shot 98.3 / 92.9 (448 px) [S20] | 1-shot 85.6 / 90.7; 16-shot 93.8 / 93.8 [S20] | | DINOv2 ViT-S ~21M [S20] | ~60 ms/img, A40, 1-shot [S20] | Yes | Yes | arXiv 2405.14529 |

### 2.2 Student-teacher / distillation

| Model | MVTec AD I-AUROC / P-AUROC / AU-PRO | VisA I-AUROC | MVTec AD 2 AU-PRO0.05 priv / mixed | Params | Published latency | Few-shot | In anomalib | Paper |
|---|---|---|---|---|---|---|---|---|
| EfficientAD-S | 98.8 / - / (92.5 avg over AD+VisA+LOCO) [S8] | 97.5 [S8] | | | 2.2 ms bs1, 614 img/s bs16, RTX A6000 [S8] | No | Yes | arXiv 2303.14535 |
| EfficientAD-M | 99.1 / - / (93.3 avg over AD+VisA+LOCO) [S8] | 98.1 [S8] | 30.8 / 19.2 ("EfficientAD", variant not stated) [S22] | | 4.5 ms bs1, 269 img/s, RTX A6000 [S8] | No | Yes | arXiv 2303.14535 |
| Reverse Distillation (RD4AD) | 98.5 / 97.8 / 93.9 [S24] | 95.0 (realistic protocol) [S21] | 26.4 / 25.0 (RD++: 27.1 / 25.2) [S22] | | 0.31 s/img, Intel i7 CPU [S24] | No | Yes | arXiv 2201.10703 |
| UniNet | anomalib re-implementation: 95.6 / 97.6 / - [S25]; paper numbers UNVERIFIED (tables only as images) | UNVERIFIED | | | | No | Yes | CVPR 2025 [S26] |

### 2.3 Flow / reconstruction

| Model | MVTec AD I-AUROC / P-AUROC / AU-PRO | VisA I-AUROC / P-AUROC / AU-PRO | MVTec AD 2 AU-PRO0.05 priv / mixed, or Real-IAD | Params | Published latency | Few-shot | In anomalib | Paper |
|---|---|---|---|---|---|---|---|---|
| FastFlow (WRN50) | 99.3 / 98.1 / - [S27] | 93.9 / - / - [S8] | | +41.3M over backbone (R18: +4.9M) [S27] | 17 ms RTX A6000 [S8]; 21.8 fps (WRN50) [S27] | No | Yes | arXiv 2111.07677 |
| CFLOW-AD (WRN50) | 98.26 / 98.62 / 94.60 [S11] | | | 947 MB model (R18: 96 MB) [S11] | 27 fps 256 px, 9 fps 512 px, GTX 1080 [S11] | No | Yes | arXiv 2107.12571 |
| Dinomaly (ViT-B, multi-class) | 99.6 / 98.4 / 94.8 [S9] | 98.7 / 98.7 / 94.5 [S9] | Real-IAD 89.3 / 98.8 / 93.9 [S9] | 148M (ViT-S 37.4M, ViT-L 275.3M) [S9] | 58.1 img/s bs16 RTX 3090 at 392 px (ViT-S 153.6) [S9] | No | Yes | arXiv 2405.14325 |
| DRAEM | 98.0 / 97.3 / - (pixel AP 68.4) [S28] | 89.0 (realistic protocol) [S21] | | | | No | Yes | arXiv 2108.07610 |
| MSFlow (reference only) | | 90.7 (realistic protocol) [S21] | 24.3 / 11.9 [S22] | | | No | No | |

### 2.4 Zero/few-shot, CLIP or foundation-model based

Zero-shot protocol: auxiliary training on MVTec AD when testing VisA and vice versa (AnomalyCLIP, AdaCLIP, April-GAN). WinCLIP and MuSc use no auxiliary training.

| Model | MVTec AD I-AUROC / P-AUROC / AU-PRO | VisA I-AUROC / P-AUROC / AU-PRO | Params / backbone | Published latency | Few-shot | In anomalib | Paper |
|---|---|---|---|---|---|---|---|
| WinCLIP (0-shot) | 91.8 / 85.1 / 64.6 [S29] | 78.1 / 79.6 / 56.8 [S29] | CLIP ViT-B/16+ | | Yes (4-shot MVTec 95.2 / 96.2 / 89.0 [S30]) | Yes | arXiv 2303.14814 |
| April-GAN / VAND (0-shot) | 86.1 / 87.6 / 44.0 [S30] | 78.0 / 94.2 / 86.8 [S30] | CLIP ViT-L/14@336 | | 4-shot MVTec 92.8 / 95.9 / 91.8; VisA 92.6 / 96.2 / 90.2 [S30] | No | arXiv 2305.17382 |
| AnomalyCLIP | 91.5 / 91.1 / 81.4 [S29] | 82.1 / 95.5 / 87.0 [S29] | CLIP ViT-L/14@336, frozen | | No | No | arXiv 2310.18961 |
| AdaCLIP | 89.2 / 88.7 / - (max-F1 90.6 / 43.4) [S31] | 85.8 / 95.5 / - [S31] | CLIP ViT-L/14@336 | 82.4 ms A100 (as reported in [S15]) | No | No | arXiv 2407.15795 |
| MuSc (0-shot, transductive) | 97.8 / 97.3 / 93.8 [S32] | 92.8 / 98.8 / 92.7 [S32] | CLIP ViT-L/14@336 | 270 ms/img RTX 3090 at 336 px, 955 ms at 518 px [S32] | Needs a batch (~200) of unlabeled test images scored jointly | No | ICLR 2024 |
| AnomalyVFM (0-shot) | 94.9 / 92.7 / 87.06 (AU-PRO from anomalib impl. [S33]) [S15] | 93.6 / 96.2 / - [S15] | RADIOv2.5 ViT-L, 345.8M total, 35.4M LoRA-trainable, 768 px [S15] | 20.5 ms A100 [S15] | 1/2/4-shot MVTec 97.2/97.9/98.2 [S15] | Yes | arXiv 2601.20524 (CVPR 2026 per anomalib [S33]) |
| Bayes-PFL, FAPrompt, AA-CLIP (2025) | 9-dataset industrial average I-AUROC 90.8 / 88.1 / 86.1 [S15] | | | Bayes-PFL 208.5 ms A100 [S15] | | No | via [S15] |
| AD-DINOv3, AVA-DINO (2025-2026) | numbers UNVERIFIED | | DINOv3 | | | No | arXiv 2509.14084 |

Reading of the table:
- Full-shot memory-bank and distillation models still beat every zero-shot model by 5-15 points I-AUROC on MVTec AD and VisA. Zero-shot is useful for cold start (no normal images yet), not as the production scorer.
- On a realistic protocol (no test-set early stopping, validation split, PG2) ranking changes: on VisA RD4AD leads (I-AUROC 95.0, PG2 67.4) while PatchCore drops to 90.3 / 56.3; on the Valeo VAD dataset PatchCore has best AUROC (88.0) but PG2 of only 16.5 vs MMR 27.6 [S21]. AUROC rank does not predict our metric's rank.
- MVTec AD 2 is far from solved: best AU-PRO0.05 is ~31% and lighting shift (`test_private_mixed`) halves EfficientAD and MSFlow localisation while RD loses only 1.4 points [S22].

---

## 3. Small vision-language models as the second opinion

### 3.1 Gemma 4 facts (verified)

Apache-2.0. Sizes: E2B (2.3B effective / 5.1B with embeddings), E4B (4.5B effective / 8B), 12B "Unified" (encoder-free), 26B A4B MoE (3.8B active / 25.2B), 31B dense. Vision encoder ~150M (E2B/E4B) or ~550M (26B/31B). Image token budget is selectable: 70, 140, 280, 560 or 1120 tokens per image. Context 128K (E2B/E4B) or 256K. Card lists object detection and pointing as image skills [S40]. Launch around 2026-04-02 (secondary source [S41]; the card carries no date). No Gemma 4 result on any industrial anomaly benchmark was found.

### 3.2 (a) Published anomaly results for small VLMs

MMAD (ICLR 2025) is the standard VLM benchmark: 39,672 multiple-choice questions over 8,366 images (MVTec AD, VisA, MVTec LOCO, GoodsAD, ...), 7 subtasks; anomaly discrimination (AD) is yes/no, so 50% is chance. Expert humans 95.24 AD / 86.65 average; GPT-4o 68.63 / 74.92 [S42]. The same model's MMAD score varies by up to ~9 points across papers because of prompt/protocol differences, so treat cross-paper comparisons as rough.

| Model | Size | Setting | MMAD AD acc. | MMAD avg | Source |
|---|---|---|---|---|---|
| Qwen3-VL | 2B | 0-shot | 49.80 | 67.83 | [S43] |
| Qwen2.5-VL | 3B | 0-shot | 60.63 | 67.77 | [S43] |
| Qwen3-VL | 4B | 0-shot | 65.87 | 71.17 | [S43] |
| Qwen2.5-VL | 7B | 0-shot | 60.16 | 68.82 | [S43] |
| Qwen3-VL | 8B | 0-shot | 67.20 | 72.12 | [S43] |
| Gemma 3 | 12B | 1-shot | 51.60 | 68.36 | [S44] |
| AnomalyGPT | 7B | 1-shot | 65.57 | 36.52 | [S42] |
| AnomalyR1 (GRPO on Qwen2.5-VL-3B) | 3B | 1-shot | 60.62 | 76.96 self-reported; 74.64 in [S44] | [S45] |
| EMIT (InternVL3-8B + GRPO) | 8B | 1-shot | 73.87 | 81.95 | [S44] |
| ADOPD (Qwen3-VL-4B, distilled) | 4B | 0-shot | 88.44 | 77.31 | [S43] |
| Anomaly-LR (Qwen2.5-VL) | 3B / 7B | fine-tuned | 71.66 / 75.39 | 86.00 / 87.12 | [S46] |

IAD-R1 (AAAI-26), balanced accuracy, base then RL-trained: Qwen2.5-VL-3B MVTec 62.6 to 77.6, VisA 58.4 to 69.8; Qwen2.5-VL-7B MVTec 66.0 to 81.9 [S47].

VLM papers reporting AUROC (7B-class unless noted):

| Model | Design | Setting | MVTec I-AUROC | VisA I-AUROC | Source |
|---|---|---|---|---|---|
| AnomalyGPT | Vicuna-7B + ImageBind + feature-matching decoder, 224 px | full-shot / 1-shot | 97.4 (P-AUROC 93.1) / 94.1 | - / 87.4 | [S48] |
| Myriad | MiniGPT-4 + expert anomaly map as prompt | 1-shot | 94.1 | 87.4 (identical to its expert) | [S49] |
| VMAD | LLaVA-7B + patch-similarity cues | 0-shot | 95.8 | 89.7 | [S50] |
| Anomaly-OV | LLaVA-OneVision 0.5B / 7B | 0-shot | 94.0 | 91.1 | [S51] |
| LogicAD | VLM text features | MVTec LOCO | 86.0 (LOCO) | - | [S52] |
| GPT-4V | general | 0-shot | 77.1 | 88.0 | [S53] |

PaliGemma 2, Florence-2, SmolVLM/SmolVLM2, Moondream 2/3: no published MVTec AD, VisA, MMAD or Real-IAD results found. A Florence-2-based industrial model (InspectVLM) reports it "fails to match traditional ResNet-based models" [S54]. No VLM results on MVTec AD 2 or on food/beverage data were found.

### 3.3 (b) Latency: prefill dominates

A verdict is 1-5 output tokens, so speculative decoding (EAGLE-3, multi-token prediction) only shortens a few decode steps; vision encoding and prefill of ~260-1,400 image tokens set the floor.

Visual tokens per image: Gemma 4 = user-selected 70-1120 [S40]; Qwen2.5-VL = one token per 28x28 px (about 324 at 512 px, 1,369 at 1024 px, derived) [S55]; Qwen3-VL = one token per 32x32 px (256 at 512 px, 1,024 at 1024 px, derived) [S56]; PaliGemma 2 = fixed 256 / 1,024 / 4,096 tokens at 224 / 448 / 896 px [S57]; SmolVLM = 64 tokens per 512 px tile plus global image [S58].

Measured, NVIDIA TensorRT-Edge-LLM v0.10 dashboard, batch 1, ~263-266 image tokens (COCO images, resolution not stated), ViT and prefill reported as separate columns [S59]:

| Model, precision | Device | ViT ms | Prefill ms | Decode tok/s |
|---|---|---|---|---|
| Qwen3-VL-2B, INT4 AWQ | Orin Nano 8GB | 141.3 | 167.2 | 36.1 |
| Qwen3-VL-2B, INT4 AWQ | Orin NX 16GB | 77.1 | 93.5 | 58.5 |
| Qwen3-VL-4B, INT4 AWQ | Orin NX 16GB | 77.4 | 217.2 | 31.7 |
| Qwen3-VL-8B, INT4 AWQ | Orin NX 16GB | 110.5 | 400.5 | 18.4 |
| Qwen3-VL-4B, INT4 AWQ | AGX Orin 64GB | 36.5 | 105.7 | 54.7 |
| Gemma 4 E4B, FP16 / NVFP4 | Jetson Thor | 16.7 / 16.6 | 61.0 / 36.1 | 25.4 / 59.3 |
| Gemma 4 E4B, INT4 AWQ | DGX Spark | 21.1 | 47.8 | 60.7 |
| Gemma 4 E2B, NVFP4 | Jetson Thor | 16.7 | 24.8 | 101.4 |

No Gemma 4 rows exist for Orin, and no published ViT/time-to-first-token numbers for any of these VLMs on an RTX 4090 were found (UNVERIFIED). Derived estimate for one verdict at ~265 tokens: Orin Nano + Qwen3-VL-2B about 0.3-0.45 s; Orin NX + 4B about 0.3-0.45 s; Thor/Spark + Gemma 4 E4B under 0.15 s. At 1024 px, token counts rise ~4x and prefill scales at least linearly. Sub-second on a 4090 for E4B at a few hundred image tokens is plausible, but the speed comes from a small token budget, not from speculative decoding, and a small token budget is exactly what loses fine defects. A specialist detector does the same job in 2-30 ms (section 2).

### 3.4 (c) Localisation and hybrids

- Built-in grounding exists: Florence-2 boxes/polygons/referring segmentation, PaliGemma 2 location (0-1023) and mask tokens, Qwen2.5-VL JSON boxes/points, Gemma 4 detection/pointing, Moondream detect/point [S40][S55][S57][S60]. None of these produce a dense anomaly heatmap.
- On real defects the grounding is poor: MMAD-BBox, 1-shot, Qwen2.5-VL-7B mIoU 10.47% (box acc@0.5 5.15%), Qwen3-VL-8B 9.30% (5.74%), best fine-tuned model 25.30% (21.00%) [S61].
- Hybrids: AnomalyGPT (decoder makes the pixel map, map is turned into LLM prompt embeddings) [S48]; Myriad (expert map guides the LLM; detection equals the expert's) [S49]; VMAD [S50]; VELM (detector first, LLM names defect type only when detected; 80.4% defect-class accuracy on MVTec AD) [S62]; AnomalyClaw (2026, tool loop that refutes against normal references, +3.5 to +7.9 macro-AUROC over single-shot answers) [S63].
- MMAD's expert-agent test: feeding PatchCore/AnomalyCLIP maps improved only localisation and hurt discrimination and description; ground-truth location as text raised discrimination by 24.89 points; a plain bounding-box overlay was the best visual format [S42]. Implication: give the VLM a box/crop derived from the anomaly map, not the raw colormap.
- Failure modes: weak fine-grained perception, inability to exploit normal reference images, gains saturate after a few shots, chain-of-thought did not help discrimination, prompt brittleness, answering from priors rather than the image, forced downscaling of high-resolution inputs [S42][S54][S63][S45].

### 3.5 (d) Recommendation

Not as primary scorer; not in the MVP; possibly as an explainer in a later phase.

1. Primary scorer: no. Zero-shot small VLMs score 50-67% on a balanced yes/no anomaly question; the best fine-tuned 3-8B models reach 72-88%. VLM-AD in anomalib returns only a parsed yes/no with no continuous score [S34], so no operating point can be set at near-zero escape. Our metric needs a monotone score with a well-behaved defect-score tail; generated tokens do not provide that (token log-probabilities can be used as a score, but no published calibration for industrial defects was found).
2. Explainer on top of an anomaly model: viable after MVP. Anomaly model gives score, map and the decision; the VLM receives the crop or a box overlay from the thresholded map plus a normal reference and returns a defect type and a one-line reason for the technician. It must never flip a reject to an accept. Budget ~0.1-0.45 s per rejected item on Orin NX/AGX or Thor-class hardware, which fits a reject lane.
3. MVP: ship a calibrated specialist model (section 4) and the heatmap. Revisit the VLM once there is a labelled set of real line rejects to test whether technicians actually use text explanations.

---

## 4. Metric and thresholding

### 4.1 Recovery at fixed escape = TNR at fixed FNR

Positive class = defective. For score s and threshold t (accept if s < t):
- Escape rate = FNR(t) = share of true defects with s < t (released by us).
- Recovery rate = TNR(t) = share of good items in the reject lane with s < t (returned to the line).
- Choose t as the alpha-quantile of defect scores on a calibration set; recovery is then the share of normal scores below t. With alpha = 0, t = minimum defect score: recovery is the share of good items scoring below the least anomalous defect.

This is already implemented: anomalib `PGn(fnr=...)` ("presorted good", TNR at n% FNR) and `PBn(fpr=...)` (TPR at n% FPR), from [S21], code computes `thr = torch.quantile(pos_scores, fnr)` and `pg = mean(neg_scores < thr)` [S16][S35]. Default fnr is 0.05; we need `PGn(fnr=0.0)` and `PGn(fnr=0.01)`. Note `torch.quantile` interpolates linearly between order statistics, so for small defect sets PG at 1% and PG at 0% can be nearly identical; report the underlying threshold too.

### 4.2 What "near-zero" can mean statistically

| Calibration defects n | 95% upper bound on escape if 0 escapes observed (rule of three, ~3/n) | Smallest alpha a split-conformal threshold can guarantee (1/(n+1)) |
|---|---|---|
| 60 (typical MVTec AD category) | ~5% | ~1.6% |
| 100 (VisA category) | ~3% | ~1% |
| 300 | ~1% | ~0.33% |
| 1,000 | ~0.3% | ~0.1% |

(Arithmetic, not sourced.) Consequences: per-category MVTec AD / VisA results can only certify escape at the few-percent level; pool categories or use repeated splits; report the escape rate on a held-out set (not the set used to pick t), with a Clopper-Pearson interval. Choosing t on the test set (anomalib default `val_split_mode=SAME_AS_TEST` [S19]) makes "0 escapes" true by construction and is not a result.

### 4.3 How anomalib exposes scores and thresholds

- Model forward returns `InferenceBatch` with `pred_score` (image), `anomaly_map` (pixel), and after the PostProcessor `pred_label`, `pred_mask` [S17].
- PostProcessor fits `F1AdaptiveThreshold` (threshold maximising F1 on validation) for image and pixel levels, then min-max normalises so the threshold maps to 0.5, clamped to [0,1]; `image_sensitivity` shifts the cut [S17]. F1-optimal is the wrong operating point for asymmetric costs.
- For our metric: set `PostProcessor(enable_normalization=False, enable_thresholding=False)` or read raw scores, set our own threshold from a defect calibration split, and evaluate with `PGn` via `Evaluator(test_metrics=[PGn(fnr=0.0, fields=["pred_score","gt_label"]), ...])` [S18]. `ManualThreshold` exists for deploying a fixed cut [S16].

### 4.4 What else to report and why AUROC alone misleads

| Metric | Report? | Why |
|---|---|---|
| PG0 / PG1 / PG2 (recovery at 0/1/2% escape) on the reject lane | Primary | Matches the cost asymmetry |
| Escape rate at the deployed threshold, held-out, with CI | Primary | The constraint itself |
| Image AUROC | Secondary | Averages over all thresholds; dominated by easy normals. Near-zero escape lives in the extreme tail of the defect-score distribution, which AUROC barely weights. On VAD, PatchCore had the best AUROC (88.0) but PG2 of 16.5 vs 27.6 for MMR [S21] |
| F1-max | Secondary | Single threshold that balances precision and recall; not cost-aligned |
| AU-PRO at FPR 0.05 (as in MVTec AD 2) | For heatmap quality | Region-equal weighting; the 0.30 limit includes maps too noisy to show an operator [S22] |
| AUPIMO | For heatmap quality on normals | Per-image overlap with FPR measured only on normal images, integrated at very low FPR; in anomalib [S36] |
| Human handoff rate | Operational | Share of reject lane sent to the technician |

### 4.5 Calibration and abstention for the human handoff

Three-way decision on one score s:
- s < t_low: recover (return to line). t_low is set by the escape constraint on defect calibration scores (class-conditional split-conformal quantile: the k-th smallest defect score, k = floor(alpha (n+1))).
- s > t_high: confirm reject. t_high from normal calibration scores, e.g. above the 99th percentile of good items in the reject lane.
- otherwise: send to human. Handoff volume is the share of the lane between the two cuts; trade it against recovery.

Literature: selective classification (reject option with a target risk, risk-coverage curves) [S37]; conformal p-values for outlier detection with FDR/FPR control [S38]; conformal risk control, which bounds expected FNR with a monotone loss, tight to O(1/n) [S39]; conformal pixel-level defect segmentation with risk guarantees (steel surfaces) [S64]. All require exchangeability between calibration and deployment data: a lighting change or new product lot breaks the guarantee, so recalibrate per SKU/lot and monitor the normal-score distribution.

Uncertainty from ensembles: disagreement between two different families (e.g. PatchCore and EfficientAD or RD) is a cheap extra handoff trigger: recover only if both are below their own t_low. This also guards against a single model's blind spot (section 5.5).

Simplest to implement: one model, raw scores, two thresholds fitted on a held-out calibration split of the reject lane, plus a "both models agree" rule if a second model is affordable. Conformal wording is then just a correct description of how t_low was chosen.

---

## 5. Reject-lane benchmark construction

### 5.1 The problem

The reject lane = (defects the primary caught) + (normals the primary falsely rejected). Those normals were selected because they look anomalous to some detector (glare, shift, borderline features). Measuring recovery on all test normals of MVTec AD/VisA measures recovery on mostly easy normals and overstates performance. The gap is largest when the primary and our model react to the same nuisance cues.

### 5.2 What the literature already has

No public image benchmark measures hard-normal recovery at fixed escape among items another system already rejected. Closest work:

| Work | Relevance | Key numbers |
|---|---|---|
| AOI false-call reduction, Pfab & Rothering (COMPSAC 2025) [S66] | Same setup: ML gate between PCB AOI and manual re-inspection. Metrics: volume reduction v = TN/(TN+FP) (= our recovery), slip rate s = FN/(TP+FN) (= our escape), V@S = best v subject to s at or below target. Targets v at least 40%, s at most 1% | Random split: RFC v 0.897 / s 0.010, BRFC v 0.797 / s 0.008. Chronological slices: slip 5-15% (5-15x target). Thresholds tuned on random splits do not survive drift |
| Public AOI SMT false-call dataset [S67] | Tabular, pin-level, not images | 440,274 rows, 98.9% false calls |
| AOI tolerance optimisation [S68] | | 18% fewer false calls at 100% recall |
| Semiconductor nuisance filtering / overkill [S69][S70] | Overkill (good die rejected) vs underkill (escape) is the standard pair in wafer/die inspection | Up to 95% nuisance reduction (vendor, SPIE); AOI overkill above 20% in one chip-AOI case. No public image benchmark reports overkill at fixed underkill |
| HETMM (IJCV 2024) [S71] | Names "hard normal examples" as the source of false positives; template matching that keeps hard normals as templates | |
| SegAD (CVPR 2024) [S73] | Second-stage classifier on anomaly-map statistics plus optional supervised score; directly reusable as a re-check stage | +2.1 AUROC on VAD, +0.4 on VisA |
| Kaputt (ICCV 2025) [S74] | Retail packaging; no/minor/major defect labels; reports recall at 1% FPR | Best supervised R@1%FPR 59.36; zero/few-shot AUROC at most 56.96 |
| Specification ambiguity on MVTec AD [S72] | Anomaly-to-normal and normal-to-anomaly relabelling (borderline items) | S-AUROC metric |
| MVTec AD 2 `test_private_mixed` [S22] | Real lighting shift on normals and defects | AU-PRO0.05 EfficientAD 30.8 to 19.2, MSFlow 24.3 to 11.9, RD 26.4 to 25.0; VAND 3.0 winner SegF1 53.81 to 51.43 [S76] |
| Robustness sets: RAD (viewpoint, illumination, blur) [S78], RobustAD (CC BY 4.0) [S80], M2AD (12 views x 10 illuminations) [S81], MAD-Bench [S79] | Nuisance-shifted normals | Memory-bank and synthetic-anomaly methods most robust on RAD [S78]; noise read as defect on MAD-Bench [S79] |

The AOI work supplies the metric template (V@S) and the main warning: thresholds fitted on a random split leak; use time- or lot-ordered splits as soon as line data exists.

### 5.3 Proposed protocol (MVTec AD, VisA; extendable to MVTec AD 2, GoodsAD)

Step 0, pools. Per category: normals N = train good + test good; defects D = test defective. Test-good counts are too small (MVTec AD bottle 20, hazelnut 40; VisA cashew 50 [S104]), so use 5-fold cross-validation over N: each fold trains both the stand-in and our model on 80% of N and evaluates on the held-out 20% plus D. Split D into calibration and evaluation halves, stratified by defect type, over 10 random seeds.

Step 1, simulate the primary camera with deliberately different, shallow methods. Run at least two; each threshold is set to the minimum stand-in score over D (catches 100% of defects; with a 99.5% variant reported separately):

| Stand-in | Implementation | Expected false-reject cause | Correlation with deep-feature models |
|---|---|---|---|
| P1 Variation model (golden template) | Per-pixel mean and std image from aligned training normals; flag pixel if abs(I - mean) > max(a, b*std); reject if flagged area > k px. Standard rule-based method (HALCON `create_variation_model` [S75]) | Misalignment, rotation, lighting, specular highlights | Low |
| P2 Registered template difference | Register to the nearest of M normal templates (ECC or feature homography), absolute difference, blur, threshold, blob area | Residual misregistration, natural shape variation | Low |
| P3 Classical feature thresholds | Colour/intensity histogram distance, blob count/area, edge density vs normal ranges | Colour drift, borderline appearance | Low-medium |
| P4 Tiny PaDiM | ResNet-18, 128 px, few channels | Feature-space outliers | High if our model uses CNN ImageNet features |

The reject lane for a fold = everything the stand-in flags. Record the stand-in false-reject rate and the lane normal count; a category with fewer than ~30 lane normals per fold is pooled or dropped (rule of thumb, not sourced).

Step 2, score the lane with our model only. Fit t_low on the defect calibration half (section 4.5) using lane defects; report on the evaluation half: PG0, PG1, PG2 on lane normals; escape with Clopper-Pearson CI; handoff rate for the three-way rule. Also report the same metrics on all held-out normals; the difference "PG(all) - PG(lane)" is the overstatement and should appear in every results table.

Step 3, diagnose shared cues. Spearman correlation between stand-in score and our score on normals; recovery by stand-in score decile; heatmap overlap (IoU of top-k% regions) between stand-in difference map and our anomaly map on lane normals. High correlation means the stand-in selected normals our model also finds hard: recovery is pessimistic for that stand-in. Low correlation means the lane is close to a random sample for our model: recovery is optimistic if the real primary shares cues with us. Report the result range across P1-P4 as the uncertainty bracket, not a single number.

Step 4, optional physical-cause augmentations on held-out normals only, before the stand-in: specular glare (saturated elliptical highlights), translation 2-10 px and rotation 1-5 degrees, defocus and motion blur, gamma/brightness/white-balance shift, vignetting, partial occlusion at the border. Augmented normals that the stand-in flags join the lane, tagged by cause; report recovery per cause. Rules: (i) valid only if our system re-scores the same frame the primary captured; if the reject lane is re-imaged at a second station, glare and pose change, and the benchmark must instead model the second camera; (ii) do not use the same augmentation family in our model's training when evaluating on it, or report both; (iii) keep augmentations sub-defect (a glare spot can resemble contamination), have a human check a sample, and report how many augmented normals a technician would still reject. MVTec AD 2 `test_public` contains normals under several lighting conditions with ground truth [S22], which gives real (non-synthetic) lighting nuisances for this step.

### 5.4 Categories likely to give meaningful hard-normal counts

Judgement based on the variation each category contains (not measured):

| Category | Why it should generate false rejects from shallow stand-ins | Note |
|---|---|---|
| VisA macaroni1, macaroni2, candle, capsules | Multiple instances with position variation; template/variation models false-reject heavily | 900-1,000 normals (capsules 602) [S83]; food-like |
| VisA cashew, fryum, pipe_fryum, chewinggum | Natural shape/colour variation of single food items | 500 normals each (chewinggum 503) [S83]; most food-like in VisA |
| MVTec AD hazelnut | Rotation and shape variation; 391 train normals | Food |
| MVTec AD pill, capsule | Printed text, colour speckle, slight pose | Pharma, small defects |
| MVTec AD bottle | Aligned top view, low variation | Likely too few lane normals; use as a control |
| MVTec AD 2 walnuts, rice, fruit_jelly, vial, can | Real lighting variation, natural products | Best realism; NC license; private test only via server |
| GoodsAD drink_bottle, drink_can, food_bottle, food_box, food_package | Pose and label variation on real packaged goods | No anomalib datamodule (use Folder) |

### 5.5 Failure mode to flag in every report

If the stand-in and our model fail on the same cues, the benchmark measures the wrong thing. Example: P4 (tiny PaDiM) and our PatchCore both rely on ImageNet CNN features; normals with glare score high in both, so these normals enter the lane and our model cannot recover them (pessimistic). The dangerous direction is the opposite: a pixel-space stand-in flags normals for reasons our model ignores (pose), which our model then easily recovers, while the real primary false-rejects for reasons our model shares (glare), so the reported recovery is too high. Mitigations: use at least one low- and one high-correlation stand-in; report per-cause recovery from step 4; once line data exists, replace all stand-ins with real primary reject logs and lot-ordered splits [S66].

---

## 6. Datasets

NC = non-commercial license (blocks use in a commercial product without a separate agreement). Counts are train normal / test normal / test anomalous unless stated.

| Dataset | Size | Categories | License | anomalib datamodule | Download | Closest to food/beverage |
|---|---|---|---|---|---|---|
| MVTec AD | 5,354 images: 3,629 train, 467 test good, 1,258 test anomalous [S104] | 15 (10 objects, 5 textures) | CC BY-NC-SA 4.0 (NC) [S19][S82] | `MVTecAD` (auto-download) | mvtec.com | hazelnut, bottle, pill, capsule |
| VisA | 10,821 (9,621 normal / 1,200 anomalous); 1cls split ~8,659 / 962 / 1,200 [S83][S104] | 12 | CC BY 4.0 per official README [S83]; anomalib docstring says CC BY-NC-SA 4.0 [S84] (conflict; treat the official README as authoritative, confirm before commercial use) | `Visa` (auto-download) | S3 tarball listed in [S83] | cashew, chewinggum, fryum, pipe_fryum, macaroni1/2, candle, capsules |
| MVTec AD 2 | 8,004 images; per category ~140-430 train, test_public / test_private / test_private_mixed [S22] | 8: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts | CC BY-NC-SA 4.0 (NC) [S22] | `MVTecAD2` | mvtec.com; private-test scoring via benchmark.mvtec.com | fruit_jelly, rice, walnuts, can, vial |
| MVTec LOCO AD | 3,644: 1,772 train + 304 val / 575 good / 561 logical + 432 structural [S87] | 5 | CC BY-NC-SA 4.0 (NC) [S86] | `MVTecLOCO` | mvtec.com | juice_bottle (fill level, label), breakfast_box |
| Real-IAD | 151,050 (99,721 normal / 51,329 anomalous), 5 views, 30 objects [S88] | 30 | CC BY-NC-SA 4.0, research only (NC) [S89] | `RealIAD` | Hugging Face, gated [S89] | bottle_cap, mint, vcpill |
| Real-IAD D3 / Variety | D3: 8,450 samples, 20 categories, RGB + pseudo-3D [S90]; Variety: 198,950 images, 160 categories [S91] | | Research use; no named license | No | project pages | none food-specific |
| MPDD | >1,000 images (exact split UNVERIFIED) | 6 metal parts | CC BY-NC-SA 4.0 (NC) [S92] | `MPDD` | GitHub link | none |
| BTAD | 2,830 images | 3 industrial products | Conflicting: anomalib says CC BY-NC-SA 4.0 [S93], a mirror says CC BY-SA (UNVERIFIED) | `BTech` | | none |
| GoodsAD (PKU) | 6,124 images, 3000x3000: drink_bottle 733/356/425, drink_can 234/147/147, food_bottle 1014/243/361, food_box 432/146/251, food_package 540/253/230, cigarette_box 183/183/246 [S94] | 6 packaged goods | Repo GPL-3.0; no dataset license stated (verify) [S94] | No (use `Folder`) | OneDrive / Baidu [S94] | All except cigarette_box; PatchCore I-AUROC 74-94 per category [S95] |
| VAD (Valeo) | train 2,000 good + 1,000 bad / test 1,000 good + 1,000 bad (165 unseen defect types) [S96] | 1 automotive part | CC BY-NC-SA 4.0 (NC) [S96] | `VAD` | GitHub / Google Drive | none; real production data |
| KolektorSDD2 | train 2,085 neg + 246 pos / test 894 neg + 110 pos [S97] | 1 surface | CC BY-NC-SA 4.0 (NC) [S97] | No (`Kolektor` is KSDD v1, 399 images [S98]) | vicos.si | none |
| Kaputt (ICCV 2025) | 238,421 images; test 10,067 [S74] | Retail items / packaging | CC BY-NC-ND 4.0 + research terms (NC) [S74] | `Kaputt` | Hugging Face | packaging |
| MVTec 3D-AD | 2,656 train + 294 val / 249 good / 948 anomalous [S100] | 10 | CC BY-NC-SA 4.0 (NC) | `MVTec3D` (depth) | mvtec.com | bagel, carrot, cookie, peach, potato |
| Eyecandies (synthetic) | per category 1,000 train / 100 val / 400 test (half defective) [S101] | 10 candies, 6 lighting renders | Dataset terms UNVERIFIED (code GPL-3.0) | No | `eyec` tool [S101] | all (confectionery) |
| MANTA (CVPR 2025) | 137,338 multi-view (128,721 normal / 8,617 anomalous) [S99] | 38, incl. groceries and grains | None stated | No | project page | groceries, wheat, maize, soybean |
| RobustAD (CVPRW 2025) | 741 train / ~1.4k test over shifted domains [S80] | PCB, metal, pharma packets | CC BY 4.0 | No | Hugging Face | pharma packets; useful for shift |
| FDD-48 | 13 food types, 48 defect classes, box labels [S103] | | UNVERIFIED | No | UNVERIFIED | food, but detection not AD |

Food-specific: no public pixel-level anomaly benchmark for food or beverage lines was found beyond GoodsAD, the MVTec/VisA food categories and the small image-label sets above. No public bottle-cap or label-inspection anomaly dataset was found.

Food-like test counts: MVTec AD bottle 209/20/63, hazelnut 391/40/70, pill 267/26/141, capsule 219/23/109 [S104]. VisA (1cls) candle 900/100/100, capsules 542/60/100, cashew 450/50/100, chewinggum 453/50/100, fryum 450/50/100, macaroni1 900/100/100, macaroni2 900/100/100, pipe_fryum 450/50/100 [S104]. MVTec AD 2 test_public good/bad: fruit_jelly 20/60, rice 42/90, vial 35/105, walnuts 60/90, can 72/90 [S22].

License consequence: for an internal prototype, NC data is fine; any model shipped commercially should be retrained on line data (or on VisA / RobustAD after license confirmation).

---

## 7. Edge deployment

### 7.1 Published numbers

| Model | Hardware / runtime | Latency / throughput | Memory / size | Source |
|---|---|---|---|---|
| EfficientAD-S / -M | RTX A6000, PyTorch, bs1 | 2.2 / 4.5 ms; 614 / 269 img/s bs16 | 100 / 161 MB GPU | [S8] |
| EfficientAD-S | A5000 / V100 / RTX 3080 / RTX 2080 Ti | 2.5 / 3.9 / 3.8 / 4.5 ms | | [S8] |
| PatchCore | A6000 / A5000 / V100 / 3080 / 2080 Ti | 32.0 / 31.5 / 47.1 / 41.1 / 53.2 ms | 637 MB plus kNN (not included) | [S8] |
| FastFlow, DSR, SimpleNet | RTX A6000 | 17 / 17 / 12 ms | 404 / 314 / 508 MB | [S8] |
| PatchCore WRN50, coreset 100 / 10 / 1% | datacentre GPU | 0.6 / 0.22 / 0.17 s; IVFPQ approx. search 0.2 s at 100% | | [S7] |
| anomalib PatchCore, 320 px, WRN50, 8000x1536 bank | RTX 1000 Ada laptop | 61 ms forward: kNN 35.9 ms, map step 25.4 ms (later found to be a cuDNN issue), backbone 8.3 ms | | [S106] |
| Dinomaly ViT-S / B / L | RTX 3090, bs16 | 153.6 / 58.1 / 24.2 img/s | 37 / 148 / 275 M params | [S9] |
| STFPM | Jetson Xavier NX: PyTorch FP32 / TensorRT FP16 / INT8 | 52.1 / 25.0 / 24.9 ms; AUROC 0.94 / 0.94 / 0.92 | 88 / 52.7 / 48.5 MB | [S108] |
| Reverse Distillation | Xavier NX: PyTorch FP32 / TRT FP16 / INT8 | 166.6 / 19.7 / 18.0 ms; AUROC 0.96 / 0.82 / 0.82 | 645 / 527 / 314 MB | [S108] |
| STFPM / RD | Xeon W-2265 CPU, PyTorch FP32 | 275 / 1,365 ms | | [S108] |
| PatchCore memory bank | WRN50 vs MobileNetV2 | | ~300 MB vs ~31 MB; PaDiM WRN50 3.72 GB; STFPM 5.3 MB | [S109] |
| Streaming/covariance PatchCore | training | | peak 5.41 to 2.78 GB | [S110] |
| TAO Visual ChangeNet-Segmentation (golden image + test, 256 px, FP16, bs16) | Orin Nano 8GB / Orin NX 16GB / AGX Orin 64GB | 15.2 / 21.9 / 55.1 FPS | | [S112] |
| ChangeNet-Classification (512x128) | same three | 31.0 / 44.7 / 113.2 FPS | | [S113] |
| Qwen3-VL-2B / Gemma 4 E4B (VLMs) | see section 3.3 | 0.1-0.45 s per verdict | | [S59] |

Gaps (UNVERIFIED / not found): any primary-source latency for PatchCore, EfficientAD, PaDiM, FastFlow or Dinomaly on Orin Nano/NX/AGX, and any anomalib numbers on Intel iGPU/NPU. One open project lists PatchCore TensorRT on Orin Nano as "TBD" with a [21401, 1536] FP32 bank (~131 MB) [S121]. Plan to measure ourselves with `trtexec` and OpenVINO `benchmark_app`.

### 7.2 Practical findings

- Quantisation can silently break a model: RD lost 14 AUROC points under TensorRT FP16 on Xavier NX while STFPM held [S108]. Re-run the full metric suite (PG0, escape) after every export and precision change, not just AUROC.
- PatchCore cost on edge is the memory bank and kNN, not the backbone [S106]. anomalib computes a full distance matrix (out of memory on small GPUs); FAISS would help but blocks ONNX/OpenVINO export [S107]. Options: 1% coreset (about 0.4 PRO points cost [S7]), smaller backbone, FP16 bank with matmul + top-k inside the engine, or EfficientAD/STFPM-style models whose cost is fixed.
- Runtime choice: an independent 2026 benchmark found OpenVINO fastest on CPUs and TensorRT fastest on GPUs; TensorRT did not beat PyTorch on the transformer it tested [S111], which matters for ViT-based Dinomaly/AnomalyDINO.
- The reject lane relaxes the budget: if the primary rejects a few percent of items, even 100-300 ms per item on an Orin NX or a CPU is usually enough, which brings PatchCore and Dinomaly-S within reach. Exact budget depends on line rate (to be measured).

### 7.3 Vendor reference pipelines

- Intel: Open Edge Platform "Industrial Edge Insights - Vision" on DL Streamer Pipeline Server, with a PCB anomaly-detection sample; validated on Core Ultra CPU/iGPU/NPU and Arc GPUs; minimum 12th-gen Core, 16 GB RAM; no performance numbers published [S116][S117]. Models come from OpenVINO or Intel Geti (OpenVINO IR FP32/FP16/INT8) [S119]. OEP 2026.1 lists Anomalib Studio and anomalib 2.5.0 [S118]. "Geti Inspect" appears as an anomalib epic; relation to Anomalib Studio UNVERIFIED [S120].
- NVIDIA: no Metropolis, DeepStream or Jetson Platform Services reference workflow using one-class anomaly models was found; JPS reference workflows are video recording and generative-AI apps [S115]. The inspection-relevant NVIDIA model is TAO Visual ChangeNet, a supervised golden-image comparison model (MVTec bottle: 99.67% accuracy, 92.3% mIoU) [S114]. It needs labelled defects and an aligned golden image, so it is a reference design for comparison-based inspection, not a drop-in for good-only training.

---

## 8. Explainability

### 8.1 Heatmaps by model family

| Family | What the map is | Native resolution / sharpness | Faithfulness |
|---|---|---|---|
| Memory bank (PatchCore) | Distance of each patch feature to its nearest normal patch | Stride-8 feature grid, bilinear upsample, Gaussian blur (sigma 4 in anomalib): blobby, misses edges of small defects | Map is the decision (image score = max patch distance). Nearest normal patch is retrievable, so an example-based explanation is free [S7][S106] |
| Per-position Gaussian (PaDiM) | Mahalanobis distance per patch position | Coarse | Faithful to score; false hot spots on misregistered parts |
| Student-teacher / distillation (EfficientAD, RD, STFPM, Dinomaly) | Feature mismatch between teacher and student/decoder | EfficientAD averages quantile-normalised local and global maps [S8]; Dinomaly 28x28 cosine map, score = mean of top 1% [S9] | Faithful to "student fails here", which also fires on unusual but acceptable texture |
| Normalizing flow (FastFlow, CFLOW) | Negative log-likelihood per feature position | Feature-map resolution, smooth | Score-faithful; likelihood not comparable across positions |
| Reconstruction (DRAEM, DSR) | Pixel residual or learned segmentation head | Sharpest, noisy on edges and high-frequency texture | DRAEM's head is learned, so no longer a pure residual |
| CLIP-based (WinCLIP, AnomalyCLIP) | Patch/window similarity to "damaged" vs "normal" text | 14-16 px patches, coarse | Least precise; WinCLIP zero-shot P-AUROC 85.1 on MVTec [S29] |

These maps are the model's actual score field (unlike Grad-CAM), so they pass model-dependence sanity checks by construction [S123]; they are faithful to feature distance, not to "defect". Upsampling and blur suggest more precision than exists, and natural variation in food lights up. Measure map quality with AU-PRO at FPR 0.05 and AUPIMO (FPR on normal images only) rather than pixel AUROC, which large defects dominate and which is saturated on MVTec AD and VisA [S36][S22]. sPRO (MVTec LOCO) covers logical anomalies where boundaries are undefined [S86].

### 8.2 Human-factors evidence

| Study | Setting | Finding |
|---|---|---|
| Senoner et al. 2024 [S133] | 48 factory workers, 9,600 assessments, electronics inspection with anomaly heatmaps vs black-box AI | Five-fold drop in median error; balanced accuracy 88.6% to 96.3%; overruling wrong AI 86.4% to 96.9%. In the radiology arm overruling wrong AI fell (57.7% to 50.8%) |
| See 2015, Sandia [S134] | 82 inspectors, 140 parts | 85% hits, 35% false alarms; high mental demand |
| Kim et al. 2022, HIVE [S125] | ~1,000 participants | Participants believed 67.2% of wrong Grad-CAM predictions; picking the right explanation for wrong predictions at chance |
| Nguyen et al. 2021 [S126] | Image classification | Attribution maps no better than nearest training examples; hurt on fine-grained classes |
| Bansal et al. 2021 [S127] | AI about as accurate as humans | Explanations raised acceptance regardless of correctness |
| Zhang et al. 2020 [S128] | Confidence vs explanation | Confidence scores calibrated trust; local explanations did not |
| Bucinca et al. 2021 [S129] | Cognitive forcing | Reduced over-reliance; least liked by users |
| Dratsch et al. 2023 [S132] | Radiologists given wrong AI suggestions | Accuracy fell from 82.3% to 45.5% (very experienced) and 79.7% to 19.8% (inexperienced) |
| Saporta et al. 2022 [S130] | 7 saliency methods, chest X-ray | All worse than human localisation, worst on small complex shapes |
| Borkin et al. 2011 [S135] | Rainbow vs perceptual colormaps | Diverging/perceptual maps found substantially more targets than rainbow |

### 8.3 Design rules for the technician view

1. Raw image first or side by side, overlay on one-key toggle; never overlay only.
2. Fixed score-to-colour scale calibrated on held-out normals, not per-image min-max (per-image scaling makes every good item show a "hottest" spot).
3. Default to a thresholded contour or box at t_low on the raw image; continuous heat optional, perceptual colormap, never jet; state true map resolution.
4. Show the nearest normal reference (PatchCore gives it for free; otherwise a golden image). Examples beat attribution maps [S126].
5. Show a calibrated score ("above 99.7% of good items"), and the band (recover / human / reject).
6. Guard against confirmation bias: the item was already rejected and our map will often agree. Have the technician judge some items before seeing the map, seed known-good and known-bad items, track override rates of wrong model outputs [S129][S132].
7. Say what the map cannot show: logical/global defects (fill level, missing cap, wrong label) need a global score, not a diffuse heatmap.
8. Log technician decisions as labelled data for recalibration; feeding confirmed false positives back into a PatchCore bank recovers a median 66% of the gap to a full bank in simulation [S137] (simulated reviewers only).
9. Keep review sessions short; human inspection itself runs at roughly 80-85% hit rate [S134].

---

## 9. Sources

All accessed 2026-10-17.

| ID | URL | Supports |
|---|---|---|
| S1 | https://api.github.com/repos/open-edge-platform/anomalib/releases | Release tags, dates, v2.5-v2.7 changes |
| S2 | https://github.com/open-edge-platform/anomalib | License, CLI, export, Anomalib Studio |
| S3 | https://github.com/open-edge-platform/anomalib/releases/tag/v2.0.0 | v2.0 changelog (task removal, renames, PostProcessor) |
| S4 | https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image | 33 image model directories |
| S5 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html | Model descriptions and papers |
| S6 | https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/data/datamodules | Image, depth, video datamodules |
| S7 | https://arxiv.org/html/2106.08265 | PatchCore results, timing, few-shot |
| S8 | https://arxiv.org/html/2303.14535 | EfficientAD results, latency per GPU, memory |
| S9 | https://arxiv.org/html/2405.14325 | Dinomaly results, params, throughput |
| S10 | https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/deploy/export.py | ExportType, CompressionType |
| S11 | https://arxiv.org/html/2107.12571 | CFLOW-AD results, size, fps; PaDiM numbers as cited |
| S12 | https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/deploy/inferencers | Torch/OpenVINO inferencers |
| S13 | https://github.com/open-edge-platform/anomalib/tree/main/tools/inference | Inference scripts |
| S14 | https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/pipelines/benchmark/index.html | Benchmark pipeline |
| S15 | https://arxiv.org/html/2601.20524 | AnomalyVFM results, 2025 zero-shot baselines, latency |
| S16 | https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/metrics | Metric list incl. PGn/PBn, thresholds |
| S17 | https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/post_processing/post_processor.py | Threshold, normalisation, output fields |
| S18 | https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/evaluation/evaluator.html | Evaluator usage |
| S19 | https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py | Split defaults, MVTec AD license |
| S20 | https://arxiv.org/html/2405.14529 | AnomalyDINO few-shot results |
| S21 | https://arxiv.org/html/2503.23451 | PG/PB metrics, realistic-protocol AUROC and PG2 |
| S22 | https://arxiv.org/html/2503.21622 | MVTec AD 2 data, splits, baselines, license |
| S23 | https://arxiv.org/html/2207.14315 | VisA paper baselines |
| S24 | https://arxiv.org/html/2201.10703 | Reverse Distillation results |
| S25 | https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/uninet/README.md | UniNet anomalib numbers |
| S26 | https://github.com/pangdatangtt/UniNet | UniNet venue (CVPR 2025); paper numbers not extractable |
| S27 | https://arxiv.org/html/2111.07677 | FastFlow results, params, fps |
| S28 | https://arxiv.org/html/2108.07610 | DRAEM results |
| S29 | https://arxiv.org/html/2310.18961 | AnomalyCLIP and zero-shot baselines |
| S30 | https://arxiv.org/html/2305.17382 | April-GAN zero/few-shot, WinCLIP few-shot |
| S31 | https://arxiv.org/html/2407.15795 | AdaCLIP results |
| S32 | https://github.com/xrli-U/MuSc | MuSc results, latency |
| S33 | https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/anomalyvfm/README.md | AnomalyVFM anomalib numbers, CVPR 2026 |
| S34 | https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/vlm_ad/lightning_model.py | VLM-AD outputs and backends |
| S35 | https://raw.githubusercontent.com/open-edge-platform/anomalib/main/src/anomalib/metrics/pg_pb.py | PGn/PBn code |
| S36 | https://arxiv.org/abs/2401.01984 | AUPIMO |
| S37 | https://arxiv.org/abs/1705.08500 | Selective classification |
| S38 | https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-1/Testing-for-outliers-with-conformal-p-values/10.1214/22-AOS2244.full | Conformal p-values for outliers |
| S39 | https://arxiv.org/abs/2208.02814 | Conformal risk control, FNR control |
| S40 | https://ai.google.dev/gemma/docs/core/model_card_4 | Gemma 4 sizes, encoder, token budgets |
| S41 | https://simonwillison.net/b/9404 | Gemma 4 launch date (secondary) |
| S42 | https://arxiv.org/html/2410.09453v3 | MMAD benchmark, failure modes, expert-agent test |
| S43 | https://arxiv.org/html/2608.09789 | ADOPD; Qwen VL 2B-8B zero-shot MMAD |
| S44 | https://arxiv.org/html/2507.21619 | EMIT; Gemma 3, Qwen2.5-VL on MMAD |
| S45 | https://arxiv.org/html/2504.11914 | AnomalyR1 |
| S46 | https://arxiv.org/html/2609.29457 | Anomaly-LR and 2025-26 comparison |
| S47 | https://arxiv.org/html/2508.09178v1 | IAD-R1 |
| S48 | https://arxiv.org/html/2308.15366 | AnomalyGPT |
| S49 | https://arxiv.org/html/2310.19070v3 | Myriad |
| S50 | https://arxiv.org/html/2409.20146v1 | VMAD |
| S51 | https://arxiv.org/html/2502.07601v2 | Anomaly-OV |
| S52 | https://arxiv.org/pdf/2501.01767 | LogicAD |
| S53 | https://arxiv.org/pdf/2311.02612 | GPT-4V zero-shot AD |
| S54 | https://arxiv.org/abs/2508.01921 | InspectVLM (Florence-2) failure modes |
| S55 | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | Qwen2.5-VL tokens, grounding |
| S56 | https://arxiv.org/pdf/2511.21631 | Qwen3-VL encoder (patch 16, merge 2) |
| S57 | https://arxiv.org/html/2412.03555 | PaliGemma 2 tokens, location/mask tokens |
| S58 | https://arxiv.org/html/2504.05299v1 | SmolVLM tokens |
| S59 | https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html | VLM ViT/prefill/decode on Orin, Thor, Spark |
| S60 | https://moondream.ai/blog/moondream-3-preview | Moondream 3 detect/point |
| S61 | https://arxiv.org/html/2603.13779 | AD-Copilot, MMAD-BBox localisation |
| S62 | https://arxiv.org/abs/2505.02626 | VELM detector-then-LLM |
| S63 | https://arxiv.org/abs/2605.10397 | AnomalyClaw |
| S64 | https://arxiv.org/abs/2504.17721 | Conformal defect segmentation |
| S65 | https://arxiv.org/abs/2303.14814 | WinCLIP |
| S66 | https://arxiv.org/html/2506.14521 | AOI false-call gate, V@S, drift |
| S67 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10847760/ | Public AOI false-call dataset |
| S68 | https://arxiv.org/abs/2505.03576 | AOI tolerance optimisation (18% at 100% recall) |
| S69 | https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14085/140851G/ | Semiconductor nuisance filtering (vendor) |
| S70 | https://mdpi-res.com/d_attachment/engproc/engproc-98-00026/article_deploy/engproc-98-00026.pdf | Chip AOI overkill case |
| S71 | https://arxiv.org/abs/2303.16191 | HETMM hard normals |
| S72 | https://arxiv.org/html/2604.07097v1 | Specification-ambiguity scenarios |
| S73 | https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html | SegAD, VAD |
| S74 | https://arxiv.org/html/2510.05903 ; https://huggingface.co/datasets/amazon/kaputt | Kaputt dataset, R@1%FPR, license |
| S75 | https://www.mvtec.com/doc/halcon/2405/en/create_variation_model.html | Variation model (golden template) |
| S76 | https://arxiv.org/html/2509.17615v1 | VAND 3.0 lighting-shift results |
| S77 | https://arxiv.org/abs/2605.14808 | SuperADD / VAND 4.0 |
| S78 | https://arxiv.org/abs/2406.07176 | RAD robustness benchmark |
| S79 | https://arxiv.org/abs/2411.14515 | MAD-Bench |
| S80 | https://huggingface.co/datasets/AmazonScience/RobustAD | RobustAD |
| S81 | https://arxiv.org/abs/2505.10996 | M2AD |
| S82 | https://www.mvtec.com/company/research/datasets/mvtec-ad | MVTec AD license/download |
| S83 | https://github.com/amazon-science/spot-diff | VisA counts, CC BY 4.0, download |
| S84 | https://raw.githubusercontent.com/open-edge-platform/anomalib/main/src/anomalib/data/datasets/image/visa.py | anomalib's conflicting VisA license note |
| S85 | https://www.mvtec.com/company/research/datasets/mvtec-ad-2 | MVTec AD 2 page |
| S86 | https://www.mvtec.com/company/research/datasets/mvtec-loco | MVTec LOCO license, sPRO |
| S87 | https://arxiv.org/html/2410.10234 | LOCO counts |
| S88 | https://arxiv.org/html/2403.12580 | Real-IAD |
| S89 | https://huggingface.co/datasets/Real-IAD/Real-IAD | Real-IAD license, access |
| S90 | https://realiad4ad.github.io/Real-IAD_D3/ | Real-IAD D3 |
| S91 | https://arxiv.org/html/2511.00540v2 | Real-IAD Variety |
| S92 | https://github.com/stepanje/MPDD | MPDD |
| S93 | https://raw.githubusercontent.com/open-edge-platform/anomalib/main/src/anomalib/data/datasets/image/btech.py | BTAD license per anomalib |
| S94 | https://github.com/jianzhang96/GoodsAD | GoodsAD counts, download, repo license |
| S95 | https://arxiv.org/abs/2307.04956 | GoodsAD paper, PatchCore baseline |
| S96 | https://github.com/abc-125/vad | VAD counts, license |
| S97 | https://www.vicos.si/resources/kolektorsdd2/ | KSDD2 |
| S98 | https://www.vicos.si/resources/kolektorsdd/ | KSDD v1 |
| S99 | https://arxiv.org/html/2412.04867v1 | MANTA |
| S100 | https://arxiv.org/html/2112.09045 | MVTec 3D-AD |
| S101 | https://github.com/eyecan-ai/eyecandies | Eyecandies |
| S103 | https://arxiv.org/abs/2605.24508 | FDD-48 food defects |
| S104 | https://arxiv.org/html/2511.06687 | Per-category MVTec AD / VisA counts |
| S106 | https://github.com/open-edge-platform/anomalib/issues/3780 | PatchCore stage profile |
| S107 | https://github.com/openvinotoolkit/anomalib/discussions/1935 | PatchCore memory, FAISS export blocker |
| S108 | https://arxiv.org/html/2407.02968 | STFPM/RD on Xavier NX with TensorRT |
| S109 | https://arxiv.org/html/2410.11591 | Memory-bank sizes for edge |
| S110 | https://arxiv.org/abs/2605.27748 | Streaming/covariance PatchCore memory |
| S111 | https://arxiv.org/abs/2607.11356 | OpenVINO vs TensorRT runtime benchmark |
| S112 | https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/visual_changenet_segmentation_mvtec | ChangeNet-Segmentation Orin FPS |
| S113 | https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/visual_changenet_classification | ChangeNet-Classification Orin FPS |
| S114 | https://developer.nvidia.com/blog/transforming-industrial-defect-detection-with-nvidia-tao-and-vision-ai-models | ChangeNet MVTec bottle result |
| S115 | https://docs.nvidia.com/jetson/jps/moj-Releasenotes.html | Jetson Platform Services workflows |
| S116 | https://docs.openedgeplatform.intel.com/dev/_sources/edge-ai-suites/ai-suite-manufacturing/industrial-edge-insights-vision/get-started/vision-system-requirements.md.txt | Intel validated hardware |
| S117 | https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/ai-suite-manufacturing/industrial-edge-insights-vision/pcb-anomaly-detection/index.html | Intel PCB anomaly sample |
| S118 | https://docs.openedgeplatform.intel.com/2026.1/OEP-articles/release-notes.html | OEP 2026.1, Anomalib Studio |
| S119 | https://docs.geti.intel.com/cloud/guide/release-notes/2.0/release-2.0.html | Geti export precisions |
| S120 | https://github.com/open-edge-platform/anomalib/issues/3065 | Geti Inspect epic |
| S121 | https://github.com/minwoo1119/edge-anomaly-detection | PatchCore Orin Nano plan (results TBD) |
| S123 | https://arxiv.org/abs/1810.03292 | Saliency sanity checks |
| S125 | https://arxiv.org/html/2112.03184 | HIVE user study |
| S126 | https://arxiv.org/abs/2105.14944 | Attribution maps vs nearest examples |
| S127 | https://arxiv.org/abs/2006.14779 | Explanations raise acceptance regardless of correctness |
| S128 | https://arxiv.org/abs/2001.02114 | Confidence vs explanation for trust |
| S129 | https://arxiv.org/abs/2102.09692 | Cognitive forcing |
| S130 | https://techxplore.com/news/2022-10-common-approach-demystify-black-ai.html | Saporta 2022 summary (secondary; DOI 10.1038/s42256-022-00536-x) |
| S132 | https://www.rsna.org/news/2023/may/ai-bias-may-impair-accuracy | Automation bias figures (secondary) |
| S133 | https://arxiv.org/html/2406.08271v1 | Heatmap field experiment in factory inspection |
| S134 | https://www.sandia.gov/research/publications/details/visual-inspection-reliability-for-precision-manufactured-parts-2015-12-01/ | Human inspection hit/false-alarm rates |
| S135 | https://iis.seas.harvard.edu/papers/borkin11-infoviz.pdf | Colormap study |
| S137 | https://arxiv.org/html/2608.17775v1 | Operator feedback into memory bank |

Gaps in numbering (S102, S105, S122, S124, S131, S136) were merged into neighbouring entries or dropped. Items marked UNVERIFIED inline: UniNet paper numbers, AD-DINOv3/AVA-DINO numbers, MPDD exact split, BTAD license, Eyecandies and FDD-48 terms, Gemma 4 launch date (secondary only), 4090 VLM prefill latency, Jetson Orin latencies for the main anomaly models, Intel iGPU/NPU anomalib latencies, Geti Inspect vs Anomalib Studio.

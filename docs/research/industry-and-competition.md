# Industry and competition: AI second opinion on rejected items in food and beverage inspection

Research snapshot as of 2026-10-17. Scope: existing inline inspection hardware, products that already re-check or "second-guess" inspection results, the semiconductor ADC analog, presentation numbers, the legal and practical question of returning rejected items to the good stream, current reject handling, and human-review UX.

**Confidence labels used below.** HIGH = regulation, official statistics or peer-reviewed. MEDIUM = vendor documentation or datasheet, trade press reporting a named deployment, or reputable secondary source. LOW = vendor marketing claim, undisclosed customer, forum or blog. UNVERIFIED = could not be confirmed against a primary source during this research.

---

## 0. Assumption check (summary)

| # | Assumption | Verdict | Key evidence |
|---|---|---|---|
| 1 | A primary inspection system exists and over-rejects | **Holds.** Over-rejection is a known, accepted design trade-off. | Heuft: good containers "are constantly being rejected" and a full-container false rejection rate of ~0.3% is described as the current average for a fully equipped inspector [S1, S2]. Mettler-Toledo: operators reduce metal detector sensitivity to cope with product effect, which raises miss risk [S30]. Mariner reports existing machine vision false reject rates of 25% and 34% at two (non-food) plants [S16]. |
| 2 | Rejects are binned today and nobody re-checks them | **Partly wrong.** It depends on the reject type. Foreign-body rejects go into locked bins for QA investigation. Some plants re-inspect correctable rejects by hand. Bulk sorting lines routinely re-sort the reject stream. | BRCGS requires rejects to go to a secure unit that only authorised staff can open [S40]. A brewery case had one operator manually re-inspecting rejected empty bottles [S53]. Key Technology describes dual-pass sorting and dedicated rework lines for nut rejects [S55]. No industry-wide survey of re-check frequency was found (UNVERIFIED). |
| 3 | A recovered good item can go back into the good flow | **Conditional.** Plausible for sealed rigid containers rejected for quality or cosmetic reasons, and for empty containers and bulk re-sort. Not acceptable as a plain "return to line" for exposed food or for rejects from food-safety controls (CCP/OPRP). See section 5. | 21 CFR 117.80(c)(9), 117.150; ISO 22000 8.9.4; BRCGS 3.8.1, 3.9.4, 4.10.3 [S34-S44]. |
| 4 | Sending a defect back costs far more than leaving a good item in the bin | **Holds, with a large margin.** A false reject costs cents to a few euros. A released defect can trigger a recall. | GMA survey: 77% of recalling firms put recall impact at up to USD 30M, and 23% higher [S47]. Undeclared allergens caused 27.4% of US recall events in 2004-2013 [S48]. |

---

## 1. What inspection stations run today

### 1.1 Vendor landscape

| Vendor / line | Form factor | Deep learning on board | How results leave the device | Cost class |
|---|---|---|---|---|
| **Cognex In-Sight 2800/3800/8900, D900** | Smart camera (IP67). The 3800 is rated 125 fps [S3]. | Yes. Edge learning tools are combined with rule-based tools on the In-Sight 3800 [S3]. The D900 runs ViDi deep learning inside the camera with no PC [S4]. | Discrete I/O plus Cognex Connect (EtherNet/IP, PROFINET, others). An OPC UA server is available on the 2000/7000 Gen2/8000/9000 series from firmware 5.8.0 [S5]. Pass and fail images can be recorded to a PC filmstrip of up to 10,000 images [S6]. | No official list price was found. One used D900 sold for USD 3,900-6,000 (LOW) [S7]. |
| **Cognex VisionPro / VisionPro Deep Learning (ViDi)** | PC-based (Windows plus GPU) | Yes. Red Analyze (anomaly or supervised segmentation with a heat map), Green Classify, and other tools [S8, S9]. | SDK and PLC integration through the host application | Software licence plus PC (no figure found; UNVERIFIED) |
| **Cognex OneVision** (June 2025) | Cloud training platform, initially for the In-Sight 3800/8900. Broader rollout was announced for early 2026 [S10]. | Trains in the cloud and runs on the camera | Same as In-Sight | Subscription (UNVERIFIED) |
| **Keyence IV4** (2025) | Vision sensor with an AI accelerator, no PC [S11] | Yes: AI Identify, OCR, Count, Trigger | Sensor I/O and fieldbus | Sensor class (no public price) |
| **Keyence VS series / VS-G controller** | Smart camera (up to 25 MP) or controller with up to 32 cores, separate DSP/CPU/AI cores and up to 64 GB RAM [S12] | Yes | Controller I/O and fieldbus | Mid to high (UNVERIFIED) |
| **Omron FH series** | Box vision controller tied to Omron PLCs (EtherCAT, EtherNet/IP, PROFINET) [S13] | AI Fine Matching (learns good-product variation to reduce overdetection) and AI Scratch Detect. Only the i7 models (FH-5050/5550) include Scratch Detect [S13]. | Fieldbus, tightly coupled to Sysmac PLCs | Controller class (no public price) |
| **SICK Inspector83x (Nova)** | 2D smart camera, up to 5 MP, quad-core [S14] | Yes, on device. Training runs in the dStudio cloud and the model is exported to the camera [S14]. | Web UI and fieldbus. Licences are device-bound with no time limit [S14]. | Licence tiers: Quality Inspection and Intelligent Inspection |
| **Basler (pylon AI, embedded kits)** | Cameras plus PC or embedded (NXP i.MX 8M Plus, NVIDIA Jetson via Syslogic rugged PCs) [S15] | On the host or embedded processor | Host application | Component pricing |
| **Zebra Aurora (ex-Matrox) Design Assistant, NS42** | PC/IDE plus smart sensors | Anomaly detection, classification, detection, segmentation, DL-OCR [S17] | Fieldbus and IDE | Sensor to PC class |
| **MVTec HALCON / MERLIC** | Software library (PC or embedded) | Global Context Anomaly Detection since HALCON 22.05, trained on good images only [S18] | Integrator-built | Runtime licence per device (UNVERIFIED) |
| **Teledyne DALSA BOA3** (Sept 2024) | Smart camera, 1.2-12 MP, IP67 [S19] | Classification and detection models from Astrocyte run on the camera | Onboard I/O | Smart camera class |
| **Food-line OEM inspectors** (Heuft, Krones Linatronic AI, Mettler-Toledo, Eagle, Ishida) | Integrated machines (empty/full bottle, X-ray, metal detection, checkweigher) | Krones applies deep learning to tell water droplets and foam from defects on empty bottles [S20]. Mettler, Eagle and Ishida market lower false rejects on X-ray [S21]. | Machine HMI plus line data software (for example Mettler ProdX logs reject time, place and reason) [S22] | Capital equipment |
| **Overview.ai OV20i** | AI smart camera with on-device training [S23] | Yes | Browser UI and I/O | **USD 9,450 per unit, no licence fees** (MEDIUM, vendor shop) [S23] |

**Interoperability standard.** OPC UA for Machine Vision (VDMA 40100) standardises control, recipe and result management. Result contents remain vendor-specific "black boxes" [S24]. A second-opinion tool therefore cannot count on a uniform reject-image interface. In practice it needs image capture of its own or vendor-specific image export (FTP or a filmstrip).

### 1.2 What "adding a GPU box" means in practice

| Option | Example | Price (as found) | Confidence |
|---|---|---|---|
| Jetson dev-class | Jetson Orin Nano Super dev kit | USD 249 at launch (Dec 2024). **USD 399 since NVIDIA's unannounced July 2026 increase.** The AGX Orin dev kit went from USD 1,999 to 3,499 and the AGX Orin 32GB module from USD 899 to 1,799 [S25, S26]. | MEDIUM |
| Jetson industrial fanless | Seeed reComputer Industrial J4012 (Orin NX 16GB) | EUR 1,849-2,349 retail [S27] | MEDIUM |
| Jetson rugged | Syslogic computers with Basler GMSL cameras (Aug 2026) | Quote only [S15] | MEDIUM |
| x86 industrial PC with NPU (OpenVINO path) | OnLogic Karbon 520 series (Intel Core Ultra with NPU, fanless, MIL-STD-810H) [S28] | Karbon 524 from USD 2,997, or USD 4,843 with an RTX A4000 Ada [S29] | MEDIUM |

The practical cost of a cheap edge box is roughly USD 400-2,500 in hardware. Integration, enclosure (IP rating or washdown), a camera at the reject station and validation usually cost more than the box (Control Engineering notes that engineering time, not parts, dominates PC-based vision cost [S31]).

**Why factories resist cloud inference** (mixed evidence, mostly secondary):
- **OT segmentation.** A manufacturer survey found that only 23% had achieved more than basic IT/OT convergence, and about a quarter keep plant operations separate from IT [S32]. Even where OT is connected, it sits behind firewalls rather than true air gaps [S33].
- **Patching and attack surface.** Cloud connectors and edge devices often go unpatched [S34a].
- **Vendor behaviour.** Even cloud-first vendors ship offline paths. LandingEdge can load a downloaded model with no cloud connection [S35]. Neurala VIA was positioned as running on existing factory PCs without internet [S36]. Cognex OneVision trains in the cloud but runs on the camera [S10].
- **Latency** is not binding for our case, because only rejects are re-checked and they sit downstream of the reject gate. IP and IT-policy concerns remain.

---

## 2. Who already does a second opinion or re-inspection with AI

### 2.1 Comparison table

Columns: **Overlay** = sits on top of an existing inspection system. **Rejects only** = explicitly re-evaluates items the primary system rejected. **HITL** = human review queue or labelling loop. **Edge** = runs on premises. **FR numbers** = publishes false-reject reduction figures.

| Product | Domain | Overlay | Rejects only | HITL | Edge | FR numbers (as published) | Evidence class |
|---|---|---|---|---|---|---|---|
| **Mariner Spyglass Visual Inspection** | Auto glass, fabric | **Yes**, on the existing MV system | Effectively yes. It targets false calls from the existing system. | Yes: in-factory app with a "Tag" button to correct labels | NVIDIA GPU (Metropolis), with cloud management | Glass: 25% to <1% FRR. Fabric: 34% to <2%. ~USD 4M and ~2M ROI per year [S16]. NVIDIA blog: 30x fewer misclassifications [S37]. | LOW-MEDIUM (vendor case, named fabric customer) |
| **Acquire Automation CapQ (False Rejection Suppression Technology)** | Beverage caps, fill level | Within its own system | Users "train in examples of product previously false rejected" [S38] | Training loop | On machine | No number published | MEDIUM (trade press, 2018-19) |
| **Krones Linatronic AI** | Empty bottle inspection | Inside the OEM inspector | Re-classifies droplet/foam artefacts that classic tools reject | Not documented | On machine | >99.9% droplet discrimination. Minimum FRR 0.3% [S20]. | LOW-MEDIUM (vendor) |
| **Heuft reflexx / SPECTRUM II** | Full and empty containers | Inside the OEM inspector | No | Not documented | On machine | reflexx "halved" FRR. Further 0.2% reduction expected. Up to 50% FRR reduction "realistic estimation" [S1, S2]. | LOW (vendor) |
| **Pleora AI Gateway** | Generic | **Yes.** It sits between any GigE/USB3/CameraLink camera and the existing inspection app [S39]. | No | Training UI | Yes (gateway) | Claims fewer false positives and "secondary screenings", no number | LOW (2020 launch) |
| **Averroes.ai** | Semiconductor, electronics AOI | **Yes**, "integrates with your existing proprietary inspection systems" | Positioned as an ADC behind AOI | Recommends routing low-confidence cases to an inspector dashboard | On-prem or cloud | "Up to 94%" FR reduction. 70-90% elsewhere [S41] | LOW |
| **Mycronic DeepReview** (Apr 2024) | PCB 3D AOI | Inside its own AOI | **Yes.** It screens AOI false calls before the review operator. | Operator review remains. Customer retrains on own data. | On machine | 50-90% less review time on false calls. 50-100% fewer false calls on eligible components [S42] | MEDIUM (vendor press release) |
| **Landing AI LandingLens / LandingEdge** | Generic | Can take images from existing cameras | No | Yes: Deploy page with Correct/Incorrect judgement, inspector ID, comments, "Add Images to Build" [S43] | LandingEdge (offline load) and Docker | NXP case: >80% purity on >75% of defects [S44] | MEDIUM. The company's 2026 home page leads with document extraction, and LandingLens is still listed [S45, S46]. |
| **Instrumental** | Electronics assembly | Works with existing AOI and cobot cameras | No. It triages anomalies for engineers. | Monitor triages "units of interest" for review | Not stated | Outcome metrics only, no FRR [S49, S50] | MEDIUM |
| **Elementary** | F&B listed (fill, caps, cans, labels, foreign material) | Own cameras (washdown-rated) | No | No-code labelling. Review queue not documented. | Jetson-based edge with cloud web app [S51, S52] | None published for F&B | MEDIUM |
| **Neurala VIA** | Generic | Uses own GigE camera and PC | No | Training only | Yes, no cloud needed | None | LOW (2020) |
| **Cognex VisionPro DL / In-Sight** | Generic | Rule-based plus DL tools chain in one job (VPDL 2.0 integrates with VisionPro 10) [S9] | No dedicated reject re-check feature found | Labelling UI with heat map and region review [S8] | Yes | None | MEDIUM (docs) |
| **Omron AI Fine Matching** | Generic | Inside FH controller | No | No | Yes | Claims less overdetection, no figure | MEDIUM |
| **Mettler-Toledo Profile Advantage** (metal detection) | Food | Own machine | No | No | Yes | "Up to 95% less false rejects" in meat and poultry [S30] | LOW (vendor) |
| **Key Technology / TOMRA / Bühler Sortex optical sorters** | Bulk food (nuts and similar) | Own machine | **Yes.** Dual-pass sorting loops the reject stream into a second sort. Reverse sort extracts good product from rejects. Dedicated rework lines reclaim rejects [S55]. | Not documented | Yes | ">99.5%" foreign material removal "with very low false reject rates" [S55] | MEDIUM (vendor white paper) |
| **Siemens (Inspekto) S70** | Generic | Own device | No | No | Yes | None | MEDIUM. Siemens acquired Inspekto in Feb 2024, and current availability is UNVERIFIED [S56]. |
| **UnitX FleX** (Dec 2025) | Electronics, batteries | Own imaging | No. It targets overkill via OK/NG criteria. | Not documented | Yes | 9x lower escapes (an escape figure, not FRR) [S57] | LOW |
| **Kitov.ai, Nanotronics nSpec** | Electronics, semiconductor | Own robotic or microscope systems | No | Not documented | Yes | None. nSpec LS Air base price USD 200,000 [S58]. | MEDIUM |
| **Mech-Mind** | 3D vision for robot guidance | n/a | n/a | n/a | n/a | No inspection second-opinion product found (UNVERIFIED) | — |
| **Onto Innovation TrueADC / Applied ExtractAI / KLA / Hitachi** | Semiconductor | **Yes.** It classifies the inspection tool's output. | **Yes**, within the inspection-review pipeline | Yes (see section 3) | Fab tools | See section 3 | MEDIUM |

### 2.2 Reading of the landscape

- **Closest analogs:** Mariner SVI (an overlay on legacy MV to reclaim false rejects), Acquire's CapQ, which trains on examples the primary system wrongly rejected, Mycronic DeepReview (a filter between AOI and operator), and semiconductor ADC. None of these is food-specific, vendor-neutral, cheap-edge, and built on a reject-stream-only workflow with a human queue.
- **The food OEMs** (Krones, Heuft, Mettler-Toledo) fight false rejects *inside* their own inspectors. This competes with our pitch at the next equipment refresh, but it does not address the installed base.
- **Gap:** no product was found that (a) is installed only at the reject bin or reject conveyor, (b) is agnostic to the primary inspector brand, (c) shows a defect heat map to a quality technician, and (d) produces reject-reason analytics to retune the primary system. This is absence of evidence, not proof.

---

## 3. The semiconductor analog: overkill, nuisance defects and ADC review

### 3.1 How the pipeline works

1. **Optical inspection is tuned "hot"** to maximise capture of defects of interest (DOI). This produces large nuisance counts. Patent literature from the KLA family says nuisance can reach about 90% of detected defects in some systems (MEDIUM; exact passage not re-verified) [S59]. Applied Materials states that chipmakers often lower optical sensitivity to cut nuisance, which risks missing critical defects [S60].
2. **Sampling and review.** A defect review SEM revisits sampled defect coordinates (automatic defect review, ADR) and classifies them (automatic defect classification, ADC). Examples are Hitachi CR7300 ("AI-based ADC", 2x throughput over the previous model) [S61], Applied SEMVision G10 (Purity ADC removes nuisance, fleet >2,000 systems) [S62], and KLA eDR7380 (linked to KLA inspectors for "smart sampling") [S63].
3. **AI linking inspection and review.** Applied's ExtractAI (2021) uses SEMVision classifications to train a model that labels the whole optical wafer map. It reduces about 1 million candidate sites to about 1,000 after reviewing about 0.001% of samples [S60].
4. **Confidence-gated human review.** Applied's patent describes this flow. The ADC assigns a class and a confidence. Defects below a per-class threshold are "rejected" to a human terminal. Thresholds come from a target **purity** or a maximum **rejection rate**. Manual results are merged into one report and used to retrain. Rare classes accumulate manual examples until they can be automated [S64].
5. **Back-end packaging inspection** has the die-level analog. Operators review millions of AOI defect images daily, and human error causes "underkill or overkill of die". Deep-learning ADC (Onto TrueADC: CNN, DNN and KNN multi-engine) is used to cut the images operators must see and to filter nuisance at the AOI engine [S65].
6. **PCB assembly** is the same pattern. AOI fails about 3-15% of boards. Operators judge most fails to be false, up to 80-98% in some configurations (patent background, LOW-MEDIUM) [S66]. IPC benchmark false-call rates are in the hundreds to low thousands of PPM per opportunity [S67]. Mycronic's DeepReview sits between AOI and the review operator [S42].

### 3.2 Metrics that transfer

| Metric | Definition (from ADC practice [S64]) | Our equivalent |
|---|---|---|
| Accuracy | Share of all items classified correctly | Not the headline metric |
| **Purity** | Share of items the model *did not* abstain on that it classified correctly | Purity of the "release as good" decision is the safety metric |
| **Rejection (abstain) rate** | Share sent to a human as "unknown" | Workload for the quality technician |
| Overkill / underkill | Good die scrapped / bad die passed | False reject left in bin / defect released |

Published operating points are scarce. One NXP/Landing AI case reports >80% purity on >75% of defects assigned for classification (MEDIUM) [S44]. No vendor publishes guaranteed purity specs (UNVERIFIED).

### 3.3 Lessons for food and beverage

1. **Optimise purity at a chosen abstain rate**, with per-class thresholds (Applied patent [S64]). For food, set the "release" class threshold extremely high and let everything else default to "stay rejected" or "to human".
2. **Keep the primary inspector hot.** Semiconductor fabs keep high sensitivity and push the burden to review. Our pitch is the same: do not de-tune the safety camera. Recover value behind it.
3. **The review stage generates training data and process feedback.** Classified Paretos drive recipe tuning (Applied, Onto) [S60, S62, S65]. For food, reject-reason analytics should be fed back to retune the primary camera. This saving is available even where reintroduction is not.
4. **Humans also err.** In a controlled study, 82 inspectors caught 85% of defective parts but also rejected 35% of good parts (HIGH, peer-reviewed) [S68]. The human queue needs reference images and clear criteria, not just accept/reject buttons.
5. **Difference:** semiconductor overkill recovery carries no food-hygiene law. In food, a released item is subject to the HACCP/FSMS rules in section 5.

---

## 4. Numbers for a presentation

| Metric | Value | Source | Confidence |
|---|---|---|---|
| Full-container inspection false rejection rate (fully equipped inspector) | ~0.3% average | Heuft [S2] | LOW-MEDIUM (vendor) |
| Krones Linatronic AI minimum false rejection rate | 0.3% | Krones [S20] | LOW-MEDIUM (vendor) |
| Cost of 1% FRR, 20,000 bph returnable PET, 2 shifts, 230 days | ~EUR 184,000/yr, container cost only | Heuft [S1] | MEDIUM (vendor worked example) |
| Cost of 0.8% FRR, 50,000 bph, 2 shifts | ~EUR 132,500/yr, container cost only | Heuft [S1] | MEDIUM |
| Value of a 0.1% FRR reduction, 40,000 cph, 1 shift, EUR 0.50/unit | 70,000 units, EUR 35,000/yr per shift. ">EUR 100,000" over 3 shifts. | Heuft [S2] | MEDIUM |
| Empty-bottle inspector reject share, brewery line at 65,000 bph | ~5% rejected. Correctable rejects are manually re-inspected and returned to the washer. | 2014 engineering thesis [S53] | MEDIUM (single plant) |
| Existing machine vision FRR before AI overlay (non-food) | 25% (auto glass), 34% (fabric). After overlay: <1%, <2%. | Mariner [S16] | LOW-MEDIUM |
| Line slowdown to allow human oversight of legacy MV | 15-20% | Mariner [S16] | LOW |
| Direct cost per false reject | ~USD 2.60 (material 1.20, throughput 0.85, handling 0.55) | iFactory blog [S69] | LOW |
| Metal detector false-reject reduction with product-signal suppression | "Up to 95%" (meat and poultry) | Mettler-Toledo [S30] | LOW |
| AOI review time per false call | 30-90 s (60 s working figure) | Vendor blog [S70] | LOW |
| Human visual inspection performance | 85% of defects caught, 35% of good parts falsely rejected | Peer-reviewed study [S68] | HIGH |
| Global food loss, post-harvest up to (not including) retail, processing included | **13.3% (2023)**, vs 13.0% in 2015 | FAO Food Loss Index [S71] | HIGH |
| EU food waste 2024 | 57.9 Mt, ~129 kg/capita. **Manufacturing of food and beverages 19% (~25 kg/capita, ~11 Mt).** | Eurostat via press [S72]. 2022 Eurostat: processing and manufacturing 19%, >11 Mt [S73] | HIGH (secondary for 2024) |
| EU legal target | **-10% food waste in processing and manufacturing by 31 Dec 2030** vs 2021-2023 average (Directive (EU) 2025/1892, in force 16 Oct 2025) | [S74] | HIGH |
| UK food waste | 10.2 Mt (2021-22). Manufacturing ~13% (~1.4 Mt). | WRAP via sector summary [S75] | MEDIUM-HIGH |
| Waste attributable to inspection false rejects | **No published figure found** | — | UNVERIFIED. Must be derived from line data. |
| Recall financial impact | 77% of recalling firms: up to USD 30M. 23%: more. | GMA 2011 survey [S47] | MEDIUM |
| US recall causes 2004-2013 | Pathogens 41%, undeclared allergens 27.4% of 4,900 recall events | USDA ERS EIB-191 [S48] | HIGH |
| UK recall causes 2016-2021 | 597 of 1,036 FSA/FSS recalls allergen-related | Academic manuscript [S76] | MEDIUM |
| Machine vision market | USD 15.83B (2025) to 23.63B (2030), 8.3% CAGR | MarketsandMarkets [S77] | MEDIUM (paid report) |
| Machine vision market (narrower scope) | USD 5.9B (2025) to 8.3B (2030), 7.2% CAGR. 2025 growth 5.2%. | Interact Analysis via Wiley [S78] | MEDIUM |
| Machine vision cameras | USD 6.73B (2025) to 10.19B (2030), 8.6% CAGR | MarketsandMarkets [S79] | MEDIUM |
| AI visual inspection systems | +USD 26.84B over 2024-2029, 38.8% CAGR | Technavio [S80] | LOW (outlier) |

**Worked example template (assumptions, not sourced).** Line rate R units/h × H h/yr × FRR × recovery share × net value per recovered unit. For example, 30,000/h × 4,000 h × 0.5% × 50% recovery × EUR 0.40 gives 300,000 units and EUR 120,000/yr. This only applies if reintroduction is permitted for that reject class (section 5). Otherwise, value comes from labour, retuning and downgrade.

---

## 5. The reintroduction question

### 5.1 Regulatory and standards frame

| Source | What it says (paraphrased unless quoted) | Implication |
|---|---|---|
| **EU Reg. 178/2002, Art. 14** | Unsafe food must not be placed on the market. Where part of a batch is unsafe, the whole batch is presumed unsafe unless a detailed assessment shows otherwise (Art. 14(6)) [S81]. | A release decision needs documented assessment. A reject is "suspect", not automatically unsafe. |
| **EU Reg. 852/2004, Art. 5 and Annex II** | Operators must run HACCP-based procedures, including corrective actions at CCPs. Food must be protected from contamination at all stages [S82]. The Commission Notice 2022/C 355/01 sets out the PRP/OPRP/CCP tiering [S83]. | The primary camera's role in the HACCP plan (CCP, OPRP or quality check) decides what is allowed. |
| **Codex CXC 1-1969 (rev. 2022)** | Corrective action includes segregating affected product and determining its disposition [S84]. | Disposition is a defined, documented decision. |
| **US 21 CFR 117.80(b)(5), (c)(5)** | Material scheduled for rework must be identified as such. Rework must be handled to protect against allergen cross-contact and contamination [S34]. | Recovered items count as rework: identified, segregated, traceable. |
| **US 21 CFR 117.80(c)(9)** | Adulterated food must be disposed of, or reconditioned by a proven method, or "reexamined and subsequently found not to be adulterated" [S34]. | A validated re-examination is a recognised route, but only for items that can be shown not adulterated. |
| **US 21 CFR 117.150** | Corrective actions must ensure "All affected food is evaluated for safety" and is prevented from entering commerce if it cannot be shown not adulterated or misbranded [S85]. | If the camera is a preventive control, every reject needs a safety evaluation before release. |
| **US 21 CFR 113.60** (low-acid canned foods) | Gross closure defects must be recorded with corrective action. Visual seam examinations are required [S86]. | Seam/closure rejects are a food-safety class. No AI-only release. |
| **ISO 22000:2018, 8.9.4** | CCP-failed product must not be released. OPRP-failed product may be released only with other evidence of conformity. Otherwise it is reprocessed, redirected to other use, or destroyed (secondary quotes) [S87]. | An independent AI check plus human sign-off could count as "other evidence" for OPRP or quality rejects only. Needs FSMS team validation. |
| **BRCGS Food Safety Issue 9, 3.8.1** | Non-conforming product needs a defined decision-maker. Listed options are destruction, reworking, downgrading to an alternative label, or acceptance by concession (forum quote of clause) [S88]. | The standard explicitly allows rework and downgrade. "Return to line" is a rework decision with an accountable owner. |
| **BRCGS 3.9.4** | Traceability must be maintained wherever rework or reworking operations are performed [S89]. | Recovered items keep lot identity and must be logged. |
| **BRCGS 5.3.5** (allergens) | Rework containing allergens must not go into products that don't already contain them. Rework goes like into like [S90]. | Recovered packs go back to the same SKU and lot only. |
| **BRCGS 4.10.3.x** (foreign-body detection) | Rejects go to a secure unit accessible only to authorised personnel. Belt-stop systems are seen as higher risk of rejects re-entering the process [S40, S91]. | Foreign-body rejects are explicitly designed **not** to re-enter casually. |
| **BRCGS 6.2.4** (on-line vision for labels) | Vision equipment checking labels must be correctly set up and capable of alerting or rejecting out-of-spec packs [S92]. | The standard regulates the primary camera, not a second stage. Any second stage would need its own validation. |
| **IFS Food v8** | Flow diagrams must include rework and reprocessing (2.3.4.1). Traceability and CCP monitoring are knock-out requirements [S93]. | The recovery loop must appear on the HACCP flow diagram. |
| **ISO/TS 22002-1** (FSSC 22000) | Allergen rework may go only into products containing the same allergen. Rework segregation must be documented [S94]. | Same as BRCGS. |
| **USDA FSIS** | Rejects should be treated as a process step in the flow diagram, with procedures for disposition [S95]. | The reject station is in scope for hazard analysis. |
| **Industry practice** | Mettler-Toledo: re-running metal-detector rejects and releasing them if nothing is detected is the wrong practice. Rejects should be locked away and investigated [S96]. Some plants still re-check rejects under QA control, while others forbid re-passing [S97]. | Expect auditors to challenge automated release of safety rejects. |

### 5.2 Verdict per product class

| Product class | Primary reject reason | Can it go back into the good stream? | Conditions |
|---|---|---|---|
| **Sealed rigid containers** (bottles, cans, jars) after filling and closing | Cosmetic or quality: label skew or wrinkle, print smudge, cap colour, dirty exterior, droplet artefacts | **Yes, conditionally.** Product contact surfaces are untouched. Treat it as a documented rework/re-inspection step. | Defined in the HACCP flow diagram. QA-authorised procedure. Lot and traceability preserved. Return to the same SKU/lot before or at the case packer. Item not physically damaged by ejection. Human sign-off for anything below the release threshold. Audit log of every release. |
| Sealed rigid containers | Fill level (under/over fill) | **Rarely.** Legal metrology (average-quantity rules, TNE) applies, and a true underfill cannot be fixed without opening. | Release only if re-measurement proves a false reading (for example foam). Otherwise rework or downgrade. |
| Sealed rigid containers | Closure/seam integrity, tamper band, glass defects, foreign body | **No automatic return.** These are food-safety controls (21 CFR 113.60 for LACF, glass hazard). | The AI may only prioritise for QA or destructive testing. |
| **Sealed flexible packs and trays** (MAP, pouches, flow-wrap) | Seal contamination or leak | **No direct return.** Seal integrity is typically an OPRP/CCP. | AI triage to a leak test (CO2 or pressure) or QA. Release only with independent physical evidence (ISO 22000 8.9.4). |
| Any packaged product | Label content, allergen statement, date code wrong or unreadable | **Only after human verification.** Allergen labelling errors are the top UK recall cause and a leading US one [S48, S76]. | AI can separate "unreadable but correct" from "wrong text". Final release is human. Relabel/over-label is the usual rework path. |
| **Empty containers** (returnable glass, pre-fill) | Dirt, residue, scuffing | **Yes.** This is established practice: rejected empties are re-washed and re-inspected [S53]. | Must pass the primary inspector again after re-wash. |
| **Bulk unpackaged food on sorting lines** (nuts, grains, produce) | Colour or shape defects, foreign material | **Yes, within designed equipment.** Dual-pass, reverse sort and rework lines are standard [S55]. | The recovered stream goes through a sorter again. This is "re-sort", not manual return. |
| **Exposed discrete food** (bakery, confectionery before wrap, meat portions) that has dropped into a reject bin | Shape, colour, size, topping | **No return to the primary flow.** Bin contact means contamination risk under 852/2004 Annex II and 117.80. | Allowed outlets: controlled rework into the same product (like into like, same day, traceable, under ~10% inclusion is common practice [S98]), downgrade or B-grade, redistribution, animal feed. |
| Any product | Foreign-body detection (metal, X-ray) reject | **No.** | Lock, investigate, root cause [S40, S96]. |

### 5.3 Where the saving comes from when direct return is not allowed

1. **Rework and relabel paths** (BRCGS 3.8.1, 117.80): over-label, repack or re-case good contents. The AI decides which rejects are worth reworking.
2. **Downgrade / B-grade / concession** routing (BRCGS 3.8.1). The AI grades severity, for example cosmetic vs safety.
3. **Retuning the primary inspector.** Classified reject Paretos show which rules over-fire, as in the semiconductor practice in section 3.3 and CapQ's "train in false rejects" [S38]. This lowers the reject rate at source without de-tuning safety checks. **It works in every product class and is likely the most defensible value.**
4. **Labour.** It replaces or speeds up existing manual re-inspection: one operator per line in the brewery case, and lines slowed 15-20% for human oversight in Mariner's cases [S16, S53].
5. **Compliance evidence.** Every reject gets an image, a reason and a disposition log. That supports traceability (BRCGS 3.9), reject reconciliation, and the EU 2030 manufacturing waste target [S74].

**Overall verdict.** Lead the pitch with "sealed rigid packaging, quality-class rejects" for reintroduction. Present "retune the primary camera" and "faster disposition with an audit trail" as the universal value. Never position the tool as releasing food-safety rejects (seal, closure, foreign body, allergen label) without human or physical verification.

---

## 6. What currently happens to rejects

| Practice | Where seen | Evidence |
|---|---|---|
| Diverted into a **locked reject bin**, then QA investigation | Metal detection and X-ray (BRCGS 4.10.3) | [S40, S96] |
| **Manual re-inspection and return** of correctable rejects | Returnable empty glass: one operator inspects rejects, and usable bottles go back to the washer on a conveyor [S53]. Fabric: labour for "re-inspection to reclaim rejected fabric" [S16]. | MEDIUM |
| **Re-run through the inspector** | Cheese plant re-running product through metal detectors was "very labor intensive" and was reduced after a move to X-ray [S99]. Some sites forbid re-passing [S97]. | MEDIUM |
| **Automated re-sort** | Nuts and bulk sorting (dual-pass, reverse sort, rework line) [S55] | MEDIUM |
| **Rework / relabel / repack** | Common under BRCGS 3.8.1 and 21 CFR 117.80. Contract rework firms offer over-labelling, date-code correction and reclaiming undamaged product [S100]. | MEDIUM |
| **Downgrade / concession / sale as substandard** | BRCGS 3.8.1 options [S88] | MEDIUM |
| **Redistribution** of edible surplus | Top of the UK food waste hierarchy after prevention [S101] | HIGH |
| **Animal feed** ("former foodstuffs") | EU Guidelines 2018/C 133/02. Feed hygiene rules and site registration apply [S102, S101]. | HIGH |
| **Anaerobic digestion / composting** | Below feed in the hierarchy. Counted as waste [S101]. | HIGH |

**Frequency and staffing data.** No industry survey quantifying how often rejects are human re-checked or how this is staffed was found (UNVERIFIED). The available data points are single cases: one operator per line in the brewery case [S53], and 15-20% line slowdown for human oversight in Mariner's cases [S16]. PCB analogs put manual review at about 3-15% of boards [S66]. Plan primary research (site interviews) to close this gap.

---

## 7. Human-in-the-loop review UX in existing QC tools

### 7.1 Documented layouts

**Cognex VisionPro Deep Learning, Red Analyze review** [S8]
- Image display with a **border colour per verdict**: green = Good, red = Bad, **grey = "Unable to discern"** (score between the Good and Bad thresholds). This is a built-in third "uncertain" state.
- **Anomaly score** in the bottom-left. A **heat map (deviation map)** overlay plus **polygon regions** outlining defects. A label band at top-right shows the human label.
- **Region list** under the image, with two tables, *Labels* (human) and *Markings* (model). Columns: Score (max pixel defect probability), Position, Area (px), Coverage (label/prediction overlap). Clicking a row highlights the region.
- Overlay toggles: **Alt+Left / Alt+Right**, right-click menu, Labels/Markings checkboxes. Right-click "Accept Region" turns a prediction into a label (release notes [S9]).

**Cognex In-Sight filmstrip** [S6]
- A horizontal strip of recorded pass and fail images (up to 10,000 on the PC). Yellow highlight marks the current image. Images can be played back through the job to re-evaluate.

**LandingLens Deploy page / Image Details** [S43]
- Grid of predicted images from cloud, LandingEdge or Docker. Images cannot be deleted, to keep records accurate.
- Image Details pop-up: **confidence score** per detection, metadata (incl. Media ID), and **Human Judgment** (Inspector ID = reviewer email, Judgment OK/NG, free-text Comment). Buttons: **Correct / Incorrect**. Tools: full-screen, brightness/contrast enhancement, show labels, arrow-key navigation, hotkeys, download original.
- Feedback loop: select images, then **"Add Images to Build"**. They arrive *unlabelled*, so a human must approve every label before retraining. A video walkthrough is at [S103].

**Mariner Spyglass in-factory app** [S16]
- Full image of the inspected item **plus a magnified crop of the highest-confidence defect**, a Pass/Fail verdict, and a **Tag** button for operators to correct misclassifications, which feeds retraining.

**Onto Innovation Discover Review** (semiconductor) [S104]
- **Gallery mode**: select one or many images by wafer or die for fast batch classification and comparison. Wafer-map viewer.
- **Concurrent multi-user review** with overwrite protection, and **partial review status** so shifts can hand over mid-lot.

**Applied Materials ADC/manual integration** (patent) [S64]
- Low-confidence defects are routed to an operator terminal. Thresholds are per class. Automatic and manual results are merged into one report. Manual results are counted per rare class until the class can be automated.

**Instrumental Monitor** [S50]
- Two triage modes: **Similarity** (find more units like a flagged one, with an adjustable similarity threshold) and **Anomaly** (flag units unlike past normal units). Triaged units go to engineers for review.

**Time per decision.** The only figure found is 30-90 s per AOI false call (vendor blog, LOW) [S70]. No food-specific benchmark was found (UNVERIFIED).

### 7.2 Patterns a UI designer can lift (synthesis, not a sourced spec)

| Element | Pattern source | Suggested form |
|---|---|---|
| Three-state verdict | Cognex green/red/grey [S8], ADC "unknown" [S64] | Release candidate / Confirm reject / Needs human |
| Defect localisation | Cognex heat map + polygons, Mariner magnified crop [S8, S16] | Heat map toggle, a zoomed crop of the top region, and a side-by-side known-good reference image |
| Context | Applied report, Onto wafer map [S64, S104] | Primary inspector's reject reason code, lot/batch, time, line, SKU |
| Decision capture | LandingLens judgement, inspector ID, comment [S43] | Two large buttons with keyboard shortcuts, mandatory reason code on override, auto-captured operator ID |
| Batch triage | Onto gallery mode [S104] | Grid view of same-reason rejects for bulk "confirm reject" (never bulk "release" for safety classes) |
| Feedback | LandingLens "Add to Build", Mariner Tag [S43, S16] | Human decisions queued as unlabelled training candidates and approved by a QA lead before retraining |
| Shift handover | Onto partial review status [S104] | Queue persists with status and age, and an alert when the reject bin backlog exceeds hold limits |
| Analytics | Classified Pareto (semiconductor) [S60] | Daily Pareto of "false reject by primary rule" to drive retuning |

---

## 8. Sources

All accessed 2026-10-17.

| ID | URL | Supports |
|---|---|---|
| S1 | https://heuft.com/en/actual-news/news/news-archiv/money-in-the-recycling-bin | False rejection cost examples (1% and 0.8% FRR). Up to 50% reduction claim. |
| S2 | https://beverage.heuft.com/nl/actual-news/news-overview/news-archive/every-mill-counts | 0.3% average FRR. 0.1% reduction worth EUR 35k per shift-year. |
| S3 | https://drivesandcontrols.ca/product-news/cognex-in-sight-3800-vision | In-Sight 3800 edge learning plus rules, 125 fps, IP67 |
| S4 | https://www.automate.org/vision/news/cognex-introduces-world-s-first-industrial-smart-camera-powered-by-deep-learning | D900 deep learning in camera, no PC |
| S5 | https://support.cognex.com/docs/is_621/web/EN/ise/Content/Communications_Reference/OPCUA-Intro.htm | In-Sight OPC UA server models and firmware |
| S6 | https://docs.cognex.com/is_611/web/EN/ise/Content/User_Interface/Filmstrip_EasyBuilder_PC.htm | Filmstrip recording of pass/fail images, 10,000 limit |
| S7 | https://www.ebay.de/itm/357529362870 | Used D900 price (LOW) |
| S8 | https://docs.cognex.com/deep-learning_321/web/EN/deep-learning/Content/deep-learning-Topics/training-red-foc-unsup/marking.htm | Red Analyze review UI: score, heat map, grey "unable to discern", region list |
| S9 | https://docs.cognex.com/deep-learning_332/EN/CognexDeepLearningReleaseNotes.pdf | VPDL release notes: Accept Region, Green Classify heat map, VisionPro integration |
| S10 | https://s2.q4cdn.com/773500753/files/doc_news/2025/Cognex-Introduces-OneVision-A-Breakthrough-Cloud-Platform-for-AI-Powered-Machine-Vision.pdf | OneVision cloud platform, June 2025, rollout plan |
| S11 | https://www.keyence.com/products/vision/vision-sensor/iv4/ | Keyence IV4 built-in AI |
| S12 | https://invision-news.de/?p=904612 | Keyence VS series and VS-G controller specs |
| S13 | https://assets.omron.com/m/7f976b753dc58298/original/FH-Series-Datasheet.pdf | Omron FH AI tools, EtherCAT/PLC integration |
| S14 | https://mvpromedia.com/sick-expands-its-comprehensive-solution-with-nova-powered-smart-camera/ | SICK Inspector83x on Nova, dStudio training, licences |
| S15 | https://embeddedcomputing.com/technology/ai-machine-learning/computer-vision-speech-processing/basler-and-syslogic-deliver-rugged-ai-vision-to-the-edge | Basler plus Syslogic Jetson rugged edge (2026) |
| S16 | https://marketplace.intel.com/file-asset/a5Y3b00000088fIEAQ_a5b3b000000LWQfAAO | Mariner SVI brochure: FRR before/after, line slowdown, in-factory app UI |
| S17 | https://www.zebra.com/content/dam/zebra_dam/en/spec-sheets/ns42-smart-vision-sensor-spec-sheet-en-us.pdf | Zebra NS42 DL-OCR and anomaly detection. Aurora naming. |
| S18 | https://www.mvtec.com/knowledge-base/technologies/deep-learning/methods/anomaly-detection | HALCON Global Context Anomaly Detection |
| S19 | https://teledynevisionsolutions.com/company/news/dalsa/teledyne-introduces-next-generation-ai-powered-smart-camera-for-industrial-automation-and-inspection | BOA3 smart camera AI |
| S20 | https://www.packagingnews.com.au/industry-4-0-and-iiot/krones-combats-false-bottle-rejects-with-ai-inspection | Krones Linatronic AI: droplets/foam, >99.9%, 0.3% min FRR |
| S21 | https://www.foodmanufacturing.com/products/test-measurement-inspection/product/22934982/eagle-product-inspection-eagle-announces-new-xray-inspection-system-for-pumped-products | Eagle X-ray lower false rejects (qualitative) |
| S22 | https://www.oemmagazine.org/leaders-new/integrated-equipment/inspection-detection/article/13277010/software-system-gives-360-degree-view-of-production-line | Mettler ProdX logs reject time, place and reason |
| S23 | https://overview.ai/shop | OV20i USD 9,450, no licence fees |
| S24 | https://opcfoundation.org/news/press-releases/vdma-releases-opc-ua-companion-specifications-robotics-machine-vision/ | OPC UA Vision (VDMA 40100): results vendor-specific |
| S25 | https://hwbusters.com/news/nvidia-jetson-prices-jump-up-to-101-the-249-orin-nano-super-is-now-399/ | July 2026 Jetson price increases |
| S26 | https://jetsonhacks.com/2024/12/17/jetson-orin-nano-super-developer-kit/ | Orin Nano Super launch at USD 249 |
| S27 | https://openelab.io/collections/seeed-studio-recomputer-discount | reComputer Industrial J4012 EUR pricing |
| S28 | https://www.storagereview.com/news/onlogic-karbon-520-brings-core-ultra-and-npu-to-the-edge | Karbon 520 Core Ultra NPU, rugged |
| S29 | https://www.notebookcheck.net/Onlogic-readies-hybrid-cooling-rugged-Karbon-524-mini-PCs-with-Intel-Ultra-Core-100H-CPUs-and-Nvidia-SFF-Ada-GPUs.1109626.0.html | Karbon 524 USD 2,997 / 4,843 |
| S30 | https://www.mt.com/au/en/home/library/know-how/product-inspection/overcoming-product-effect.html | Product effect causes false rejects. Sensitivity reduction trade-off. Up to 95% claim (via trade press). |
| S31 | https://www.controleng.com/?p=94132 | Engineering time dominates PC-based vision cost |
| S32 | https://www.netapp.com/media/77521-IT-OT-Convergence-Research-Report.pdf | IT/OT convergence survey (23%) |
| S33 | https://www.paloaltonetworks.com/cybersecurity-perspectives/the-air-gap-is-dead | OT mostly firewalled, not air-gapped (vendor view) |
| S34a | https://www.pmmi.org/blog/from-the-shop-floor-to-the-cloud-securing-your-smart-manufacturing-data | Edge and cloud connector patching risk |
| S34 | https://www.law.cornell.edu/cfr/text/21/117.80 | 21 CFR 117.80 rework and adulterated food reconditioning |
| S35 | https://docs.landing.ai/landingedge/manage-inspection-points | LandingEdge offline model load, image saving |
| S36 | https://embeddedcomputing.com/technology/ai-machine-learning/computer-vision-speech-processing/neurala-announces-via-software-to-help-manufacturers-improve-quality-inspection | Neurala VIA on existing hardware, no cloud |
| S37 | https://blogs.nvidia.com/blog/2021/12/20/mariner-visual-inspection | Mariner 30x fewer misclassifications, overlay on existing MV |
| S38 | https://www.profoodworld.com/home/news/13278991/artificial-intelligence-key-to-inspection-system | CapQ False Rejection Suppression: train in false-rejected examples |
| S39 | https://www.automate.org/products/pleora-technologies/ai-gateway | Pleora AI Gateway inline between camera and existing app |
| S40 | https://ifsqn.com/forum/index.php/topic/45255-brcgs-41032-metal-detection-rejects-internal-auditor-interpretation-disagreement | BRCGS 4.10.3.2 secure reject unit (forum quote of clause) |
| S41 | https://averroes.ai/blog/how-to-reduce-false-positives | Averroes FR reduction claims, inspector dashboard routing |
| S42 | https://storage.mfn.se/667898b5-414f-4ce9-83c5-86a795e97ad6/press-release-deepreview-20240403.pdf | Mycronic DeepReview press release |
| S43 | https://landinglens.docs.landing.ai/historical-data | LandingLens Deploy/Image Details UI and feedback loop |
| S44 | https://nafems.org/events/nafems/2023/artificial-intelligence-and-machine-learning-for-manufacturing/landing-ai-and-nxp-abstract | NXP/Landing AI ADC purity result |
| S45 | https://landing.ai/ | Company home page focus on agentic document extraction (2026) |
| S46 | https://landing.ai/landinglens | LandingLens still offered, cloud and edge |
| S47 | https://www.foodsafetynews.com/2017/12/food-safety-compromises-can-carry-costly-consequences/ | GMA 2011 recall cost survey |
| S48 | https://primary.ers.usda.gov/webdocs/publications/88497/eib191-summary.pdf | US recall causes 2004-2013 |
| S49 | https://instrumental.com/ | Instrumental current positioning and outcome metrics |
| S50 | https://instrumental.com/?p=696 | Instrumental Monitor similarity/anomaly triage |
| S51 | https://www.elementaryml.com/solutions/food-and-beverage | Elementary F&B inspections, washdown |
| S52 | https://blogs.nvidia.com/?p=48238 | Elementary Jetson edge plus cloud app |
| S53 | https://noesis.uis.edu.co/entities/publication/c849a0d4-f93f-47f4-9b14-763c50bea2b3/full | Brewery: ~5% empty-bottle rejects at 65,000 bph, manual re-inspection |
| S55 | https://www.key.net/wp-content/uploads/2022/09/WP_Sorting-Nuts_PDF.pdf | Dual-pass, reverse sort, rework lines for rejects |
| S56 | https://www.vision-systems.com/factory/manufacturing/article/14305479/siemens-acquires-automated-visual-inspection-startup | Siemens acquisition of Inspekto |
| S57 | https://www.automateshow.com/exhibitor-news/unitx-launches-flex-the-world-s-most-accurate-ai-visual-inspection-system-delivering-9x-lower-escape-rates-and-3x-faster-deployment | UnitX FleX claims |
| S58 | https://www.businesswire.com/news/home/20240711505764/en/Nanotronics-Unveils-Groundbreaking-Gen-V-AI-Model-and-Two-New-Affordable-Inspection-Products-Set-to-Transform-Global-Manufacturing | Nanotronics nSpec pricing |
| S59 | https://patents.google.com/patent/US11694327 | KLA nuisance filtering patent family (nuisance share claim; exact passage UNVERIFIED) |
| S60 | https://venturebeat.com/business/applied-materials-brings-ai-and-big-data-into-semiconductor-inspection-machines | ExtractAI: ~1M to ~1,000 sites, 0.001% reviewed, sensitivity trade-off |
| S61 | https://www.hitachi-hightech.com/sg/en/products/semiconductor-manufacturing/cd-sem/dr-sem/cr7300.html | Hitachi CR7300 AI ADC |
| S62 | https://www.appliedmaterials.com/us/en/semiconductor/products/processes/metrology-and-inspection/semvision-g10-defect-analysis.html | SEMVision G10 Purity ADC, >2,000 fleet |
| S63 | https://ir.kla.com/news-events/press-releases/detail/43/kla-announces-new-defect-inspection-and-review-portfolio | KLA eDR7380 linkage to inspectors |
| S64 | https://patents.google.com/patent/US10043264B2/en | ADC confidence routing to human, purity/rejection definitions, retraining |
| S65 | https://ontoinnovation.com/library/using-deep-learning-adc-for-defect-classification-for-automatic-defect-inspection | Onto TrueADC, back-end overkill/underkill from human review |
| S66 | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12704464 | PCB AOI 3-15% board fails, high false share (patent background) |
| S67 | https://www.ipc.org/system/files/technical_resource/E18%26S04-1.pdf | IPC AOI false call benchmarking (PPM) |
| S68 | https://www.sandia.gov/research/publications/details/visual-inspection-reliability-for-precision-manufactured-parts-2015-12-01/ | Human inspectors: 85% hits, 35% false rejects |
| S69 | https://ifactoryapp.com/blog/ai-false-reject-reduction-software | USD 2.60 per false reject (LOW) |
| S70 | https://enaovision.com/blog/ai-defect-detection-smt-lines | 30-90 s per false call (LOW) |
| S71 | https://www.fao.org/sustainable-development-goals-data-portal/data/indicators/1231-global-food-losses | Global food loss 13.3% (2023) |
| S72 | https://www.brusselstimes.com/2338497/eu-food-waste-rises-to-57-9-million-tonnes-households-driving-the-surge | EU 2024 food waste, manufacturing 19% |
| S73 | https://ec.europa.eu/eurostat/statistics-explained/index.php/Food_waste_and_food_waste_prevention_-_estimates | Eurostat sector shares (page now redirects; 2022 figures via search snippet) |
| S74 | https://food.ec.europa.eu/food-safety/food-waste/eu-food-waste-relevant-legislation/food-waste-reduction-targets_en | EU binding 2030 targets, Directive (EU) 2025/1892 |
| S75 | https://www.publicsectorcatering.co.uk/sites/default/files/attachment/news/Food-Waste-and-Food-Surplus-Key-Facts-July-2025.pdf | WRAP UK 10.2 Mt, manufacturing share |
| S76 | https://clok.uclan.ac.uk/id/eprint/39777/1/Manuscript%20.pdf | UK recalls 2016-2021, allergen share |
| S77 | https://www.marketsandmarkets.com/PressReleases/machine-vision-systems.asp | Machine vision market 2025-2030 |
| S78 | https://www.wileyindustrynews.com/en/news/global-machine-vision-market-set-to-grow-to-8-3-billion-dollars-by-2030 | Interact Analysis machine vision forecast |
| S79 | https://www.bisinfotech.com/machine-vision-camera-market-set-to-grow-from-usd-6-73-billion-in-2025-to-usd-10-19-billion-by-2030-marketsandmarkets/ | Machine vision camera market |
| S80 | https://www.technavio.com/report/ai-visual-inspection-system-market-industry-analysis | AI visual inspection market (LOW) |
| S81 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02002R0178-20220701 | Reg. 178/2002 Art. 14 (page did not render; paraphrased from the regulation) |
| S82 | https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32004R0852 | Reg. 852/2004 Art. 5 HACCP, Annex II (not fetched; paraphrase) |
| S83 | https://www.FSAI.ie/getattachment/35d81414-f56c-4111-8911-98c62b3b1dba/OJ_C_2022_355_FULL_EN_TXT.pdf?lang=en-IE | Commission Notice 2022/C 355/01 FSMS guidance |
| S84 | https://dinalexicon.micronutrientforum.org/lexicon-terms/corrective-action/ | Codex corrective action definition (secondary) |
| S85 | https://www.law.cornell.edu/cfr/text/21/117.150 | 21 CFR 117.150 corrective actions |
| S86 | https://www.law.cornell.edu/cfr/text/21/113.60 | 21 CFR 113.60 container closure |
| S87 | https://ifsqn.com/forum/index.php/topic/44427-can-i-release-material-that-has-exceeded-a-ccp-limit-8942-and-8943 | ISO 22000 8.9.4.2/8.9.4.3 (secondary quotes) |
| S88 | https://ifsqn.com/forum/index.php/topic/31532-acceptance-by-concession/?ampmode=1 | BRCGS 3.8.1 disposition options (forum quote) |
| S89 | https://public-library.safetyculture.io/products/brcgs-v9-3-9-traceability-r7l0ldbqw49nv1bn | BRCGS v9 3.9.4 rework traceability (template) |
| S90 | https://techni-k.co.uk/process-control/material-handling-process-control/ | BRCGS 5.3.5 rework, like into like |
| S91 | https://www.ifsqn.com/forum/index.php/topic/29705-metal-detector-secured-reject-box/?ampmode=1 | Interpretation: belt-stop re-entry risk, locked container |
| S92 | https://safetyculture.com/library/manufacturing/brcgs-v9-6-2-labelling-and-pack-control-5a1oaxrqf3smnsbr | BRCGS 6.2.4 on-line vision equipment |
| S93 | https://www.dqsglobal.com/en/learn/blog/ifs-food-version-8-all-changes-of-the-revision-at-a-glance | IFS Food v8 2.3.4.1 rework in flow diagram |
| S94 | https://www.evs.ee/en/iso-ts-22002-1-2009 | ISO/TS 22002-1 rework clause (allergen rework) |
| S95 | https://www.fsis.usda.gov/policy/fsis-directives/7310.5 | FSIS: rejects as a process step |
| S96 | https://bakeryandsnacks.com/Article/2013/09/24/Mettler-Toledo-metal-detection-reject-bins-PowerPhasePRO | Mettler: re-running rejects is wrong practice |
| S97 | https://www.ifsqn.com/forum/index.php/topic/46981-metal-detector-rejects | Mixed site practice on re-passing rejects |
| S98 | https://www.food-safety.com/articles/10911-how-to-handle-non-conforming-product | Disposition options, rework like into like, ~10% limit |
| S99 | https://www.dairyreporter.com/Article/2017/09/06/Schuman-Cheese-reduces-false-rejects-with-X-ray-inspection%20 | Cheese plant re-running rejects was labour intensive |
| S100 | https://www.merriwa.org.au/rework-quality-inspection-services | Contract rework: relabel, date code, reclaim |
| S101 | https://www.gov.uk/government/publications/food-and-drink-waste-hierarchy-deal-with-surplus-and-waste/food-and-drink-waste-hierarchy-deal-with-surplus-and-waste | UK food waste hierarchy |
| S102 | https://circulareconomy.europa.eu/platform/sites/default/files/feed_guidelines.pdf | EU former foodstuffs feed guidelines 2018/C 133/02 |
| S103 | https://docs.landing.ai/videos/continuous-learning-after-model-deployment-video | LandingLens continuous learning demo video |
| S104 | https://ontoinnovation.com/products/discover-review/ | Onto Discover Review gallery, multi-user, partial review (page now 404 on one path; features from search snippet) |

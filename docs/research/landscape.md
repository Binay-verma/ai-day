# Landscape Report: Explainable Edge AI Copilot for Smart Emergency Response

Prepared 2026-10-23. Scope: products and research for detecting falls, smoke/fire, and abnormal crowd behaviour on edge hardware, plus explainability, calibration, privacy, and human-in-the-loop alerting. Every claim cites a source URL. Where a fetch failed or a page gave no numbers, the report says so rather than guessing.

---

## 0. Bottom line

1. **"Plain-language alert descriptions" is not new on its own.** Verkada, Milestone, Ambient.ai, Coram, and NVIDIA VSS all generate natural-language summaries or accept natural-language alert rules, mostly using VLMs on a GPU or in the cloud (sections 1.1–1.4). Saliency heatmaps on YOLO/CNN fire and fall detectors have also been done many times (section 2.7).
2. **What is largely missing:** explanations that are **provably faithful** (built from the actual decision evidence, not narrated afterward by an LLM), **automatically checked for hallucination**, **counterfactual** ("what would have to change for no alert"), and **calibrated** (a stated false-alarm guarantee). These would run on a **2-core CPU** across **several hazards at once**. Each piece exists somewhere in isolation. We found no source that combines them for emergency alerting.
3. **Benchmarks are saturated or protocol-sensitive.** UR Fall and Le2i accuracies are reported at 96–99.6%. UMN crowd AUC is about 0.99. D-Fire YOLOv8n mAP@0.5 is reported anywhere from 62.5% to 77.3% depending on the paper. Chasing accuracy SOTA is a poor use of the team's time. Better to compete on **faithfulness, calibration, latency, energy, and cross-dataset robustness**.

---

## 1. Existing products and deployed systems

### 1.1 Summary table

| Vendor / system | Hazards | Where it runs | What it claims about explanation | Source |
|---|---|---|---|---|
| **Verkada** (activity detection) | Fall, climbing (launched Oct 9 2025 on select 3rd-gen cameras) | On-camera multi-frame video model (inferred) | None in the launch post | https://www.verkada.com/uk/blog/ai-powered-activity-detection/ |
| **Verkada** (slip & fall solution) | Slip/fall | "On-camera" | AI-generated incident summaries; natural-language rules such as "person near wet floor"; no accuracy figures | https://www.verkada.com/solutions/slip-and-fall/ |
| **Verkada SV11** sensor | PM2.5, TVOC, vape index, temp, humidity, noise, motion | Cloud-managed; paired with a camera | Per-sensor thresholds; no fusion or explanation described | https://www.verkada.com/blog/sv11-environmental-sensor/ |
| **Scylla** | Slip/fall (3–5 s clips), anomaly, behaviour | Not stated | "Up to 99.1%" / "minimum 99%" accuracy, metric undefined; alerts include "all the crucial information"; "self-learning from real-time feedback" | https://www.scylla.ai/slip-fall-detection/ |
| **Avigilon UAD/UMD** | Unusual speed, direction, or location of people/vehicles | Edge (camera) | Learns normal scene behaviour; "Focus of Attention" UI highlights events. No per-alert rationale beyond the attribute | https://daywireless.com/avigilon/unusual-activity-detection.htm |
| **Hikvision + iThermAI** (2024) | Fire and smoke | Camera-based | Announcement only. The page fetch returned no content, so details are unverified | https://www.hikvision.com/en/newsroom/latest-news/2024/hikvision-and-ithermaI-work-together-to-provide-AI-based-fire-and-smoke-detection-solutions/ |
| **Gorilla IVAR** | 50+ analytics: intrusion, loitering, sudden running, crowd (fall and fire not listed) | "Edge AI video analytics" | None | https://www.milestonesys.com/technology-partner-finder/gorilla-technology/ivar---intelligent-video-analytics-recorder |
| **Milestone XProtect GenAI plug-in** (beta Nov 2025, with NVIDIA) | Alarm validation across event types | On-prem or cloud | Natural-language incident summaries and false-alarm filtering; claims up to 30% less alarm fatigue | https://www.milestonesys.com/company/news/press-releases/xprotect-video-review-and-response/ |
| **Milestone VLM** (Dec 2025) | Traffic | Cloud / HTTPS API | Text summaries of clips; trained on about 75k hours of traffic video | https://www.milestonesys.com/company/news/press-releases/milestone-launches-vision-language-model/ |
| **Ambient.ai Pulsar** (GA Nov 19 2025) | Open-set security threats | On-prem edge appliance (NVIDIA) | "Reasoning VLM"; vendor claims up to 95% false-alarm reduction; no explanation mechanism described | https://www.ambient.ai/press/ambient-unveils-ambient-pulsar-reasoning-vlm |
| **Coram AI** | Slip/fall, firearms, PPE | Edge first stage, cloud MLLM verification | Plain-English custom alerts. Patent US12367677B1 covers an edge CLIP-embedding classifier, then a cloud multimodal LLM prompted with "does the video contain…?", then optional human annotator review and retraining on labelled clips | https://www.coram.ai/post/ambient-ai-competitors (vendor-biased), https://patents.google.com/patent/US12367677 |
| **NVIDIA VSS blueprint** | Any (prompted) | x86 GPU, DGX, Jetson AGX Orin/Thor | CV alerts "verified with VLM to reduce false positives"; summarisation and Q&A | https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization |
| **NVIDIA TAO ActionRecognitionNet** | Includes a `fall_floor` class (123 training videos) | DeepStream/TensorRT; Jetson Nano at 30 FPS (2D-RGB) | None (classifier) | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet |
| **NVIDIA PoseClassificationNet** | Skeleton actions (ST-GCN); sample classes are sitting/standing/walking/jumping etc. | DeepStream + Triton | None | https://docs.nvidia.com/tao/tao-toolkit/text/pose_classification/pose_classification.html |
| **Intel fire-and-smoke-detection** (Hugging Face) | Fire/smoke via Qwen2-VL-2B yes/no prompt | OpenVINO INT4/INT8 on CPU, GPU, or NPU (Core Ultra) | Text answer only; **no accuracy metrics published** | https://huggingface.co/Intel/fire-and-smoke-detection |
| **Edge Impulse** community projects | Fire (image + temperature fusion on an MCU); fall (FOMO on Raspberry Pi 4) | MCU / Raspberry Pi | None. Fusion demo had 120 samples and "100%" validation accuracy; fall demo had 98% training accuracy | https://www.edgeimpulse.com/blog/sensor-fusion-ignites-next-gen-fire-detection/, https://hackster.io/coderscafe/fall-detection-using-computer-vision-for-industrial-workers-b3a420 |
| **Teton.ai** (care homes) | Fall-risk movement rules | On-device in each room; depth silhouettes | **The most transparent product found.** Each alarm rule shows its measured catch rate and false-positive rate. Daily fall-risk logistic regression shows coefficient weights. Presets auto-adapt nightly with an audit trail | https://www.teton.ai/de/optimised-alarms, https://www.teton.ai/blog/teton-system-overview-how-teton-protects-resident-privacy |
| **AltumView Sentinare** | Fall | Sends stick-figure animations only, never video | None; no accuracy figures | https://hospitalnews.com/monitoring-system-for-seniors-puts-privacy-first |
| **Staqu JARVIS** (India) | Fire/smoke (e.g., Adani Tiroda plant), 70 jails | Existing CCTV; needs internet | Voice, email, and video-wall alerts; no explanation | https://krishijagran.com/industry-news/adani-power-strengthens-fire-alert-system-at-tiroda-plant-with-staqu-s-ai-powered-video-analytics-jarvis/?amp=1 |
| **Maha Kumbh 2025** (UP govt) | Crowd density per m², surge prediction | About 2,700 CCTV cameras, about 160 AI-capable in the fair area, 4 command centres | None reported; network gaps and cost noted | https://www.drishtiias.com/state-pcs-current-affairs/ai-in-maha-kumbh-2025 |
| **Genetec Mission Control** (PSIM) | Any (rule-correlated) | On-prem | **Dynamic SOPs**: on-screen step-by-step procedures with a decision matrix. Rule-based; no AI explanation | https://convergint.com/wp-content/uploads/2022/03/genetec-mission-control.pdf |

### 1.2 Takeaways on commercial explainability

- **Mostly black-box detection, with LLM narration added on top.** The 2025–26 trend (Milestone, Ambient, Coram, NVIDIA VSS, Verkada summaries) is to describe clips with a VLM after the fact or to have a VLM re-check alerts. None of the sources describes any mechanism ensuring the summary reflects why the detector fired. Benschop et al. found open VLMs (≤8B) produced "wrong label" texts at rates up to 76–88% on surveillance benchmarks: https://arxiv.org/html/2510.23190
- **Accuracy claims are mostly undefined.** Scylla's 99% has no stated metric. Verkada and AltumView publish no figures.
- **Teton is the exception.** It shows per-rule catch and false-positive rates and model coefficients. That is precisely the "evidence plus calibration" style this project should adopt and extend to video and multi-hazard settings.
- **SOP-driven response exists in PSIM software** (Genetec), but it is rule-triggered and not tied to an explained AI detection.
- **Edge VLMs need GPU-class hardware.** VSS targets Jetson Orin/Thor. Intel's VLM fire demo targets Core Ultra. None of this suits a 2-core CPU.

---

## 2. Research and open source

### 2.1 Fall detection (pose and edge)

| Work | Method | Data | Result | Hardware | Source |
|---|---|---|---|---|---|
| Three-stream ST-GCN (2024) | Skeleton GCN | UR Fall, Le2i | 99.15% (UR), 99.64% (Le2i) in tables; the abstract says 99.79/99.51 (inconsistent) | GPU | https://arxiv.org/pdf/2408.12211 |
| Same paper's baseline table | Skeleton+SVM 96.55, Bi-LSTM 96.70, Wang 97.33 (UR); Wang 96.91 (Le2i) | | | | same |
| MoveNet rule-based (2024) | Keypoint geometry, 2-frame persistence | URFD | Sens 0.917, Spec 0.725, Acc 0.797; reported F1 0.62 (inconsistent with its own P/R) | Laptop i5 | https://arxiv.org/html/2401.01587v1 |
| Proietti et al. (MetroXRAINE 2025) | YOLOv8 fall/nofall boxes | Le2i + UR + FPDS | Acc 93.38%, fall recall 96.37%, F1 0.86, mAP50 ≈ 0.8; detection < 2 s | **Raspberry Pi 3** | https://iris.univpm.it/retrieve/510a125a-055e-4e02-b770-69c0931e5844/Edge%20AI-Based%20Fall%20Detection%20with%20Standard%20RGB%20Cameras.pdf |
| Marin et al. (IIT, 2026) | YOLOv8-pose, CNN-LSTM, FSM for post-fall states; optional wrist IMU | Curated | 91.5% pose-only, 94.7% with IMU; latency < 300 ms | **Raspberry Pi 4B** + ESP32 | https://ecasp.ece.iit.edu/publications/2012-present/EIT2026_147.pdf |
| Elektor Pi 5 + Hailo-8L | YOLOv8-pose plus "flat body" and box-aspect heuristics | None | No accuracy; "≥30 FPS" claimed | Pi 5 + NPU | https://www.elektormagazine.de/labs/pi-5hailo8l-ai-powered-fall-identification-recording-project |
| Mali & Jaiswal (2025) | MediaPipe + thresholds + 20-frame voting | n/a | No metrics in abstract | Standard CPU | https://arxiv.org/abs/2503.19501v1 |
| Dinh-Tu et al. (2026) | YOLOv8n-pose | Own | "> 90%" | Jetson Nano | https://ctujs.ctu.edu.vn/index.php/ctujs/article/view/1364 |
| **OmniFall (2025/26)** | Unified staged, synthetic, and in-the-wild benchmark | 8 staged sets incl. Le2i | **Le2i in-domain Fall F1 100 drops to 58.8 in the wild.** Best in-the-wild Fall F1 is 76.0. Zero-shot 8B MLLMs reach about 68 Fall F1 but only about 30 "Fallen" F1 | GPU | https://arxiv.org/pdf/2505.19889 |
| XAI fall (Messaoudi et al. 2024) | YOLOv8 + LIME + Grad-CAM | n/a | **Visual explanations only** | n/a | https://bibliotecadigital.ipb.pt/entities/publication/d61b8ac5-cb66-4349-91d5-1deac179fd56/full |
| "Trustable and explainable fall detection" (Patel et al., Image & Vision Computing 2024) | Transfer learning + XAI | n/a | Details not retrievable | n/a | https://research.science.eus/documentos/671536e450d0ab26347c669d |

**Datasets.** UR Fall has 70 sequences (30 falls, 40 ADL), Kinect RGB/depth plus accelerometer, licensed CC BY-NC-SA: https://fenix.ur.edu.pl/~mkepski/ds/uf.html. Le2i has 191 videos across 4 scenes (home, coffee room, office, lecture room) at 320×240, 25 fps: https://imvia.ube.fr/?p=346

**Implication.** Rule-based pose falls on Pi-class hardware are common. Novelty must come from elsewhere. Cross-dataset or "in-the-wild" robustness is a credible evaluation angle (OmniFall).

### 2.2 Video fire and smoke detection

- **D-Fire**: about 21.5k images (1,164 fire only, 5,867 smoke only, 4,658 both, 9,838 neither), 14,692 fire boxes and 11,865 smoke boxes, YOLO format: https://github.com/gaiasd/DFireDataset
- **YOLO-n on D-Fire (CCi-YOLOv8n paper)**: YOLOv8n 77.3 mAP50 (2.0M params per their table, 7.7 GFLOPs), YOLO11n 75.4, YOLOv10n 77.5, YOLOv5s 76.7, YOLOv7-tiny 76.3, CCi-YOLOv8n 78.5: https://arxiv.org/pdf/2411.11011
- **Conflicting baseline**: Joshi et al. report YOLOv8n at **62.5 mAP50 / 36.3 mAP50-95** and YOLOv5n at 60.9 on D-Fire. Their uncertainty-aware rescoring network lifts these to 65.1 and 64.1. They use MC-dropout variance, **not conformal or calibrated** methods, and their N=1 deployment makes the variance term degenerate: https://arxiv.org/pdf/2510.10108. **Report your own protocol explicitly.**
- **Grad-CAM on an edge fire classifier**: Mahmoudi et al. (VISAPP 2025), Jetson Xavier, compressed CNNs at 79–607 FPS, about 98% accuracy. Heatmaps were checked only qualitatively, with **no faithfulness metric and no text**: https://www.scitepress.org/publishedPapers/2025/133825/pdf/index.html
- **SHAP for wildfire CNN** (FireXnet, Fire Ecology 2023), 98.42% accuracy: https://shdl.mmu.edu.my/11797/
- **YOLO + VLM** (Kim et al., ICLR 2025 workshop): YOLOv8s alone was insufficient for parking-garage fire because of ambiguous smoke boundaries. Adding Florence-2 improved reliability, with no numbers given: https://proceedings.mlr.press/v296/kim25a.html
- **Cross-dataset generalisation is weak**: MS-FSDB benchmark: https://arxiv.org/pdf/2410.16631. FASDD (flame and smoke, Geo-spatial Information Science 2025) is another large set: https://huggingface.co/Blaize-AI/YOLOv8n_FASDD

### 2.3 Crowd anomaly, panic, and rapid exit

- **UMN is saturated.** Sabokrou et al. report chaotic invariants at 99.4 AUC, social force at 94.9, sparse reconstruction at 99.6, and their own method at 99.6 (about 25 fps on CPU): https://arxiv.org/pdf/1511.06936
- **Marsden et al. holistic features** (collectiveness, conflict, density, mean speed; 4 named features): UMN AUC 0.929 at **40 FPS on an i5 CPU**, vs SFM 0.97 at 3 FPS and MDT 0.995 at 0.9 FPS. This is interpretable and CPU-friendly: https://ar5iv.arxiv.org/html/1606.05310
- **Helbing et al. crowd pressure** P = ρ·Var(V). Turbulence began above **0.02 s⁻²**, about 10 min before the 2006 Jamarat disaster. The authors propose online video evaluation: https://arxiv.org/pdf/physics/0701203
- **VelocityNet** (SDAIA, 2025): YOLO11 heads + RAPIDFlow optical flow, per-person speed mapped to halt/slow/normal/fast with signed % deviation. Private Makkah data, A5000 GPU, no AUC: https://arxiv.org/html/2510.18187v1
- **FluidFlow** (FOSS Hack 2026): YOLOv8n + Farneback flow, turbulence and Reynolds-number thresholds, LSTM forecast. Projected impact, not measured: https://fossunited.org/hack/fosshack26/p/0s0mrhodkh
- **Domain thresholds**: flow peaks at a critical density of about 2–3 persons/m² (Keith Still): https://gkstill.com/Support/crowd-flow/DensityFlow.html. India's NDMA crowd guide lists warning signs (counter-flows, sudden door openings, fence climbing) and relies on **human** CCTV monitoring: https://sdma.goa.gov.in/sites/default/files/2022-05/managingcrowdsguide.pdf

### 2.4 Multimodal sensor and vision fusion for fire

- **ForestProtector** (2025): LoRa IoT nodes (BME280, MQ2) steer a Jetson Nano camera via DQN, then a 3D-CNN verifies smoke (AUC 0.98). Field alert time was about 2 min: https://arxiv.org/html/2501.09926v1
- **Improved Dempster–Shafer tunnel fire fusion** (Sensors 2024): temperature + CO + smoke, detection in about 2.4 s, simulation only. The paper shows how one conflicting sensor can zero out classical DS fusion: https://www.mdpi.com/1424-8220/24/19/6455
- **Edge Impulse / Arduino** image + temperature multi-input CNN on an MCU: https://www.edgeimpulse.com/blog/sensor-fusion-ignites-next-gen-fire-detection/
- **UCI MEng × Advantech (2025)**: gas, temperature, humidity, CO₂, and spectral sensors + CV on a Hailo-8 box, Isolation Forest + autoencoder, no explainability: https://sites.uci.edu/mengprojects/advantech-25
- **Gap**: fusion of *visual behaviour* (people running toward exits) with smoke evidence as corroboration was **not found**.

### 2.5 Explainable video anomaly detection with LLMs/VLMs (2023–2026)

| Work | Approach | Result | Compute | How explanations are validated | Source |
|---|---|---|---|---|---|
| Holmes-VAD (2024) | Temporal sampler + Video-LLaVA, VAD-Instruct50k | UCF-Crime AUC 89.51; XD AP 90.67 | 2×A100; 4.24 s/video | **User study** (86 clips) | https://arxiv.org/html/2406.12235v2 |
| VAD-LLaMA (2024) | VADor + Long-Term Context + Video-LLaMA | UCF AUC 88.13; TAD 91.77 | 4×L40 | **Qualitative only** | https://arxiv.org/pdf/2401.05702 |
| LAVAD (CVPR 2024) | Training-free captioner + LLM scoring | UCF AUC 80.28; XD AUC 85.36 / AP 62.01 | n/a | Captions filtered by CLIP similarity | https://arxiv.org/html/2404.01014v1 |
| AnomalyRuler (ECCV 2024) | LLM-induced normal/anomaly **rules**, then deduction | Ped2 97.9, Avenue 89.7, SHT 85.2, UBnormal 71.9 | CogVLM-17B + GPT-4 + Mistral-7B | "Doubly-Right" reasoning metric | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf |
| VADTree (2025) | Event-boundary tree + LLaVA-Video-7B + DeepSeek-R1-14B | UCF AUC 84.74; XD AUC 90.44 | GPU | LLM reasoning text | https://arxiv.org/pdf/2510.22693 |
| VERA (CVPR 2025) | Verbalised learning of guiding questions for a frozen VLM | n/a in excerpt | InternVL2-8B | n/a | https://arxiv.org/html/2412.01095v3, https://mlanthology.org/cvpr/2025/ye2025cvpr-vera/ |
| Vad-R1 (NeurIPS 2025) | Perception-to-cognition CoT + RL with "anomaly verification" reward | Acc 0.875, F1 0.862 on Vad-Reasoning | Qwen2.5-VL-7B, 4×A100 | **BLEU/ROUGE**; self-consistency, not hallucination checks | https://arxiv.org/html/2505.19877v1 |
| LATERN (May 2026) | Test-time memory + CLIP gate on summaries + evidence aggregation | n/a in excerpt | Frozen VLM | CLIP gate; LLM-as-judge + human | https://arxiv.org/html/2605.15054 |
| Khedher et al. (AAAI-SS 2025) | VLM descriptions, then rule-driven LLM verdict naming the violated rule | UCF-Crime | n/a | **No independent check of VLM descriptions** | https://ojs.aaai.org/index.php/AAAI-SS/article/view/36886 |
| Benschop et al. (2025) | Evaluates Gemma-3, Qwen2.5-VL, VideoLLaMA3, NVILA on UCF-Crime/RWF-2000 | Wrong-label rates up to 76–88%; anonymisation costs 2–11 pp | A40/L40 | Concludes "not yet reliable" for autonomous surveillance | https://arxiv.org/html/2510.23190 |

Also relevant: HAWK (NeurIPS 2024, open-world anomaly understanding): https://neurips.cc/virtual/2024/poster/93219. CUVA causation benchmark (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html. VANE-Bench: https://arxiv.org/html/2406.10326v2

**Pattern.** All of these need GPUs and multi-billion-parameter models. Explanations are judged by user studies, BLEU, or LLM judges. **None of them guarantees the text matches the detector's evidence.**

### 2.6 Interpretable-by-design video anomaly detection (closest prior art to "faithful by construction")

- **Attribute-based VAD (Reiss & Hoshen)**: per-object velocity (flow histogram), pose (AlphaPose), and CLIP features. Each gets its own density estimator. The **frame score is the sum of the three normalised scores, so each attribute's contribution is visible**. Micro AUROC: Ped2 99.1, Avenue 93.7, SHT 85.9. Code is available, and an anomalib implementation exists: https://arxiv.org/html/2212.00789v2, https://github.com/talreiss/accurate-interpretable-vad, https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/video/ai_vad.html
- **EVAL (CVPR 2023, MERL)**: high-level appearance and motion attributes plus per-region exemplars. Explains by comparing to the nearest normal exemplar. **20–100 fps**: https://merl.com/publications/docs/TR2023-071.pdf
- **SymbolicDet (2025)**: LLM-guided evolutionary search for human-readable **logic rules over detector outputs**. Gains up to +23 AUROC on event tasks. Rule evaluation is cheap, but the search uses LLM calls: https://arxiv.org/html/2502.05843v4
- **X-MAN (CVPRW 2021)**: explains multiple anomaly sources: https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Szymanowicz_X-MAN_Explaining_Multiple_Sources_of_Anomalies_in_Video_CVPRW_2021_paper.html
- Position paper: Rudin argues for interpretable models over post-hoc explanation in high-stakes settings: https://arxiv.org/abs/1811.10154v3

**Implication.** Additive, attribute-level scoring is established for single-domain VAD. Extending it to **multi-hazard emergency alerts with exact attributions that drive the text, counterfactuals, and calibration** is the open space.

### 2.7 XAI for object detectors and faithfulness evaluation

- **D-RISE** (CVPR 2021), black-box masking: https://openaccess.thecvf.com/content/CVPR2021/papers/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.pdf
- **ODAM** (ICLR 2023): instance-specific gradient maps at about 2 ms each, vs D-RISE at about 3 min/image with 5k masks. Better insertion score (50.3 vs 43.4) and pointing game: https://arxiv.org/pdf/2304.06354
- **ODExAI**: evaluates detector XAI on localisation, faithfulness (deletion/insertion), and complexity. No single method wins (G-CAME localises best and fastest; D-CLOSE is most faithful but about 71 s/image): https://www.cs.unb.ca/~hcao3/publications/2025-KI/ODExAI.pdf
- **EigenCAM on YOLOv8**: plug-and-play and gradient-free, but qualitative and layer-dependent: https://www.datature.io/blog/understanding-your-yolov8-model-with-eigen-cam. Library: https://pypi.org/project/grad-cam
- **Sanity checks**: Guided Backprop and Guided Grad-CAM fail weight-randomisation tests. Plain gradients and Grad-CAM pass: https://arxiv.org/pdf/1810.03292
- **Faithfulness of natural-language explanations**: Atanasova et al. (ACL 2023) counterfactual tests: https://preview.aclanthology.org/setup/2023.acl-short.25. Parcalabescu & Frank on self-consistency vs faithfulness: https://arxiv.org/pdf/2311.07466
- **Data-to-text hallucination checking via bidirectional NLI** (Dušek & Kasner, INLG 2020): https://aclanthology.org/2020.inlg-1.19

### 2.8 Conformal prediction and calibrated alerting

- **Conformal p-values for outlier detection** (Bates, Candès et al., Annals of Statistics 2023). Marginal validity; BH controls FDR; calibration-conditional p-values give high-probability false-positive bounds: https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-1/Testing-for-outliers-with-conformal-p-values/10.1214/22-AOS2244.full, https://arxiv.org/pdf/2104.08279.pdf
- **Conformal anomaly detection for surveillance trajectories** (Laxhammar thesis, Skövde). Sets the threshold from a desired alarm rate (CAD/ICAD): https://www.avhandlingar.se/avhandling/a5565e371d/
- **nonconform** Python library: conformal p-values, FDR control, conformal martingales for streams. Works with scikit-learn/PyOD on CPU: https://github.com/OliverHennhoefer/nonconform
- **Conformal object detection**: Andéol et al. 2023 (railway signals, YOLOv5m, box-wise CP and conformal risk control): https://arxiv.org/pdf/2304.06052. Zouzou et al. 2025 (runway, conformal mAP): https://arxiv.org/html/2505.16740v1
- **Sequential VAD with a false-alarm bound**: Doshi & Yilmaz, kNN + CUSUM with asymptotic FAR ≤ e^(−ω₀h). UR Fall few-shot AUC 86.1 (1-shot) and 91.3 (10-shot): https://arxiv.org/pdf/2004.02072, https://arxiv.org/pdf/2103.11299
- A CVPR 2026 weakly-supervised VAD paper lists a "desired FPR α" symbol but never uses it. Thresholds are grid-searched: https://openaccess.thecvf.com/content/CVPR2026/supplemental/Acharya_The_Road_Less_CVPR_2026_supplemental.pdf
- **Gap**: no fall, fire, or crowd edge system found that reports **target vs achieved false alarms per hour** under a conformal guarantee.

### 2.9 Privacy-preserving (skeleton-only) surveillance

- **Survey**: P2VAD taxonomy (non-identifiable elements, skeleton/optical-flow intermediates, edge-cloud/federated): https://arxiv.org/pdf/2411.14565
- **Skeleton VAD survey**: https://arxiv.org/pdf/2301.00114. MSG-Flow (2025/26): skeleton-only, 10.3M params, HR-ShanghaiTech AUC 88.1: https://arxiv.org/pdf/2603.26745
- **Products**: Sentinare stick figures; Teton depth silhouettes with no raw video leaving the room (section 1.1).
- **Caveat**: skeleton-only cannot see smoke. Fire requires pixel-level analysis, so "pixels never leave the device; only skeletons and evidence vectors are transmitted" is the realistic design.

### 2.10 Human-in-the-loop alert triage

- **ALFred** (BMVC 2025): active learning + human-confirmed pseudo-labels set a **per-environment adaptive threshold** for VAD: https://bmvc2025.bmva.org/proceedings/1127/
- **Commercial**: Scylla "self-learning from feedback"; Teton nightly auto-presets with audit trail; Coram patent claims human verification and retraining from labelled clips (section 1.1).
- **SOC analyst study** (Rastogi et al., ACM CCS 2025; N=248 survey + 24 interviews): triage operators ignore long explanations. They want concise, evidence-backed rationale with progressive disclosure, provenance logs, and helpful/not-helpful feedback: https://arxiv.org/html/2503.02065v2. This is directly transferable to the UI design.

### 2.11 Counterfactual explanations for alerts

- **Time-series anomaly detection** (bearings, CoMTE what-if): https://arxiv.org/pdf/2408.11935
- **AR-Pro** (NeurIPS 2024): diffusion-based anomaly "repair" counterfactuals for images and time series: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d3591b6746204b332acb464b775d38d-Abstract.html
- **Smart-environment rules** ("the TV would be on if it was not after 11 pm"), Master's thesis, 17-person study. Counterfactuals were preferred in later scenes: https://kups.ub.uni-koeln.de/78813/1/Counterfactual_Explanations_in_Smart_Environments.pdf
- **Video classifier counterfactuals** (CVPR 2026, title only seen): https://cvpr.thecvf.com/virtual/2026/poster/39921
- **Gap**: not found for emergency video or sensor alerts on edge hardware.

### 2.12 Compute cascades and energy

- **Frigate NVR**: motion detection is a "first line check" before object detection: https://docs.frigate.video/configuration/motion_detection/
- **FrameHopper** (DCOSS 2022): RL frame skipping filters about 91% of frames with a 4× speedup on Jetson Nano (vs Reducto at 3×): https://arxiv.org/pdf/2203.11493
- **Survey of edge video analytics**: https://arxiv.org/pdf/2211.15751
- **Implication**: a motion gate is engineering hygiene, not novelty. Its value is in **measuring** the energy and latency savings on your own hardware.

### 2.13 SOP and recommendation layer

- **PSIM dynamic SOPs** (Genetec): see section 1.1.
- **US patent family on AI building-emergency egress guidance** (continuations of US 11,043,095 and 11,501,621): predicts fire spread and generates egress routes for occupants: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12205447
- **Digital-twin evacuation tracking** (Sun et al., EAAI 2026): YOLOv8-Pose + DeepSORT, cloud processing, **no LLM or recommendations**: https://ira.lib.polyu.edu.hk/bitstream/10397/118154/1/1-s2.0-S0952197626007219-main.pdf
- **Implication**: recommending the "nearest exit/AED" is not novel by itself. **Recommendations that cite the specific evidence that triggered them, drawn from a campus playbook, on a CPU edge node** are uncommon in the research literature.

---

## 3. Gap analysis

Novelty: 1 = commodity, 5 = no prior art found. Feasibility is for a 2-core CPU prototype in one night: **H** = a few hours with standard libraries; **M** = doable but tight or needs data prep; **L** = not realistic.

| # | Idea | Already exists? | Evidence / citation | Novelty | Feasibility |
|---|---|---|---|---|---|
| 1 | Plain-language alert justification (generic) | **Yes**, commercial and research | Verkada summaries, Milestone GenAI, Ambient Pulsar, NVIDIA VSS; Holmes-VAD, VAD-LLaMA (sections 1.1, 2.5) | 1 | H (templates) / L (LLM on CPU) |
| 2 | Grad-CAM / EigenCAM saliency on the YOLO detector | **Yes** | Mahmoudi VISAPP 2025; Messaoudi 2024; Datature EigenCAM (sections 2.2, 2.1, 2.7) | 1 | H (EigenCAM); L (D-RISE) |
| 3 | **Faithful-by-construction additive evidence fusion with exact per-evidence attributions** driving the text | **Partially.** Additive attribute scoring exists for single-domain VAD; Teton shows LR coefficients. Not found for multi-hazard emergency alerts where the text is a rendering of the attribution ledger | Reiss & Hoshen https://arxiv.org/html/2212.00789v2; Marsden https://ar5iv.arxiv.org/html/1606.05310; Teton https://www.teton.ai/de/optimised-alarms; EBM/GAM tooling https://interpret.ml/docs/ebm | 3 | H |
| 4 | **Counterfactual alert explanations** ("would not fire if smoke-growth < 0.12/s or exit-flow < 1.5×") | **Not for emergency video or sensor alerts.** Exists for time series, smart-home rules, image anomaly repair | https://arxiv.org/pdf/2408.11935; https://kups.ub.uni-koeln.de/78813/1/Counterfactual_Explanations_in_Smart_Environments.pdf; AR-Pro https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d3591b6746204b332acb464b775d38d-Abstract.html | 4 | H (closed form when the fusion is additive or thresholded) |
| 5 | **Automatic verification that natural-language explanations are grounded in computed evidence** (claim-level check; reject or repair) | **Pieces exist.** NLI-based data-to-text checking; VAD explanations judged by user study, BLEU, or LLM judges; LATERN's CLIP gate filters summaries; Vad-R1 uses self-consistency rewards. No claim-by-claim verifier against an alert evidence ledger found | Dušek & Kasner https://aclanthology.org/2020.inlg-1.19; Holmes-VAD; Vad-R1 https://arxiv.org/html/2505.19877v1; LATERN https://arxiv.org/html/2605.15054; wrong-label rates https://arxiv.org/html/2510.23190 | 4 | H (slot/number/entity matching; optional small NLI model) |
| 6 | **Conformal / calibrated alert thresholds with false-alarm guarantees** | **Method exists**, including for surveillance trajectories and object detection. Not found in a fall/fire/crowd edge system reporting target vs empirical FAR | Bates et al. https://arxiv.org/pdf/2104.08279.pdf; Laxhammar https://www.avhandlingar.se/avhandling/a5565e371d/; Andéol https://arxiv.org/pdf/2304.06052; Doshi & Yilmaz https://arxiv.org/pdf/2004.02072; nonconform https://github.com/OliverHennhoefer/nonconform | 3 | H |
| 7 | Operator feedback adapting per-zone thresholds | **Yes** (research and commercial) | ALFred https://bmvc2025.bmva.org/proceedings/1127/; Scylla; Teton; Coram patent | 2 (3 if feedback updates a per-zone conformal calibration set so the guarantee is kept) | H |
| 8 | Privacy-by-design skeleton-only pipeline | **Yes** | Sentinare; Teton; P2VAD survey https://arxiv.org/pdf/2411.14565; MSG-Flow | 1–2 | H for fall/crowd; not applicable to smoke |
| 9 | Compute cascade (motion gate) for energy | **Yes** | Frigate; FrameHopper https://arxiv.org/pdf/2203.11493 | 1 | H |
| 10 | Campus SOP playbook recommendations (nearest exit/AED) | **Partially.** PSIM dynamic SOPs; egress-guidance patents. Not found coupled to explained edge detections with evidence-cited recommendations | Genetec https://convergint.com/wp-content/uploads/2022/03/genetec-mission-control.pdf; patent https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12205447; NDMA guide | 2–3 | H |
| 11 | Multi-hazard (fall + fire + crowd) in one edge system | **Commercially yes** (VMS suites, mostly GPU or cloud). Research is mostly single-hazard; student and MEng projects exist | Scylla suite; Staqu; Gorilla; UCI/Advantech https://sites.uci.edu/mengprojects/advantech-25 | 2 | M |
| 12 | **Cross-hazard corroboration** (smoke growth + rapid exit flow raise each other's confidence; contradiction triggers deferral) | **Not found** for vision-behaviour + smoke. Multi-*sensor* fire fusion exists | DS fusion https://www.mdpi.com/1424-8220/24/19/6455; ForestProtector https://arxiv.org/html/2501.09926v1 | 3–4 | M–H |
| 13 | **Uncertainty-aware explanation text** ("p = 0.004: normal minutes in this zone look like this less than 1 in 250 times; expected false alarms ≈ 0.2/day at this setting") | **Not found** | Teton shows rule-level FP rates, but not per-alert calibrated evidence in text | 3–4 | H |
| 14 | Quantitative faithfulness evaluation (deletion/insertion) of saliency on the edge fire detector | Methods exist; fire- or edge-specific reports rare | ODAM, ODExAI (section 2.7) | 2 | M (EigenCAM + deletion curve on about 200 images is feasible; D-RISE is not) |
| 15 | Crowd-physics features (Helbing pressure, flow vs density) as explainable crowd evidence | **Yes** in physics and hackathons; rarely in explainable alert text | Helbing https://arxiv.org/pdf/physics/0701203; FluidFlow; VelocityNet | 2 | H |
| 16 | LLM/VLM explanation generation **on the CPU edge device** | Exists on GPU/NPU edge (VSS, Intel OpenVINO VLM) | Pi 5 runs 1–1.5B Q4 LLMs at about 10–11.5 tok/s https://stratospherelinuxips.readthedocs.io/en/develop/immune/research_rpi_llm_performance.html | 1 | L (seconds per alert; unverifiable) |
| 17 | Principled **abstain/defer-to-human** when evidence conflicts or calibration is uncertain | Human verification is common (Coram, Volt.ai). Principled conflict- or p-value-based deferral not found here | Coram patent claim 10; SOC study https://arxiv.org/html/2503.02065v2 | 3 | H |
| 18 | Operator usability study of explanation formats (time-to-decision, trust calibration) for physical emergencies | Done for cyber SOCs, not found for physical emergency alerts | Rastogi et al. CCS 2025 https://arxiv.org/html/2503.02065v2 | 3 | L in one night (a 5–8 classmate micro-study is possible later) |

---

## 4. Benchmarks to compare against

### 4.1 Fall detection

| Dataset | Reported number | Notes | Source |
|---|---|---|---|
| UR Fall | 96.5–97.3% accuracy (skeleton+SVM, Bi-LSTM, Wang 2020); **99.15%** (3-stream GCN) | Saturated; small (70 sequences) | https://arxiv.org/pdf/2408.12211 |
| Le2i | 96.9% (Wang 2020); **99.64%** (3-stream GCN) | Saturated | same |
| URFD, rule-based MoveNet on laptop CPU | Sens 0.917, Spec 0.725, Acc 0.797 | Realistic **rule-baseline floor**; internal inconsistencies | https://arxiv.org/html/2401.01587v1 |
| Le2i + UR + FPDS, YOLOv8 on **RPi 3** | Acc 93.38%, F1 0.86, fall recall 96.4% | Closest edge comparator | Proietti et al. 2025 (section 2.1) |
| Curated data, YOLOv8-pose + CNN-LSTM on **RPi 4** | 91.5% (pose only); 94.7% (+IMU); < 300 ms latency | Edge comparator | https://ecasp.ece.iit.edu/publications/2012-present/EIT2026_147.pdf |
| UR Fall, few-shot sequential detector | AUC 86.1 (1-shot) / 91.3 (10-shot) | Anomaly framing | https://arxiv.org/pdf/2103.11299 |
| Cross-domain (OmniFall) | Le2i Fall F1 **100 → 58.8** in the wild; best in-the-wild Fall F1 76.0 | **Report cross-dataset results** (train UR, test Le2i, or vice versa) | https://arxiv.org/pdf/2505.19889 |

### 4.2 Fire and smoke

| Dataset | Model | mAP@0.5 | Source |
|---|---|---|---|
| D-Fire | YOLOv8n | **77.3** (another paper: 62.5; mAP50-95 36.3) | https://arxiv.org/pdf/2411.11011; https://arxiv.org/pdf/2510.10108 |
| D-Fire | YOLO11n | 75.4 | https://arxiv.org/pdf/2411.11011 |
| D-Fire | YOLOv5s / YOLOv7-tiny / YOLOv10n | 76.7 / 76.3 / 77.5 | same |
| D-Fire | CCi-YOLOv8n (current small-model SOTA claim) | 78.5 | same |
| D-Fire | YOLOv5n | 60.9 (→ 64.1 with rescoring) | https://arxiv.org/pdf/2510.10108 |
| Forest fire classification on Jetson Xavier | ResNet18 compressed | 96.9% accuracy at 606 FPS | https://www.scitepress.org/publishedPapers/2025/133825/pdf/index.html |

Advice: state your split (official D-Fire train/test), input size (640), and confidence/IoU thresholds. The 62.5 vs 77.3 spread shows protocol matters more than architecture tweaks.

### 4.3 Crowd anomaly

| Dataset | Method | AUC | Speed | Source |
|---|---|---|---|---|
| UMN | Social force (Mehran 2009) | 0.949–0.97 | 3 FPS | https://arxiv.org/pdf/1511.06936; https://ar5iv.arxiv.org/html/1606.05310 |
| UMN | Chaotic invariants / sparse reconstruction | 0.994 / 0.996 | n/a | https://arxiv.org/pdf/1511.06936 |
| UMN | MDT | 0.995 | 0.9 FPS | https://ar5iv.arxiv.org/html/1606.05310 |
| UMN | Sabokrou et al. | 0.996 | about 25 FPS (CPU, MATLAB) | https://arxiv.org/pdf/1511.06936 |
| UMN | **Marsden holistic (4 interpretable features)** | 0.929 (single-scene), 0.869 (cross-scene) | **40 FPS on i5** | https://ar5iv.arxiv.org/html/1606.05310 |
| Avenue / SHT (general VAD) | Attribute-based (interpretable) | 93.7 / 85.9 (micro) | n/a | https://arxiv.org/html/2212.00789v2 |
| Avenue / SHT | EVAL (interpretable) | 86.0 / 76.6 (frame) | 20–100 fps | https://merl.com/publications/docs/TR2023-071.pdf |
| UCF-Crime (LLM-VAD reference) | Holmes-VAD / VAD-LLaMA / VADTree / LAVAD | 89.5 / 88.1 / 84.7 / 80.3 | GPU only | section 2.5 |

Advice: UMN is saturated near 0.99, so report it, but compete on **FPS on a 2-core CPU at matched AUC** and on **cross-scene AUC** (Marsden's 0.869 is a fair interpretable baseline).

### 4.4 Edge latency and throughput

| Platform | Model / runtime | Latency / FPS | Source |
|---|---|---|---|
| Raspberry Pi 5 | YOLO26n, PyTorch / ONNX / OpenVINO / **NCNN** (640, FP32, inference only) | 299 / 126 / 105 / **67 ms** | https://docs.ultralytics.com/guides/raspberry-pi/ |
| Raspberry Pi 5 | YOLO11n ONNX | 6.79 FPS (YOLO26n: 7.79 FPS) | same |
| Raspberry Pi 4 / 5 | YOLOv8n, ncnn (640) | **3.1 FPS / 20 FPS** | https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4 |
| Reference CPU (Ultralytics) | YOLOv8n ONNX | 80.4 ms | https://docs.ultralytics.com/models/yolov8/ |
| Reference CPU (Ultralytics) | YOLO26n-pose ONNX | 40.3 ms (mAP-pose50 83.3) | https://docs.ultralytics.com/tasks/pose/ |
| Jetson Orin Nano 8GB | YOLOv8n PyTorch FP32 / TRT FP16 / TRT INT8 | 20.9 ms / **4.43 ms (225 FPS)** / 3.49 ms | https://github.com/hokwangchoi/jetson-orin-nano-benchmarks |
| Jetson Nano | TAO ActionRecognitionNet 2D-RGB | 30 FPS | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet |
| Jetson Nano | Frame-skipping cascade | 16.8 → 63.5 FPS (4×) | https://arxiv.org/pdf/2203.11493 |
| Raspberry Pi 5 power | idle / 4-core load | about 3.0 W / about 8.8 W (up to 16 W with peripherals) | https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-power-consumption-update-2026-all-models-compared |
| Raspberry Pi 5 LLM | 1–1.5B models Q4_K_M (Ollama) | about 10–11.5 tok/s, so a 40-token explanation takes about 4 s | https://stratospherelinuxips.readthedocs.io/en/develop/immune/research_rpi_llm_performance.html |

### 4.5 Cloud round-trip and cloud-VLM latency (the "why edge" argument)

| Item | Number | Source |
|---|---|---|
| Network RTT to cloud regions | about 15 ms (nearby) to 80+ ms (cross-ocean); edge best case about 1 ms | https://www.arxiv.org/pdf/2104.14050 |
| Median min-RTT, Asia | 16.5 ms to cloud regions vs 8.1 ms to cloud-edge sites; 80th-percentile tails much worse | https://arxiv.org/pdf/2304.05481 |
| Edge "performance inversion" | At high utilisation, edge queuing can exceed cloud latency (inversion at about 40–85% utilisation) | https://www.arxiv.org/pdf/2104.14050 |
| **Cloud VLM per image** | Gemini 1.5 Flash-8B 0.86 s, Gemini 2 Flash 0.94 s, GPT-4o-mini 3.18 s, GPT-4o 4.57 s; local InternVL2-8B on A6000 5.3 s | https://arxiv.org/pdf/2504.09083 |
| Explainable-VAD MLLM | Holmes-VAD 4.24 s per video on 2×A100 | https://arxiv.org/html/2406.12235v2 |
| Field multimodal fire alert | about 118–130 s end-to-end (ForestProtector) | https://arxiv.org/html/2501.09926v1 |

Framing: network RTT alone is tens of ms, but **VLM-based explanation costs about 1–5 s per alert** and needs connectivity, which the Maha Kumbh deployment flagged as a weak point. A template-plus-verifier explanation from an evidence ledger costs **milliseconds** on CPU. That is a measurable latency claim the team can own.

---

## 5. Recommended add-ons, ranked

Ranking weighs novelty (gap table), defensibility (claims you can measure), and CPU feasibility.

### Rank 1: Evidence ledger with additive or rule fusion; explanations faithful by construction
- **What**: each detector emits named, unit-bearing evidence terms. Examples: `smoke_area_growth (%/s)`, `smoke_conf`, `exit_flow_ratio`, `mean_speed_z`, `torso_angle_drop (deg/s)`, `ground_time (s)`, `crowd_density (p/m²)`, `helbing_pressure (s⁻²)`. A per-hazard score is a **sum of calibrated contributions** (logistic/GAM/EBM, or log-likelihood ratios). The text template lists the top-k contributions with their actual values. Attributions are exact because the model is additive.
- **Why novel**: additive scoring exists for VAD (Reiss & Hoshen; Marsden), but not as the source of multi-hazard emergency text (gap rows 3 and 1).
- **Measure**: attribution exactness (sum of contributions minus score = 0); deletion test (removing a cited term changes the score by exactly its contribution); accuracy vs a black-box classifier on the same features.

### Rank 2: Explanation grounding verifier (zero-hallucination guarantee plus a measured LLM baseline)
- **What**: parse every generated sentence into claims (entity, quantity, direction, zone, time) and check each against the ledger. Reject or repair unsupported claims. Optionally run a small NLI model over ledger-to-text pairs, following Dušek & Kasner.
- **Experiment**: run the same alerts through (a) templates and (b) an LLM or VLM narrator (cloud or Pi 5 llama.cpp). Report **unsupported-claim rate**: expected 0% for (a) by construction, and some measured rate for (b), which can be compared with Benschop et al.'s 76–88% wrong-label rates.
- **Why novel**: VAD explanation work evaluates with user studies, BLEU, or LLM judges, not claim-level grounding (gap row 5).

### Rank 3: Exact counterfactuals for every alert
- **What**: with an additive or thresholded model, compute the minimal change per evidence term that drops the score below threshold, holding the others fixed. Example: "would not have fired if smoke growth were below 0.9 %/s (observed 2.3) **or** exit flow were below 1.4× baseline (observed 2.6×)." Also give the reverse for near-misses: "would have fired if…"
- **Why novel**: not found for emergency video or sensor alerts (gap row 4). Cheap and exact on CPU.
- **Measure**: validity (re-scoring the counterfactual flips the decision: 100%), sparsity (terms changed), and plausibility (value within the observed normal range).

### Rank 4: Conformal per-hazard, per-zone thresholds with a stated false-alarm budget
- **What**: calibrate each hazard score on "normal" footage per zone and alert when the conformal p-value < α. Tie α to a budget such as "≤ 1 false alarm per camera per day" by converting windows per day to α. Use `nonconform` or about 20 lines of NumPy. Put the p-value and expected false alarms per day into the explanation (gap row 13).
- **Measure**: target vs empirical FAR on held-out normal video (UR ADL sequences, Le2i non-fall clips, D-Fire "neither" images, UMN normal segments), plus recall at that FAR. Compare against a fixed 0.5 threshold.
- **Novelty**: the method is known (Bates et al.; Laxhammar), but its use in this application with reported FAR vs α is not.

### Rank 5: Cross-hazard corroboration and conflict-aware deferral
- **What**: a fusion layer where smoke evidence and crowd rapid-exit evidence **corroborate** each other (for example, a summed log-likelihood or a Dempster–Shafer combination with a conflict measure). High conflict, such as strong smoke with calm, stationary people, or a sensor-vs-camera disagreement, routes to "verify" (human) rather than "alarm" (gap rows 12 and 17).
- **Measure**: on scripted or composited clips, false-alarm reduction from corroboration at matched recall, and the share of deferred alerts that were truly ambiguous.

### Rank 6: Feedback-driven per-zone recalibration that preserves the guarantee
- **What**: operator marks an alert "false" or "true" and the confirmed-normal windows join that zone's conformal calibration set, with a rolling window to track drift. This differs from ALFred and Scylla in keeping the FAR statement valid under the exchangeability caveat, and it logs provenance.
- **Novelty**: 2–3. It is cheap to build and a strong demo.

### Rank 7: Evidence-cited SOP playbook recommendations
- **What**: a YAML campus playbook (zones, exits, AED and extinguisher locations, warden phone tree), with recommendations chosen by rules over ledger terms. Each recommendation cites its trigger: "Direct occupants to Exit B (not A): smoke evidence concentrated near A." Align with NDMA crowd guidance (counter-flow avoidance, never a single exit).
- **Novelty**: 2–3. PSIM SOPs exist, but evidence-cited recommendations on an explained edge detector are rare.

### Rank 8: Motion-gated cascade with measured energy (engineering, not novelty)
- Report FPS and estimated joules per hour with and without the gate on the 2-core CPU, using Pi 5 numbers as a reference (about 3 W idle, about 8.8 W load). Cite Frigate and FrameHopper as prior art.

### Rank 9: Privacy mode (design property, not novelty)
- Fall and crowd pipelines use only keypoints and tracks. Fire uses pixels on-device. Only the evidence ledger, skeletons, and optional blurred snapshots leave the device. Cite Sentinare, Teton, and the P2VAD survey as prior art.

### What NOT to claim as novel
- "Plain-language explanations", Grad-CAM/EigenCAM heatmaps, skeleton privacy, motion gating, operator-feedback thresholds, and running YOLO on a Raspberry Pi all have clear prior art (section 3).
- Avoid saying "first". Use "to our knowledge, not previously combined for multi-hazard emergency alerting on CPU edge devices", and cite the closest prior art for each component.

### Suggested headline evaluation
1. Detection: D-Fire mAP@0.5 (state the protocol), UR→Le2i cross-dataset fall F1, UMN AUC plus cross-scene AUC.
2. Calibration: target α vs empirical false alarms per hour per hazard.
3. Explanation: unsupported-claim rate (template+verifier vs LLM narrator), counterfactual validity and sparsity, attribution exactness, and saliency deletion AUC (EigenCAM) for the fire detector.
4. Systems: CPU FPS, p95 alert latency, energy per hour with and without the gate, and explanation latency (ms for templates vs seconds for an LLM, per section 4.5).

---

## 6. Caveats on sources
- The Hikvision/iThermAI, diva-portal, Nature Sci Rep, and some PMC pages could not be fetched. Statements about them are limited to titles.
- Several papers have internal inconsistencies (e.g., 3-stream GCN abstract vs tables; MoveNet F1; CCi-YOLOv8n parameter claims). These are flagged inline.
- Vendor figures (Ambient 95%, Milestone 30%, Scylla 99%) are unverified marketing claims.
- Benchmark numbers come from different protocols and hardware. Compare like-for-like only after rerunning under your own setup.

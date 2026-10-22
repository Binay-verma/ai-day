# Methods Reference: Explainable Edge Emergency Detection (falls, smoke/fire, crowd panic)

Target platform: 2-core CPU, ONNX Runtime. Every section lists recipes you can implement directly, then sources.
Tags used below: **[src]** means the value comes from the cited paper or code. **[derived]** means it is our own engineering suggestion built from the cited material, to be tuned on our data.
Local micro-benchmarks were run on this sandbox (2 cores, NumPy 2.5, OpenCV 5.0, 2 threads). They are labelled **[bench]**.

---

## 1. Saliency for YOLO-style detectors at the edge

### 1.1 Method comparison

| Method | Needs gradients | Class/instance specific | Cost per image | Edge verdict |
|---|---|---|---|---|
| Grad-CAM | yes | class-specific, not instance-specific | 1 fwd + 1 bwd | Not available in plain ORT inference; poor for detectors (COCO pointing game 26.7%) [src: ODAM] |
| EigenCAM | no | neither (first principal component of a layer) | 1 SVD/eigendecomposition on one feature map | **Best fit for live use**. Make it box-local by renormalising inside each box |
| D-RISE | no (black box) | instance-specific | N forward passes (N=5000 in paper): ~70 s/image for YOLOv3 on a V100 [src] | Offline evaluation only |
| ODAM | yes | instance-specific | ~2 ms per prediction on GPU [src] | Offline (PyTorch) reference for the paper, not for ORT |

ODAM vs D-RISE vs Grad-CAM, pointing game on MS-COCO boxes: 91.9 / 82.6 / 26.7. Energy-based pointing game: ODAM 73.1 vs D-RISE 17.4. On CrowdHuman, D-RISE gets 1.5 and ODAM 95.5. [src: ODAM, arXiv 2304.06354]

### 1.2 EigenCAM: the maths

- Feature map A ∈ R^{C×H×W}. Reshape it to X ∈ R^{(HW)×C}, one row per spatial location.
- Centre each channel (column): X ← X − mean_rows(X). The library's comment notes the map comes out negative without centring.
- SVD: X = U Σ Vᵀ. The saliency is the projection onto the first right singular vector: s = X·v₁, reshaped to H×W. The paper writes this as L = O·V₁ [src: Eigen-CAM, arXiv 2008.00299].
- Sign ambiguity: v₁ and −v₁ are equally valid. Apply the Seg-Eigen-CAM correction: if |min(s)| > |max(s)|, set s ← −s. The library ships this as `get_2d_projection_with_sign_correction`. This matters because the library then applies **ReLU** (`np.maximum(cam,0)`), which zeroes the salient region if the sign came out flipped.
- Post-processing in `BaseCAM`: ReLU, then min-max scale to [0,1], then `cv2.resize` to the input size. With several layers: ReLU, mean across layers, rescale.
- Eigen-CAM paper reports lower top-1 localisation error than Grad-CAM on 5 ImageNet CNNs (e.g., VGG-16: 47.67 vs 56.51). No timing is reported.

### 1.3 How pytorch-grad-cam does YOLOv5 EigenCAM (tutorial "EigenCAM for YOLO5")

- `model = torch.hub.load('ultralytics/yolov5','yolov5s')`, input resized to 640×640, `target_layers = [model.model.model.model[-2]]`. This is the last neck block before `Detect`.
- No reshape transform (the layer already returns a 4-D tensor). No target function, because EigenCAM is class-agnostic.
- `cam = EigenCAM(model, target_layers); gray = cam(tensor)[0]`.
- **Box renormalisation** (`renormalize_cam_in_bounding_boxes`): start from a zero map; for each detected box, copy the CAM crop and min-max scale it to [0,1]; finally rescale the whole map. This gives a per-box explanation and suppresses background activations.
- YOLOv8 equivalent (from the ultralytics `yolov8.yaml` layer order): layer 15 = P3/8 output (80×80 at 640), 18 = P4/16 (40×40), 21 = P5/32 (20×20), 22 = Detect. `model.model.model[-2]` is therefore layer 21 (P5). For small smoke or fire, P3 (layer 15) gives finer maps. **[derived]**

### 1.4 Exporting the feature map as an extra ONNX output

**[derived]** Two ways to do it:
1. **At export time**: wrap the PyTorch model. Register a forward hook on the target layer and return `(preds, feat)`, then export with `output_names=['preds','feat']`.
2. **Post-hoc graph edit** (works on the Ultralytics ONNX):
```python
import onnx
m = onnx.load("yolov8n.onnx")
m = onnx.shape_inference.infer_shapes(m)
name = "/model.21/cv2/act/Mul_output_0"      # C2f output (SiLU = Sigmoid*Mul); verify in Netron
vi = [v for v in m.graph.value_info if v.name == name][0]
m.graph.output.append(vi)
onnx.save(m, "yolov8n_feat.onnx")
```
Add the output **before** INT8 quantisation, so the quantiser keeps a float (dequantised) graph output.

**Fast EigenCAM in NumPy.** Use the Gram/covariance trick instead of a full SVD. It gives the same map (correlation 1.0 with SVD in the benchmark):
```python
def eigencam(A, iters=0):                    # A: (C,H,W) float32
    C,H,W = A.shape
    X = A.reshape(C,-1).T; X = X - X.mean(0)
    S = X.T @ X                              # C×C
    if iters:                                # power iteration (cheapest for C>=128)
        v = np.ones(C, np.float32)/np.sqrt(C)
        for _ in range(iters): v = S@v; v /= np.linalg.norm(v)+1e-12
    else:
        v = np.linalg.eigh(S)[1][:,-1]
    s = (X @ v).reshape(H,W)
    if abs(s.min()) > abs(s.max()): s = -s   # sign correction
    s = np.maximum(s,0); return s/(s.max()+1e-7)
```
**Cost [bench, 2 threads]:**

| Feature map | Library SVD (`full_matrices=True`) | Economy SVD | eigh(XᵀX) | 15 power iterations |
|---|---|---|---|---|
| 64×40×40 | 129 ms | 9.6 ms | 0.7 ms | 0.5 ms |
| 128×20×20 | 36 ms | 16.5 ms | 2.0 ms | 0.5 ms |
| 256×20×20 | 49 ms | 26.7 ms | 13.7 ms | 1.5 ms |
| 64×80×80 | – | 46.7 ms | 2.0 ms | 3.2 ms |

The library's `full_matrices=True` builds an HW×HW U matrix it never uses. Avoid it on the edge. **Expected overhead with power iteration: ~1–3 ms per frame**, plus a ~0.4 MB tensor copy.

**Box-local, more instance-specific variant [derived]**: crop the feature map to the box, mapped to feature coordinates (stride 8/16/32) with a 1-cell margin, then run EigenCAM on that crop only. This is cheaper still and gives a principal pattern specific to the box. Optionally, on an alert only, run a coarse occlusion test (4×4 grid = 16 forward passes at 320 px) as an instance-specific check.

Sources: https://pypi.org/project/grad-cam ; https://github.com/jacobgil/pytorch-grad-cam (`pytorch_grad_cam/utils/svd_on_activations.py`, `base_cam.py`, `tutorials/EigenCAM for YOLO5.ipynb`) ; https://arxiv.org/pdf/2008.00299 ; https://arxiv.org/pdf/2304.06354 ; https://arxiv.org/pdf/2006.03204v2 ; https://www.datature.io/blog/understanding-your-yolov8-model-with-eigen-cam

---

## 2. Evaluating saliency faithfulness

### 2.1 Deletion / insertion (RISE, Petsiuk et al. 2018)
- **Ordering**: sort pixels by decreasing saliency.
- **Deletion**: at step i, replace the next `step` pixels with a substrate. The reference code uses `torch.zeros_like` on the normalised tensor, i.e., the dataset mean colour. Record h_i = score of the originally predicted class. **Lower AUC is better.**
- **Insertion**: start from a blurred image (reference code: Gaussian kernel `klen=11`, `sigma=5`) and copy original pixels back in decreasing-saliency order. **Higher AUC is better.**
- Steps: n = ⌈HW/step⌉. The reference uses step = 224 (one row's worth at 224×224); the batch evaluation uses 224×8.
- **AUC** (reference code): `auc(h) = (h.sum() - h[0]/2 - h[-1]/2) / (len(h)-1)`. This is the trapezoid rule over x ∈ [0,1].
- Reference ImageNet numbers (ResNet50, deletion↓/insertion↑): Grad-CAM 0.1232/0.6766, RISE 0.1076/0.7267.
- **Detector adaptation [derived from D-RISE]**: define h_i = max over detections in the perturbed image of IoU(box_t, box_j) × cos(p_t, p_j) × objectness_j. Here p_t is one-hot for the target class. If the YOLOv8 head has no objectness, drop that term. Run at 320 px with step = 1% of pixels: 101 forward passes per explanation, which is fine offline.

### 2.2 ROAD (Remove And Debias, Rong et al. 2022)
- Removed pixels are filled by **noisy linear imputation** from their neighbours (noise 0.01 in Quantus) instead of a constant. This limits the out-of-distribution artefacts and information leakage of plain deletion.
- Quantus default removal percentages: 1, 3, …, 99. MoRF = most-relevant-first. pytorch-grad-cam ships `ROADMostRelevantFirst`, `ROADLeastRelevantFirst` and `ROADCombined` (both orders combined).
- Use it as a robustness check next to deletion/insertion.

### 2.3 Pointing game (Zhang et al., Excitation Backprop 2016)
- Take the single arg-max pixel of the saliency map (resized to the image). It is a **hit** if it falls inside any GT box of the cued class, **expanded by a 15-px tolerance**; otherwise a miss.
- Acc_c = hits / (hits + misses) per category. Report the **mean over categories**.
- "Difficult" subset: the class's total box area is < 1/4 of the image **and** at least one distractor class is present.

### 2.4 Energy-based pointing game (Score-CAM, Wang et al. 2020)
- Binarise the GT box(es) into a mask B (1 inside, 0 outside). Use the non-negative, min-max-normalised saliency L.
- **Proportion = Σ_{x∈B} L(x) / Σ_x L(x)**. (The printed formula in the paper is garbled; this is the intended meaning.)
- Score-CAM 63.7% vs Grad-CAM 48.1% on 500 ILSVRC val images (single box, object < 50% of the image).
- **[derived]** Also report a **lift** = Proportion / (box area / image area), so that large boxes don't inflate the score. A uniform map gives lift = 1.

Sources: https://arxiv.org/pdf/1806.07421 ; https://github.com/eclique/RISE (`evaluation.py`, `Evaluation.ipynb`) ; https://arxiv.org/pdf/1608.00507 ; https://arxiv.org/pdf/1910.01279v2 ; https://quantus.readthedocs.io/en/stable/docs_api/quantus.metrics.faithfulness.road.html ; https://arxiv.org/abs/2202.00449v2

---

## 3. Pose-based fall detection

COCO-17 indices (YOLO-pose): 0 nose, 5/6 shoulders, 11/12 hips, 13/14 knees, 15/16 ankles.

### 3.1 Features (definitions)
- **Torso angle from vertical**: v = mid_shoulder − mid_hip; θ = atan2(|v_x|, |v_y|) in degrees (0° = upright).
  - Chen et al. 2020 (OpenPose): centreline-to-ground angle < 45° indicates a fall, i.e., θ > 45°.
  - YOLOv7-W6-Pose paper: same 45° threshold.
  - A public YOLOv8-pose demo uses θ > 30° for ≥ 3 consecutive frames. This is too sensitive: bending forward triggers it.
- **Hip-centre descent speed**: Chen et al. use v = |Δy_hip| / Δt, sampled every 5 frames (0.25 s). The threshold is given inconsistently (0.009 vs 0.09 m/s) and in metres, so it is not usable in pixels.
  - **[derived] Normalise by body height**: v_norm = (y_hip(t) − y_hip(t−Δ)) / (H_ref·Δ), with image y pointing down. H_ref = the person's upright height in pixels (median bbox height over the last ~2 s while θ < 30°), falling back to the current bbox height.
  - A fall lasts about 0.4–0.8 s [src: Kinect study citing the literature], and the hip drops roughly 0.4 H_ref. **Start with v_norm ≥ 0.6 H/s over a 0.3–0.5 s window** and tune on URFD/Le2i.
- **Torso-angle rate**: Ω = Δθ/Δt. The BAM-HRNet+YOLOv8 paper (Sensors 2025) uses Ω > 75 (°/s), centroid descent V > 350 (px/s, units unstated) and aspect A = H/W < 1, checked every 15th frame. It reports > 95% accuracy on URFD at 18 fps.
- **Aspect ratio**: W/H of the person bbox ≥ 1 means lying [src: Chen 2020; YOLOv7-pose paper uses H < W]. The URFD feature set also provides HeightWidthRatio and **HHmaxRatio** (current height / standing height).
- **MoveNet rule (normalised coordinates)**: vertical gap between upper- and lower-body keypoints Δy ≤ 0.05 and horizontal gap Δx > 0.5, for 2 consecutive frames. URFD sensitivity 0.917. Keypoint confidence cut-off 0.5.
- **On-ground / inactivity confirmation**:
  - ElderFallGuard (MediaPipe): fall pose held > 3 s **and** sustained motion drop > 2 s; 5–10 min alert cooldown.
  - YOLOv7-pose paper: at most one alert per 2 min.
  - Chen 2020: cancel the alarm if the person gets up (H/W < 1 and centreline angle > 45°) within the recovery window.

### 3.2 Reported accuracy (for context)

| Work | Pose | Data | Result |
|---|---|---|---|
| Chen et al. 2020 (rules: speed + 45° + W/H) | OpenPose | self-built, 100 videos | Acc 97, Sens 98.3, Spec 95 |
| YOLOv7-W6-Pose rules (45°, H<W, shoulder near feet) | YOLOv7-pose | Le2i, 130 seq | Acc 95.7, P 94.7, R 96.4, Spec ~95, F1 95.5 (361 ms/frame pose) |
| Hasan et al. 2019 | OpenPose + LSTM | URFD / Le2i | Sens 99 / Spec 96 ; Sens 99 / Spec 97 |
| MediaPipe + LSTM (IJACSA 2025) | MediaPipe | URFD test, 21 videos | Acc 95.2, Sens 1.0, Spec ~91.7 |
| Trunk-angle-only vs fused features | MediaPipe + att-GRU | URFD + Multicam + FDD | 84.4% (angle only) vs 96.2% (fused) |
| MoveNet rule | MoveNet Thunder | URFD | Sens 0.917 |

Dataset facts:
- **URFD**: 70 sequences (30 falls, 40 ADL), Kinect; camera 0 parallel to the floor.
- **Le2i**: per the review, 143 fall + 79 ADL videos, 320×240 at 25 fps. The common Coffee room + Home protocol has 130 videos (99 falls / 31 ADL).

### 3.3 Recommended interpretable rule (per tracked person) [derived]
1. Valid pose: at least one shoulder and one hip with confidence ≥ 0.3, otherwise mark cue "missing".
2. **Descent cue**: v_norm(0.4 s window) ≥ 0.6 H/s, or Ω ≥ 75°/s.
3. **Horizontal cue**: θ ≥ 60°, or (θ ≥ 45° and W/H ≥ 1.0), or HHmax ≤ 0.5.
4. **Ground-dwell cue**: horizontal cue held ≥ 2–3 s with low keypoint motion (mean keypoint speed < 0.1 H/s).
5. Recovery (θ < 30° for ≥ 1 s) cancels the alert. Cooldown of 2 min per track.

Feed the continuous values (v_norm, θ, W/H, dwell seconds) to the fusion stage (Sec. 6) rather than hard-ANDing them, so each value gets its own attribution.

Sources: https://www.mdpi.com/2073-8994/12/5/744 ; https://www.mdpi.com/1999-5903/16/12/472 ; https://www.mdpi.com/1424-8220/25/13/4128 ; https://arxiv.org/pdf/2207.10952 ; https://fenix.ur.edu.pl/~mkepski/ds/uf.html ; https://arxiv.org/pdf/2505.11845 ; https://www.themoonlight.io/en/review/real-time-human-fall-detection-using-a-lightweight-pose-estimation-technique ; https://thesai.org/Downloads/Volume16No8/Paper_11-A_New_Method_for_Real_Time_Fall_Detection.pdf ; https://ijtech.eng.ui.ac.id/download/article/5840 ; https://www.scitepress.org/Papers/2014/47424/47424.pdf ; https://huggingface.co/spaces/luminoussg/yolofallpose_public/blob/main/app.py ; https://docs.ultralytics.com/tasks/pose/

---

## 4. Smoke density and haze cues

### 4.1 Dark channel prior (He, Sun, Tang)
- Haze model: I(x) = J(x)t(x) + A(1 − t(x)).
- Dark channel: dark(x) = min_{y∈Ω(x)} min_{c∈{r,g,b}} I^c(y), with a **15×15 patch**.
- Statistic: in haze-free outdoor images, ~90% of dark-channel pixels are < 25 (of 255) and ~75% are exactly 0.
- Atmospheric light A: take the top 0.1% brightest pixels of the dark channel; among them, the brightest pixel in I.
- Transmission: t(x) = 1 − ω · min_Ω min_c (I^c/A^c), with **ω = 0.95**. Lower bound t₀ = 0.1 when dehazing. The paper refines t with soft matting; a guided filter is the usual fast substitute.
- **Smoke concentration** α = 1 − t. Stacking α with RGB as a 4-channel input to YOLOv7-tiny raised mAP@0.5 from 89.48 to 92.33 and recall from 88.21 to 91.12 [src: SAI 2023].
- Cost [bench]: min over channels + 15×15 erode takes 2.6 ms at 320×240 and 9.5 ms at 640×480. **Compute it only inside detection boxes plus a reference ring.**

### 4.2 Other interpretable cues (definitions and published values)
- **Grey-ness / low saturation**:
  - HSV rule S < 0.28 and V > 108 (EUSIPCO 2014). A 16×16 block is a candidate if ≥ 10% of its pixels are moving and smoke-coloured.
  - RGB rule: R, G and B within ±α of each other, **α = 15–20**, with intensity in a dark-grey band 80–150 or a light-grey band 150–220 (US7609856).
- **Chrominance drop**: smoke lowers U and V relative to the background [Töreyin 2006]. Shadows are a confounder because they keep the background's RGB direction.
- **Contrast / edge-energy drop**: semi-transparent smoke softens edges. Wavelet energy E = HL² + LH² + HH², averaged per block. A block is smoke-like when T₂·E_bg < E < T₁·E_bg with 0 < T₂ < T₁ < 1. No numeric T values are published; calibrate them. A sudden drop to ~0 indicates an opaque occluding object, not smoke.
- **Temporal variance**: V = (1/T) Σ_t (E_t − Ē)², the spatio-temporal energy variance per block.
- **Growth**: average rate of change of the smoke-pixel area, ΔA/Δt, over n steps (patent).
- **Disorder**: perimeter / area of the smoke region (patent).
- **Entropy drop [derived]**: H = −Σ p_k log₂ p_k of the grey histogram in the box, compared with the same region in the background model.

### 4.3 Pairing with detector confidence
- **Uncertainty-aware post-detection** (arXiv 2510.10108): per YOLOv8n box, inputs are [conf, MC-dropout variance, HSV saturation/colour score, Canny edge score, Haralick contrast/homogeneity], fed to an MLP (2×32 units), keep if ĉ ≥ 0.5.
  - D-Fire results: precision 0.712 → 0.845, recall 0.674 → 0.82.
  - Latency 12.8 → 20.2 ms.
  - **For explainability, swap the MLP for logistic regression** on the same features, as in Sec. 6.
- **Fusion of per-cue probabilities** (EUSIPCO 2014): a = mean(p_i); score = (1 − a)·min(p_i) + a·max(p_i); smoke if ≥ 0.5. Detection rate 93.37% vs 84.08% for the baseline. Failures: smoke in front of the sun (73%) and thin smoke (82%).
- **[derived] Recommended smoke feature vector per box** (each as a z-score against the same region in a slow background model, EMA α ≈ 0.01/frame):
  - detector logit
  - Δdark (mean dark channel, box − background); higher means hazier
  - Δsaturation (negative for smoke)
  - edge-energy ratio E/E_bg
  - Δentropy
  - temporal variance of the edge energy over 2–3 s
  - box-area growth rate
  - fire only: flicker energy, red/yellow ratio
- Known confounders to log in explanations: fog, cloud, steam, dust, shadows, lights and sun glare.

Sources: https://people.csail.mit.edu/kaiming/publications/pami10dehaze.pdf ; https://ftp.saiconference.com/Downloads/Volume14No9/Paper_114-Improved_Model_for_Smoke_Detection_Based_on_Concentration_Features.pdf ; https://kilyos.ee.bilkent.edu.tr/~eee520/resources/eusipco2006_2.pdf ; https://new.eurasip.org/Proceedings/Eusipco/Eusipco2014/HTML/papers/1569925173.pdf ; https://patents.google.com/patent/US7609856 ; https://arxiv.org/pdf/2510.10108 ; https://opus.lib.uts.edu.au/handle/10453/115055

---

## 5. Crowd panic / abnormal motion

### 5.1 Optical flow on a 2-core CPU
- OpenCV `FarnebackOpticalFlow::create` defaults: levels 5, pyrScale 0.5, winSize 13, iters 10, polyN 5, polySigma 1.1. The common functional call is `calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)`.
- Cost [bench, 2 threads]:

| Resolution | Typical params | Lighter params (0.5,2,9,2,5,1.1) |
|---|---|---|
| 160×120 | 4.1 ms | 3.2 ms |
| 320×240 | 21 ms | 15 ms |
| 640×480 | 74 ms | 58 ms |

- **Use 160×120 (or 320×240 for wide scenes), every 2nd frame.**

### 5.2 Features (exact definitions) [derived formulas, standard]
Let f_i = (u_i, v_i) on foreground or moving pixels M = {i : ‖f_i‖ > τ_m}. Take τ_m ≈ 0.3–0.5 px/frame at 160 px width.
- **Mean / p95 magnitude**: m̄ = mean‖f_i‖. Normalise by image diagonal or by the median person height (from the detector) to get "body-heights per second".
- **Direction entropy**: histogram of angles atan2(v,u) in K = 8 (or 16) bins, magnitude-weighted, giving p_k. H_dir = −Σ p_k ln p_k / ln K ∈ [0,1].
  - Scattering or panic: high H plus high m̄.
  - Orderly flow: low H.
  - Collective flight in one direction: low H but a magnitude spike.
- **Coherence (mean resultant length)**: R = ‖Σ w_i (cos φ_i, sin φ_i)‖ / Σ w_i with w_i = ‖f_i‖. Circular variance = 1 − R.
- **Divergence / curl**: smooth the flow (Gaussian σ ≈ 2 px), then div = ∂u/∂x + ∂v/∂y and curl = ∂v/∂x − ∂u/∂y via Sobel. Mean positive divergence on M indicates radial dispersal (the UMN escapes are radial scatter). Div-curl descriptors with conditional entropy were used by Pattern Recognition 2019 (DOI 10.1016/j.patcog.2018.11.023).
- **Kinetic-energy jump**: KE_t = mean‖f‖². Use the ratio KE_t / EMA_baseline(KE). Panic onset is a sudden simultaneous speed-up.
- **ViF (Violent Flows, Hassner 2012)**:
  - Binary map b = 1{|m_t − m_{t−1}| ≥ θ_t}, where θ_t = mean of |Δm| over the frame.
  - Average b over T frames, then 4×4 grid, then per-cell histograms (20 bins; 6 for real time), then a linear SVM.
  - Results: 81.3% accuracy, AUC 0.85 on Violent Flows, 30 ms/frame. Real-time mode uses 5-frame windows and processes every 3rd frame.
- **Social Force Model (Mehran, Oyama, Shah 2009)**:
  - Place particles on a grid and advect them with the space-time-averaged flow O_ave.
  - Desired velocity: v^q = (1 − p)·O(x,y) + p·O_ave(x,y), where p is the panic weight.
  - Interaction force: F_int = (1/τ)(v^q − v) − dv/dt. Some re-implementations drop τ.
  - Force-flow patches feed a bag-of-words classifier.
  - **UMN AUC: social force 0.96 vs pure optical flow 0.84.**

### 5.3 UMN reference numbers (frame-level ROC AUC)
- UMN: 11 sequences, 3 scenes, 320×240 at 30 fps. Lawn: 2 videos, 1,433 frames. Indoor: 6 videos, 2,031 frames. Plaza: 3 videos, 4,038 frames.

| Method | AUC |
|---|---|
| Pure optical flow (Mehran baseline) | 0.84 |
| Social force | 0.96 |
| Interaction Energy Force (optical-flow based) | 0.974 overall |
| Tracklet features (density, speed, collectiveness, conflict) + GMM, single scene / cross scene | 0.929 / 0.869, at 40 fps |
| Entropy of flow-magnitude "activity map" + 2 thresholds (prior work) | lawn 0.96, plaza 0.97, indoor 0.88 |
| Same features + stacking (RF + SVC) | lawn 0.99, plaza 0.99, indoor 0.98, ~54 fps on i7-9700K |
| Sparse reconstruction (SRC) | 0.995 / 0.975 / 0.964 |
| MDT | 0.995 (0.9 fps) |
| Self-adaptive SFM | 0.950 |

- **Caveat**: UMN is saturated. Simple features reach about 0.93–0.99. Also report a harder or cross-scene set.
- **[derived] Protocol**: the first 200 frames of each clip are "normal" (as in arXiv 1606.05310). Convert each feature to a z-score against a running calm-period baseline: μ and σ by EMA, updated only while no alert is active.

Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC3892866/table/t3-sensors-13-17130 ; https://mlanthology.org/cvpr/2009/mehran2009cvpr-abnormal ; https://so01.tci-thaijo.org/index.php/APST/article/download/101488/78647/256054 ; https://arxiv.org/pdf/1606.05310 ; https://ceur-ws.org/Vol-3097/paper29.pdf ; https://www.openu.ac.il/Lists/MediaServer_Documents/Academic/CS/RealTimeDetectionOfViolent.pdf ; https://journal.ump.edu.my/mekatronika/article/download/916/271 ; https://docs.opencv.org/4.x/de/d9e/classcv_1_1FarnebackOpticalFlow.html

---

## 6. Evidence fusion with exact attributions

### 6.1 Additive log-odds fusion (primary)
- Model: **L(x) = logit P(E|x) = b + Σ_i g_i(x_i)**; P = σ(L).
  - **Logistic**: g_i(x_i) = w_i·x̃_i, with x̃ standardised.
  - **Naive Bayes**: b = log prior odds and g_i(x_i) = log[P(x_i|E) / P(x_i|¬E)], using binned likelihood ratios with Laplace smoothing. Same additive form [src: Možina et al. 2004, nomograms].
- **Missing cue** (occluded pose, no box): g_i = 0, i.e., neutral evidence. The NB nomogram assigns 0 points to unknown values.
- **Exact Shapley values** (additive model, independent or interventional features):
  - φ_i = g_i(x_i) − E[g_i(X_i)]. For linear terms: **φ_i = w_i(x_i − E[x_i])** (SHAP LinearExplainer). Cost O(n).
  - Efficiency: **Σ_i φ_i = L(x) − E[L(X)]**. The attributions are exact and live in **log-odds units**. Report them as "evidence points": e.g., ×10/ln10 for decibans, or the nomogram scaling where the largest |log OR| = 100 points.
  - **[derived]** Alternative baseline: x⁰ = "cue absent". Then φ_i = g_i(x_i) − g_i(x_i⁰) and Σφ = L(x) − L(x⁰). This is easier to verbalise ("raised the alert odds by ×e^{φ}").
- Probability-space attributions are **not** additive. Keep the explanations in log-odds and show the final probability separately.
- **[derived]** Interpretable non-linearity: g_i = monotone piecewise-linear (3–5 knots) or binned. Fit with L2 regularisation and enforce monotonic sign constraints (e.g., more darkening never lowers smoke evidence).

### 6.2 Dempster–Shafer (optional alternative)
- Frame Θ = {E, ¬E}. Each cue gives masses m(E), m(¬E), m(Θ) (ignorance).
- Combination: m₁₂(A) = [Σ_{B∩C=A} m₁(B)m₂(C)] / (1 − K), with conflict K = Σ_{B∩C=∅} m₁(B)m₂(C).
- Bel(A) = Σ_{B⊆A} m(B); Pl(A) = 1 − Bel(¬A).
- **Reliability discounting** (Shafer, standard): m'(A) = r·m(A) for A ≠ Θ, and m'(Θ) = 1 − r + r·m(Θ).
- Pro: explicit "unknown" mass when a cue is unavailable.
- Con: there is no exact additive attribution, and Zadeh's example shows that high conflict (K = 0.9999) gives absurd certainty. **Use log-odds as the primary model.** If DS is used, report K as a "conflicting evidence" flag.

### 6.3 Temporal smoothing for alerts
- **EMA on log-odds**: L̄_t = β·L̄_{t−1} + (1 − β)·L_t, with β = exp(−Δt/τ).
  - **[derived]** τ ≈ 1 s (fall), 3–5 s (smoke), 1–2 s (crowd).
  - Attributions stay exact: φ̄_{i,t} = β·φ̄_{i,t−1} + (1 − β)·φ_{i,t}, and Σ_i φ̄_i = L̄_t − E[L].
- **2-state HMM / Bayes filter**:
  - Predict: π_t = (1 − p_off)·P_{t−1} + p_on·(1 − P_{t−1}).
  - Update: logit P_t = logit π_t + [L_t − b], i.e., the evidence log-likelihood ratio.
  - **[derived]** p_on ≈ 0.01/frame, p_off ≈ 0.05/frame.
  - Behaves like a bounded CUSUM. Attribution per frame is still additive within the update term.
- **Hysteresis**: alert ON when P̄ ≥ θ_on for k_on consecutive frames; OFF when P̄ ≤ θ_off for k_off frames; θ_on > θ_off.
  - **[derived]** θ_on = conformal threshold (Sec. 7), θ_off = θ_on − 0.2 in probability (or −1.5 in log-odds), k_on = 3–5, k_off = 15.
  - Per-event cooldown: fall 2 min [src], smoke and crowd 1 min [derived].

### 6.4 Counterfactuals for additive models
- Alert iff L(x) ≥ τ (τ = logit θ_on). Gap Δ = τ − L(x).
- **Single-cue counterfactual**: find x_i' with g_i(x_i') = g_i(x_i) + Δ.
  - Linear: **x_i' = x_i + Δ/w_i**. This is the boundary action from Ustun, Spangher & Liu 2019, Remark 2.
  - Monotone piecewise g_i: invert by bisection.
  - Keep only cues where x_i' lies in the feasible domain (e.g., 0 ≤ conf ≤ 1, angle ≤ 90°). Rank by standardised change |x_i' − x_i|/σ_i, or by percentile shift |Q_i(x_i') − Q_i(x_i)| (Ustun's cost).
- **Necessity test** (cheap and very readable): cue i is *necessary* if L(x) − φ_i < τ, i.e., without it the alert would not fire. A set S is *sufficient* if b + Σ_{i∈S} g_i ≥ τ (other cues at baseline).
- If no single cue can flip the decision, search pairs greedily, or solve Ustun's small integer program over discretised actions.
- Example verbalisation: "Had the torso angle been ≤ 38° (it was 71°), no fall alert would have fired."

Sources: https://www.datavis.ca/papers/titanic/sources/Mozina_etal_PKDD_2004_p337-348.pdf ; https://cdn.jsdelivr.net/npm/claude-code-skills@0.5.0/data/skills/shap/references/theory.md (SHAP linear-model summary) ; https://arxiv.org/pdf/1809.06514 ; https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory

---

## 7. Calibration and conformal alert thresholds

### 7.1 Calibration
- **Temperature scaling**: p = σ(z/T) (binary) or softmax(z/T). Fit the single T by minimising NLL on held-out data. Ranking is unchanged [Guo et al. 2017; sklearn `method="temperature"`].
- **Platt**: p = 1/(1 + exp(A·f + B)), fit by logistic regression. Best for small calibration sets. Platt's target smoothing: y₊ = (N₊ + 1)/(N₊ + 2), y₋ = 1/(N₋ + 2).
- **Isotonic**: monotone step function minimising squared error. Use only with > ~1,000 calibration samples (otherwise it overfits). It can create ties that change ROC-AUC.
- **ECE**: bins I_m = ((m−1)/M, m/M].
  - **ECE = Σ_m (|B_m|/n)·|acc(B_m) − conf(B_m)|**; MCE = max_m |acc − conf|.
  - For a binary alert, use the positive-class version: per-bin fraction of positives vs mean predicted probability. Use M = 10–15. Report the Brier score alongside, noting it mixes calibration and discrimination.
- **Detector-level (D-ECE, Küppers 2020)**: precision replaces accuracy. A detection counts as a TP if its class matches and IoU ≥ 0.6. Histogram binning is best for confidence-only calibration; logistic or beta calibration when box position or size is included.
- **[derived] Where to calibrate**: (1) the detector confidence per class (Platt or temperature) before it enters fusion; (2) the fused L via a final Platt layer. The logistic fusion is already approximately calibrated if it was trained on representative data.

### 7.2 Conformal false-alarm control
- **Split conformal (outlier form)**:
  - Calibration data: n normal (no-emergency) segments, e.g., 10-s windows from different days or cameras, so they are approximately exchangeable.
  - Score s_j = max over the segment of the smoothed alert score L̄.
  - Threshold: **λ̂ = the ⌈(n+1)(1−α)⌉-th smallest s_j**. Alarm iff score > λ̂.
  - Guarantee: P(false alarm on a new normal segment) ≤ α (marginal).
  - Feasibility: need ⌈(n+1)(1−α)⌉ ≤ n, i.e., **n ≥ 1/α − 1**.
  - False alarms per hour ≲ α × (3600 / segment length).
- **Conformal Risk Control** (Angelopoulos et al., arXiv 2208.02814):
  - Losses L_i(λ) non-increasing and right-continuous in λ, bounded by B, with L_i(λ_max) ≤ α.
  - **λ̂ = inf{λ : (n/(n+1))·R̂_n(λ) + B/(n+1) ≤ α}**, where R̂_n is the mean calibration loss.
  - Guarantee: E[L_{n+1}(λ̂)] ≤ α. Lower bound: ≥ α − 2B/(n+1) (i.i.d., no jumps).
  - With L_i(λ) = 1{s_i > λ} on normal segments, this reduces exactly to the quantile rule above.
  - CRC also controls **graded losses**, e.g., L = fraction of a normal clip's duration spent in alert, or the false-negative rate on positive clips with λ parametrised so the loss is non-increasing.
- **Two-sided recipe for the prototype [derived]**:
  1. Fit the cue transforms and fusion on a train split. Calibrate detector confidences (Platt) on a validation split.
  2. On a disjoint calibration split: from normal segments pick λ̂ for α_FA (e.g., 0.01 per 10-s segment, so n ≥ 99, ideally ≥ 1,000).
  3. On positive events, report the miss rate at λ̂ with a Clopper–Pearson upper bound. If several α must hold jointly (per class or camera), use Learn-Then-Test with Bonferroni-corrected binomial p-values for probability 1 − δ guarantees.
  4. Re-calibrate per camera or scene if the distribution shifts. Temporal correlation breaks exchangeability, so calibrate on segments, not frames.
- Sample size reference: n = 1,000 gives empirical coverage of 0.88–0.92 at α = 0.1. With δ = 0.1, slack 0.1 / 0.05 / 0.01 needs n = 22 / 102 / 2,491 [Angelopoulos & Bates, gentle intro].
- Conformal object detection (railway, arXiv 2304.06052) targets false negatives (box coverage and recall), not false positives. Our false-alarm use is the outlier-detection form.

Sources: https://arxiv.org/pdf/1706.04599 ; https://scikit-learn.org/stable/modules/calibration.html ; https://arxiv.org/pdf/2004.13546 ; https://arxiv.org/pdf/2208.02814v4 ; https://arxiv.org/pdf/2107.07511 ; https://mapie.readthedocs.io/en/latest/theoretical_description_risk_control.html ; https://arxiv.org/pdf/2304.06052

---

## 8. Grounded natural-language explanations

### 8.1 Template vs LLM
- **Templates** are faithful by construction and need near-zero CPU. They are the default on the edge.
  - LLM-written rule-based (Python) data-to-text systems had fewer minor hallucinations than a prompted Llama-3-70B (0.04 vs 0.08) and than fine-tuned BART (0.20). Major hallucinations: 0.13 vs 0.07 (LLM) vs 0.33 (BART).
  - The rule-based system ran the whole WebNLG test set in ~3 s on one CPU, vs 1,910 s for BART on CPU [arXiv 2502.20609].
- **LLMs** read more fluently but make factual errors. In human evaluation, Llama2-7B outputs contained incorrect or irrelevant facts in 12–21% of cases (E2E / DART / WikiTableText). Consistency improves with scale and degrades as source and reference diverge [arXiv 2411.19203].
- **[derived] Hybrid**: a template produces the canonical sentence set from an evidence JSON. An optional LLM paraphrase runs off-device or asynchronously, must cite evidence keys, and goes through the verifier below. If verification fails, the template output is used.

### 8.2 Claim verification against structured evidence [derived pipeline, metrics from sources]
Evidence JSON example: `{event:"fall", track:3, zone:"corridor-2", t:"14:03:21", P:0.93, cues:{torso_angle_deg:71, v_norm:0.84, dwell_s:4.2}, phi:{torso_angle:1.9, v_norm:1.1, dwell:0.7}, saliency_energy_in_box:0.78}`.
1. **Claim extraction**: split into sentences, then atomic claims (one fact each, following the FActScore decomposition). For template output the claims are known by construction.
2. **Typed checks**:
   - **Numbers**: regex for numbers plus units or percentages. Map each to an evidence key (by the cited key or by unit and nearest match). Supported iff |claim − value| ≤ max(abs_tol, rel_tol·|value|), with rounding-aware tolerances (e.g., "about 70°" vs 71 → rel_tol 0.05; percentages vs probabilities → ×100).
   - **Entities / categories** (event type, zone, track id, time, class names): exact match against a closed vocabulary from the evidence.
   - **Qualitative predicates** ("rapid descent", "dense smoke", "increasing"): a lexicon maps phrases to predicates (e.g., "rapid" ⇔ v_norm ≥ 0.6; "increasing" ⇔ slope > 0 over the window).
   - **Attribution claims** ("mainly due to X"): X must be argmax φ_i. "X and Y" must be the top-2. A claimed share must match φ_X / Σ⁺φ.
   - **Counterfactual claims**: recompute with the Sec. 6.4 formula.
   - **Free text not covered by a rule**: NLI or hallucination classifier. Example: Vectara HHEM-2.1-Open (T5-based, runs on CPU), as used in RAGAS `FaithfulnesswithHHEM`.
3. **Decision**: any unsupported or contradicted claim triggers regeneration (max 1 retry), then the template fallback.

### 8.3 Metrics
- **Claim-level faithfulness / groundedness** = supported claims / total claims. This is RAGAS faithfulness, and FActScore f(y) = (1/|A_y|) Σ_{a∈A_y} 1[a supported]. The automated FActScore estimator had < 2% error vs humans for two of the three models tested.
- **Fully-grounded rate** = fraction of explanations with zero unsupported claims. This is the headline safety metric.
- **Numeric accuracy** = correct numbers / numbers mentioned. **Entity hallucination rate** = unmatched entities / entities mentioned.
- **Coverage / completeness** = fraction of the top-k (k = 3) attributions mentioned. **Contradiction rate** = claims refuted by evidence.
- Also report abstention or fallback rate and claims per explanation (FActScore recommends reporting these next to precision).
- Validate the automatic checker against a human-labelled sample (e.g., 100 explanations) and report agreement.

Sources: https://arxiv.org/pdf/2502.20609 ; https://arxiv.org/html/2411.19203v1 ; https://arxiv.org/pdf/2305.14251 ; https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/ ; https://arxiv.org/pdf/2202.03629

---

## 9. Edge optimisation on CPU (ONNX Runtime)

### 9.1 INT8 quantisation
- ORT guidance:
  - **Static quantisation for CNNs** (dynamic is for RNNs and transformers). QDQ format; **S8S8 QDQ is the x86-64 default**. Try U8U8 if accuracy drops.
  - U8S8 on AVX2/AVX512 can saturate; use `reduce_range=True` (7-bit weights), which is not needed on VNNI.
  - `per_channel=True` helps when weight ranges are wide (needs reduce_range on AVX2/512).
  - Calibration methods: MinMax, Entropy, Percentile.
  - Run `quant_pre_process()` first.
  - Debug with `qdq_loss_debug`, then exclude diverging nodes via `nodes_to_exclude`.
- **Keep the YOLO head in FP32.** Box coordinates and class scores have very different ranges.
  - YOLOv8n FP32 37.3 mAP50-95.
  - OpenVINO: INT8 35.7, INT8 + FP32 head 37.1.
  - TFLite: INT8 32.9, INT8 + FP32 head 35.2. Per-channel adds about +1 mAP. Excluding the head cost only 0.3% speed.
  - ORT example exclusion list for YOLOv8: `/model.22/` dfl (Reshape, Transpose, Softmax, Conv), Concat_3/4/5, Split, Sigmoid, Slice, Add_1, Sub, Div_1, Mul_2. YOLOv8n static INT8 went from 9 to 11 FPS (+22%). That example calibrated on a single image; **use 200–1,000 representative frames [derived; the YOLO12 study used 1,000]**.
- Other reported magnitudes:
  - Ultralytics OpenVINO YOLO26n: INT8 mAP 0.4765 → 0.4573–0.4634 (−1.3 to −1.9 pts); 9.1 → 5.8 ms on a Core Ultra 7 155H.
  - Naive full-INT8 TensorRT YOLO12n: 0.405 → 0.333 (−7.2 pts), which is why head exclusion matters.
  - INT8 is slightly more noise-sensitive than FP32.
  - ORT INT8 on a Raspberry Pi 4: ~2× speed-up, ~4× smaller model, "small" accuracy loss.
- **[derived] Expectation for our 2-core x86**: 1.2–2× speed-up and −0.5 to −2 mAP with the head kept in FP32. Always measure mAP on our own validation set before and after.

### 9.2 Threads and session settings
- `intra_op_num_threads` defaults to one per physical core. Setting it explicitly disables ORT's affinity pinning.
- `ORT_SEQUENTIAL` (default) suits single-branch CNNs. Benchmark `ORT_PARALLEL` (it can help or hurt).
- `session.intra_op.allow_spinning = "0"` stops idle workers busy-waiting. Use it, because OpenCV and NumPy share the same 2 cores.
- Several sessions (detector + pose) each create their own pool and contend for cores. Either run them sequentially with 2 threads each, or pin 1 thread each with `session.intra_op_thread_affinities`, or use a global thread pool.
- **[derived]** Call `cv2.setNumThreads(1)` and set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` to 1–2 to avoid oversubscription. Use `graph_optimization_level = ORT_ENABLE_ALL`.

### 9.3 Input resolution
- YOLOv8n at 640: 37.3 mAP, 8.7 GFLOPs, 80.4 ms CPU-ONNX (Ultralytics reference CPU). YOLO26n-pose at 640: 57.2 mAP-pose, 40.3 ms CPU-ONNX.
- FLOPs scale with pixel count, so 320 px ≈ 1/4 of the compute but loses small objects. **[derived]**
  - 320–416 px for near-field indoor falls and person detection.
  - 480–640 px for distant smoke.
  - Cascade: run 320 px continuously and re-run at 640 on an ROI crop when a cue fires.

### 9.4 Motion-gated cascades and frame skipping (reported savings)
- **Reducto** (on-camera frame differencing; pixel, edge and area features; threshold picked per segment from a k-means-clustered table): filters a median 51–97% of frames, cuts backend compute 50–96% and bandwidth 21–86%, keeps ≥ 90% accuracy. A static pixel-difference threshold (Glimpse) almost never met the accuracy target.
- **FrameHopper** (RL skip-length policy): ~91% of frames filtered at a target F1 of 0.8; ~4× throughput on a Jetson Nano (16.8 → 63.5 fps) vs ~3× for Reducto.
- **NoScope**: difference detector (> 100k fps on CPU), then a specialised tiny CNN, then the full model. Up to ~1000× on binary queries.
- ViF processes every 3rd frame.
- **[derived] Our cascade**:
  1. Every frame: grey 160×120 frame difference (< 1 ms): fraction of pixels with |Δ| > 15.
  2. Every 2nd frame: Farneback at 160×120 (3–4 ms) to get crowd features.
  3. Detector (person, smoke, fire) at 320–416 px at 2–5 fps, plus immediately when motion exceeds an adaptive threshold (per-segment percentile, not static).
  4. Pose only on person crops when a person is present.
  5. **Heartbeat**: run the detector at ≥ 0.5 fps even without motion. Slow smoke and a motionless fallen person produce little motion, so gating must never suppress the dwell timer.

Sources: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html ; https://onnxruntime.ai/docs/performance/tune-performance/threading.html ; https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions ; https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2 ; https://docs.ultralytics.com/integrations/openvino ; https://arxiv.org/pdf/2508.19600 ; https://arxiv.org/pdf/2501.17062 ; https://docs.ultralytics.com/models/yolov8/ ; https://www.cs.princeton.edu/~ravian/publications/reducto_sigcomm20.pdf ; https://arxiv.org/pdf/2203.11493 ; https://dawn.cs.stanford.edu/2017/06/22/noscope

---

## 10. Privacy-preserving video analytics

- **Data minimisation [derived]**:
  - Process on-device.
  - Persist only derived signals: cue values, attributions, keypoints if needed, flow statistics.
  - Keep a short ring buffer (e.g., 10 s) in RAM. Write a clip (e.g., 5 s before + 10 s after) only on an alert, with faces redacted, encrypted at rest, auto-deleted after a fixed retention, and with access logging.
- **Face redaction**:
  - OpenCV **YuNet**: 337 KB ONNX, ~5.1–5.4 ms at 320×320 and ~22 ms at 640×640 on an i7-10710U. It struggles with small faces.
  - **[derived]** Fallback: redact a head region from pose keypoints (nose, eyes, ears, expanded by 1.5×), or the top ~20% of the person box. Prefer a solid fill or strong pixelation over light blur.
- **Skeleton-only is not anonymous**:
  - Skeleton sequences allow ~80% top-1 re-identification (NTU60, ~97% top-5) and ~87% gender inference (ETRI).
  - An adversarial anonymiser cut re-ID from 0.81 to 0.04 while action accuracy dropped from 0.95 to 0.92 [Moon et al., arXiv 2111.15129].
  - Treat stored keypoint tracks as personal data too: retain only aggregates or short windows.
- **India DPDP Act 2023 + DPDP Rules 2025** (brief, not legal advice):
  - Scope: the Act covers digital personal data of identifiable individuals, which includes camera footage once digitised. Processing needs notice + consent **or** a "legitimate use" (Sec. 7). Legitimate uses include **medical emergencies** and **breakdown of public order**, which are relevant to emergency detection.
  - Fiduciary duties: accuracy, **reasonable security safeguards** (encryption, masking, access control, logging), breach notice to the Data Protection Board and affected persons, **erasure once the purpose is served**.
  - Children (< 18): parental consent; no tracking or behavioural monitoring of children. This matters in schools.
  - Penalty: up to **₹250 crore** for security-safeguard failures.
  - Rules notified 13 Nov 2025, phased in. Notice, consent, security, breach-intimation (detailed report within 72 h) and retention duties apply from **13 May 2027**.
  - Design implications: purpose limitation (emergency detection only), retention limits, on-device processing, redaction by default, and an audit log of explanations and alerts.

Sources: https://opencv.org/opencv-face-detection-cascade-classifier-vs-yunet/ ; https://arxiv.org/pdf/2111.15129 ; https://ceur-ws.org/Vol-3702/paper17.pdf ; https://www.prsindia.org/files/bills_acts/bills_parliament/2023/Summary_Digital_Personal_Data_Protection_Bill_2023.pdf ; https://www.hoganlovells.com/en/publications/indias-digital-personal-data-protection-act-2023-brought-into-force-

---

## Appendix A: Default parameter sheet (start values to tune)

| Component | Parameter | Start value | Basis |
|---|---|---|---|
| EigenCAM | layer | YOLOv8 layer 15 (P3) for smoke/fire, 21 (P5) for person | [derived] |
| EigenCAM | solver | eigh(XᵀX), or 15 power iterations; sign flip; ReLU; per-box renormalisation | [src + bench] |
| Deletion/insertion | substrate | zeros (deletion), Gaussian blur k = 11, σ = 5 (insertion); step 1% of pixels | [src RISE code] |
| Pointing game | tolerance | 15 px | [src] |
| Fall | torso angle | > 45° from vertical (strong > 60°) | [src 45°] |
| Fall | descent | v_norm ≥ 0.6 body-heights/s over 0.4 s | [derived] |
| Fall | aspect | W/H ≥ 1.0 | [src] |
| Fall | dwell | ≥ 2–3 s horizontal + low motion; cooldown 2 min | [src] |
| Smoke | dark channel | 15×15 patch, ω = 0.95, A = top 0.1%, t₀ = 0.1 | [src] |
| Smoke | grey-ness | S < 0.28 & V > 108, or RGB spread ≤ 15–20 | [src] |
| Crowd | flow | Farneback 160×120, (0.5, 3, 15, 3, 5, 1.2); K = 8 direction bins | [bench + derived] |
| Fusion | model | logistic / NB log-odds, monotone piecewise g_i; φ_i = g_i(x_i) − E[g_i] | [src] |
| Smoothing | EMA τ | 1 s fall, 3–5 s smoke, 1–2 s crowd; hysteresis k_on = 3–5, k_off = 15 | [derived] |
| Calibration | method | Platt (n < 1,000), isotonic (n > 1,000); ECE with M = 15 | [src] |
| Threshold | conformal | λ̂ = ⌈(n+1)(1−α)⌉-th order statistic of normal-segment max scores | [src] |
| ORT | session | intra_op = 2, sequential, allow_spinning = 0, S8S8 QDQ static, head FP32, per-channel | [src] |
| Input size | detector | 320–416 px (indoor), 640 ROI re-check | [derived] |
| NLG | gate | fully-grounded rate = 100% required, otherwise template fallback | [derived] |

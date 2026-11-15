# Retinal Disease Detection Using Deep Learning on OCT Images
## Comprehensive Technical & Clinical Project Documentation

---

### Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Clinical Background & Pathology Overview](#2-clinical-background--pathology-overview)
3. [Dataset Architecture & Exploration](#3-dataset-architecture--exploration)
4. [Data Preprocessing & Augmentation Pipeline](#4-data-preprocessing--augmentation-pipeline)
5. [Neural Network Fundamentals (First Principles in NumPy)](#5-neural-network-fundamentals-first-principles-in-numpy)
6. [Deep Learning Model Architectures](#6-deep-learning-model-architectures)
   - Custom Convolutional Neural Network (Scratch)
   - ResNet18 (Residual Learning)
   - MobileNetV2 (Edge-Optimized Inverted Residuals)
   - DenseNet121 (Dense Feature Reuse)
   - VGG16 (Deep Convolutional Baseline)
7. [Optimization, Regularization & Training Pipeline](#7-optimization-regularization--training-pipeline)
8. [Comprehensive Evaluation Metrics & Diagnostic Benchmarks](#8-comprehensive-evaluation-metrics--diagnostic-benchmarks)
9. [Model Explainability with Grad-CAM](#9-model-explainability-with-grad-cam)
10. [Model Serialization to PKL Format & Pipeline Architecture](#10-model-serialization-to-pkl-format--pipeline-architecture)
11. [Streamlit Web Application User Guide](#11-streamlit-web-application-user-guide)
12. [Clinical Limitations, Ethical Considerations & Future Scope](#12-clinical-limitations-ethical-considerations--future-scope)

---

### 1. Executive Summary & Problem Statement
Optical Coherence Tomography (OCT) is the preeminent non-invasive imaging modality in modern ophthalmology, utilizing low-coherence interferometry to generate micrometer-resolution cross-sectional tomograms (B-scans) of the neurosensory retina, retinal pigment epithelium (RPE), and choroid. 

However, the sheer volume of scans acquired in clinical workflows presents severe challenges:
- High clinical cognitive load and risk of diagnostic fatigue among retinal specialists.
- Shortage of ophthalmologists globally, especially in developing and remote rural regions.
- Irreversible visual loss resulting from delayed detection of urgent conditions like Choroidal Neovascularization (Wet AMD) and Diabetic Macular Edema (DME).

**Project Objective:**
To engineer an intelligent, automated, and explainable deep learning classification framework capable of differentiating between **Choroidal Neovascularization (CNV)**, **Diabetic Macular Edema (DME)**, **Drusen**, and **Normal** retinal scans, exporting production-ready models serialized in `.pkl` format, and delivering an intuitive Streamlit interface equipped with real-time Grad-CAM biomarker localization.

---

### 2. Clinical Background & Pathology Overview

| Condition | Category | OCT Pathognomonic Biomarkers | Clinical Risk & Treatment |
| :--- | :--- | :--- | :--- |
| **CNV** | Choroidal Neovascularization | Disruption of RPE layer, subretinal fluid (SRF) accumulation, hyperreflective fibrovascular membranes. | Leading cause of severe, rapid central vision loss in wet AMD. Requires urgent intravitreal anti-VEGF therapy. |
| **DME** | Diabetic Macular Edema | Hyporeflective intraretinal cystoid fluid spaces, retinal thickening, loss of foveal depression. | Primary cause of moderate-to-severe visual impairment in working-age diabetic patients. Managed with anti-VEGF, focal laser, and glycemic control. |
| **DRUSEN** | Drusen Deposits | Convex dome-shaped or undulating elevations of the hyperreflective RPE layer over basal laminar deposits. | Key hallmark of dry AMD. Monitored for progression to geographic atrophy or conversion to wet AMD. |
| **NORMAL** | Normal Retina | Preserved foveal pit contour, intact layered architecture (RNFL, GCL, IPL, INL, OPL, ONL, ELM, EZ, RPE), zero fluid. | Healthy baseline retina with preserved phototransduction strata. |

---

### 3. Dataset Architecture & Exploration
The project utilizes the verified clinical Kermany2017 OCT dataset (obtained via Hugging Face: `zacharielegault/Kermany2017-OCT`):
- **Image Resolution**: High-resolution cross-sectional B-scans (~1024 × 496 pixels, 8-bit grayscale).
- **Target Classes**: 4 mutually exclusive categories: `CNV`, `DME`, `DRUSEN`, and `NORMAL`.
- **Partitioning Strategy**: Stratified shuffle splitting into:
  - **Training Split (70%)**: 700 scans
  - **Validation Split (15%)**: 150 scans
  - **Testing Split (15%)**: 150 scans
- **Stratification Integrity**: Guaranteed identical class proportions across all splits, preventing class distribution shift between optimization and testing.

---

### 4. Data Preprocessing & Augmentation Pipeline
Raw clinical scans undergo an end-to-end tensor transformation pipeline:
1. **Spatial Resampling**: Bilinear interpolation resizing to a uniform `(224, 224)` spatial dimension, matching standard convolutional receptive fields.
2. **Channel Dimension Expansion**: Converting single-channel luminance arrays into 3-channel tensors `(3, 224, 224)` for transfer learning compatibility.
3. **Data Augmentation (Training Only)**:
   - *Random Horizontal Flip* ($p=0.5$): Simulates left vs. right eye (OD vs. OS) scan orientations.
   - *Random Rotation* ($\pm 10^\circ$): Accounts for patient head tilt and scan angle variations.
   - *Random Affine Scaling* ($0.95 - 1.05$): Accounts for ocular axial length differences and optical magnification variance.
   - *Color Jitter* (Brightness & Contrast $\pm 15\%$): Simulates laser signal attenuation and media opacities (e.g., cataracts).
4. **Standardization & Normalization**: Standard ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$) scaling.

---

### 5. Neural Network Fundamentals (First Principles in NumPy)
To rigorously demonstrate neural network foundations, a 2-layer Multi-Layer Perceptron (MLP) was implemented from scratch using only NumPy (no PyTorch autograd):
- **Mathematical Forward Pass**:
  $$Z_1 = X W_1 + b_1$$
  $$A_1 = f(Z_1)$$
  $$Z_2 = A_1 W_2 + b_2$$
  $$P = \text{Softmax}(Z_2)$$
- **Analytical Gradient Backpropagation**:
  $$\frac{\partial \mathcal{L}}{\partial Z_2} = P - Y$$
  $$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \frac{\partial \mathcal{L}}{\partial Z_2} + \lambda W_2$$
  $$\frac{\partial \mathcal{L}}{\partial Z_1} = \left(\frac{\partial \mathcal{L}}{\partial Z_2} W_2^T\right) \odot f'(Z_1)$$
  $$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \frac{\partial \mathcal{L}}{\partial Z_1} + \lambda W_1$$
- **Activation Function Empirical Benchmark**:
  - **ReLU**: Loss = $0.7656$, Accuracy = **$77.2\%$** (fastest convergence, no gradient vanishing).
  - **Tanh**: Loss = $1.0828$, Accuracy = **$56.2\%$** (zero-centered, moderate saturation).
  - **Sigmoid**: Loss = $1.2780$, Accuracy = **$44.2\%$** (severe gradient vanishing in deeper regimes).

---

### 6. Deep Learning Model Architectures
Five distinct convolutional neural network architectures were developed, optimized, and evaluated:

#### 1. Custom OCT CNN (Trained from Scratch)
- **Stage 1**: `Conv2d(3, 32, 3)` + `BatchNorm2d(32)` + `ReLU` + `MaxPool2d(2, 2)`
- **Stage 2**: `Conv2d(32, 64, 3)` + `BatchNorm2d(64)` + `ReLU` + `MaxPool2d(2, 2)`
- **Stage 3**: `Conv2d(64, 128, 3)` + `BatchNorm2d(128)` + `ReLU` + `MaxPool2d(2, 2)`
- **Stage 4**: `Conv2d(128, 256, 3)` + `BatchNorm2d(256)` + `ReLU` + `AdaptiveAvgPool2d((4, 4))`
- **Classification Head**: `Flatten` $\rightarrow$ `Linear(4096, 256)` + `BatchNorm1d` + `ReLU` + `Dropout(0.4)` $\rightarrow$ `Linear(256, 64)` + `ReLU` + `Dropout(0.3)` $\rightarrow$ `Linear(64, 4)`

#### 2. ResNet18 (Residual Learning)
- Incorporates residual skip connections ($F(x) + x$) to eliminate the degradation problem in deep networks.
- Replaced terminal fully connected layer with `Dropout(0.3)` + `Linear(512, 4)`.

#### 3. MobileNetV2 (Inverted Residuals & Linear Bottlenecks)
- Depthwise separable convolutions significantly reduce floating-point operations (FLOPs).
- Replaced classifier with `Dropout(0.3)` + `Linear(1280, 4)`. Optimal for edge clinical devices.

#### 4. DenseNet121 (Dense Feature Reuse)
- Connects all layers directly with matching feature-map sizes, promoting multi-scale feature concatenation and direct gradient feedback.
- Replaced classifier with `Dropout(0.3)` + `Linear(1024, 4)`.

#### 5. VGG16 (Deep Classical Baseline)
- Classical deep architecture utilizing uniform $3 \times 3$ convolutional filters.
- Replaced terminal classifier with `Dropout(0.3)` + `Linear(4096, 4)`.

---

### 7. Optimization, Regularization & Training Pipeline
- **Loss Criterion**: Categorical Cross-Entropy Loss with numerically stable LogSoftmax computation.
- **Optimizer**: AdamW with decoupled weight decay ($L_2 = 1 \times 10^{-4}$) to control parameter norm explosion.
- **Learning Rate Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=2)` to adaptively decay step size upon validation loss stagnation.
- **Dropout Regularization**: $30\% - 40\%$ dropout across fully connected layers to eliminate co-adaptation of features.
- **Early Stopping**: Monitored validation loss with a patience threshold of 3 epochs; automatically restores historical optimal checkpoint weights.

---

### 8. Comprehensive Evaluation Metrics & Diagnostic Benchmarks
All models are benchmarked across the unseen test cohort (150 images) on:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (Macro & Weighted)**: $\frac{TP}{TP + FP}$
- **Recall / Sensitivity (Macro & Weighted)**: $\frac{TP}{TP + FN}$
- **F1-Score (Macro & Weighted)**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Confusion Matrix Analysis**: Quantifies exact class-to-class confusion pairs (e.g., subtle drusen misclassified as normal or early DME cysts).

All metrics and generated plots are automatically saved in `outputs/` and serialized to `models/evaluation_metrics.pkl`.

#### Experimental Benchmark Results (150 Unseen Test OCT Scans)
| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Misclassified (/150) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **VGG16** *(Selected Best)* | **81.33%** | **0.837** | **0.814** | **0.810** | **28 / 150** | `models/best_retinal_oct_model.pkl` |
| **DenseNet121** | **78.67%** | **0.828** | **0.787** | **0.783** | **32 / 150** | `models/densenet121.pkl` |
| **MobileNetV2** | **76.00%** | **0.784** | **0.761** | **0.757** | **36 / 150** | `models/mobilenet_v2.pkl` |
| **ResNet18** | **72.00%** | **0.724** | **0.720** | **0.720** | **42 / 150** | `models/resnet18.pkl` |
| **Custom OCT CNN** | **65.33%** | **0.687** | **0.656** | **0.622** | **52 / 150** | `models/custom_cnn.pkl` |

---

### 9. Model Explainability with Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual verification for medical practitioners:
1. Computes the gradient of the predicted class score $y^c$ with respect to the feature activation map $A^k$ of the final convolutional layer:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. Computes the weighted combination:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
3. Upsamples and projects a thermal JET colormap over the original grayscale OCT B-scan.

**Clinical Alignment:**
- **CNV**: Heatmap focal points converge on disrupted RPE subretinal fluid pockets.
- **DME**: Heatmap highlights sponge-like intraretinal hyporeflective cystic cavities.
- **Drusen**: Heatmap isolates undulating hyperreflective RPE elevations.
- **Normal**: Uniformly dispersed activations along the foveal pit without pathological hotspots.

---

### 10. Model Serialization to PKL Format & Pipeline Architecture
In strict fulfillment of the request (**"save output ml to pkl format"**):
- **Primary Deliverable**: `models/best_retinal_oct_model.pkl`
  Contains:
  - `model`: Trained PyTorch neural network instance
  - `state_dict`: Checkpoint tensor weights
  - `model_name`: Best architecture label (e.g. `resnet18`)
  - `metrics`: Test accuracy, precision, recall, and F1-score
  - `class_names`: `['CNV', 'DME', 'DRUSEN', 'NORMAL']`
  - `input_size`: `(3, 224, 224)`
  - `benchmark_table`: Full cross-model comparison dictionary
- **Evaluation Package**: `models/evaluation_metrics.pkl`
  Contains all test confusion matrices, training histories, classification reports, and misclassification records.
- **Dual Checkpoint**: Also exports `.pt` weights for standard PyTorch loading.

---

### 11. Streamlit Web Application User Guide
Launch the application with:
```bash
streamlit run app.py
```
**Application Features:**
1. **Model Switcher**: Select between Best Model (PKL), ResNet18, MobileNetV2, DenseNet121, Custom CNN, or VGG16.
2. **Scan Source**: Upload custom OCT B-scans or choose from verified pre-loaded clinical samples (CNV, DME, Drusen, Normal).
3. **Diagnostic Readout**: Displays predicted condition badge, confidence percentage, and full probability distribution.
4. **Live Grad-CAM**: Toggle real-time thermal overlay with biomarker localization explanations.
5. **Clinical Knowledge Base**: Embedded reference guide for pathophysiology and clinical interventions.
6. **Benchmark Tab**: Interactive display of model comparisons, training curves, and confusion matrices.
7. **Clinical Prototype Disclaimer**: Prominently displayed advisory alerting users to academic prototype scope.

---

### 12. Clinical Limitations, Ethical Considerations & Future Scope
- **Ethical Boundary**: This deep learning software is an educational research prototype. It is NOT cleared as a Software as a Medical Device (SaMD) and must not be used for direct clinical diagnosis.
- **Edge Cases**: Image quality degradation from dense cataracts or severe vitreous floaters can introduce noise into the B-scan.
- **Future Work**:
  1. Expansion to 3D volumetric macular cube processing.
  2. Multi-modal fusion with Color Fundus Photography.
  3. Semantic segmentation (U-Net) for exact millimeter fluid volume quantification.

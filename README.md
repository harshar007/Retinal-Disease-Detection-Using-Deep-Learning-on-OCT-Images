<div align="center">

# 👁️ Retinal Disease Detection Using Deep Learning on OCT Images
### 🔬 End-to-End Computer Vision & Explainable AI for Ophthalmic Optical Coherence Tomography (OCT)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](DOCKER.md)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Accuracy](https://img.shields.io/badge/Top%20Accuracy-81.33%25-00C853?style=for-the-badge&logo=target&logoColor=white)](#-comprehensive-model-benchmarks)
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-FF007F?style=for-the-badge&logo=probot&logoColor=white)](#-explainable-ai--grad-cam-biomarker-localization)
[![Model Format](https://img.shields.io/badge/Export-PKL%20Format-7928CA?style=for-the-badge&logo=dropbox&logoColor=white)](#-production-model-serialization-pkl)
[![License](https://img.shields.io/badge/License-MIT-4361EE?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)

<br/>

**[🚀 Quick Start](#-quick-start)** • 
**[💻 Live Streamlit App](#-interactive-streamlit-diagnostic-suite)** • 
**[🔬 Grad-CAM XAI](#-explainable-ai--grad-cam-biomarker-localization)** • 
**[📊 Benchmarks](#-comprehensive-model-benchmarks)** • 
**[🩺 Pathology Guide](#-clinical-pathology--biomarkers)** • 
**[🐳 Docker](#-docker-container-deployment)** • 
**[📖 Full Technical Docs](DOCUMENTATION.md)**

---

### 🌟 *Bridging Deep Convolutional Networks, Explainable AI, and Clinical Decision Support*
An intelligent, clinically informed computer vision framework engineered for the automated multi-class diagnosis of retinal abnormalities from **Optical Coherence Tomography (OCT)** cross-sectional B-scans with real-time **Grad-CAM visual biomarker localization**.

</div>

<br/>

---

## ⚡ Core Diagnostic Metrics & Deliverables

<div align="center">

| 🏆 Top Test Accuracy | 👁️ Diagnosed Classes | 🧠 Neural Architectures | 🔬 Explainability Engine | 💾 Serialization Standard |
| :---: | :---: | :---: | :---: | :---: |
| <b style="font-size: 1.4rem; color: #00C853;">81.33%</b><br/><sub>VGG16 Transfer Benchmark</sub> | <b style="font-size: 1.4rem; color: #1E88E5;">4 Conditions</b><br/><sub>CNV • DME • DRUSEN • NORMAL</sub> | <b style="font-size: 1.4rem; color: #7928CA;">5 Networks</b><br/><sub>NumPy MLP + CNN + 4 Pretrained</sub> | <b style="font-size: 1.4rem; color: #FF007F;">Grad-CAM</b><br/><sub>Anatomical Heatmap Localization</sub> | <b style="font-size: 1.4rem; color: #F59E0B;">.PKL Standard</b><br/><sub>Model + State Dict + Metrics</sub> |

</div>

<br/>

> [!IMPORTANT]
> **Mandatory Clinical Prototype Notice**: This software is an **academic and research prototype** built to advance medical computer vision and explainable AI. It is **NOT** a certified medical diagnostic device (FDA/CE-cleared SaMD). It is not intended for primary clinical diagnosis or autonomous patient treatment decisions.

---

## 📑 Table of Contents

- [🌟 Executive Overview](#-executive-overview)
- [🩺 Clinical Pathology & Biomarkers](#-clinical-pathology--biomarkers)
- [🏗️ End-to-End Architecture & Pipeline](#️-end-to-end-architecture--pipeline)
- [📊 Comprehensive Model Benchmarks](#-comprehensive-model-benchmarks)
- [🔬 Explainable AI & Grad-CAM Biomarker Localization](#-explainable-ai--grad-cam-biomarker-localization)
- [🧮 Neural Network Fundamentals (Pure NumPy MLP)](#-neural-network-fundamentals-pure-numpy-mlp)
- [💻 Interactive Streamlit Diagnostic Suite](#-interactive-streamlit-diagnostic-suite)
- [💾 Production Model Serialization (.pkl)](#-production-model-serialization-pkl)
- [🚀 Quick Start & Installation](#-quick-start)
- [🐳 Docker Container Deployment](#-docker-container-deployment)
- [📂 Repository Architecture](#-repository-architecture)
- [👥 Contributing & License](#-contributing--license)

---

## 🌟 Executive Overview

**Optical Coherence Tomography (OCT)** is the non-invasive gold standard in modern ophthalmology, capturing micrometer-scale cross-sectional cross-cuts (B-scans) of the neurosensory retina, retinal pigment epithelium (RPE), and choroid. 

<div align="center">
<img src="outputs/class_distribution.png" width="85%" alt="OCT Class Distribution" />
<p><i>Figure 1: Stratified split distribution across Training (70%), Validation (15%), and Test (15%) cohorts.</i></p>
</div>

While OCT scans are routinely ordered in millions worldwide, their interpretation requires specialized retinal subspecialists who face cognitive fatigue and severe diagnostic backlogs. This repository provides:
- 🩺 **Automated 4-Class Diagnostic Screening**: Differentiating **Choroidal Neovascularization (CNV)**, **Diabetic Macular Edema (DME)**, **Drusen**, and **Normal** retina.
- 🔬 **Clinical Explainability (Grad-CAM)**: Visual attention heatmaps pinpointing exact pathological biomarkers (intraretinal cysts, subretinal fluid, RPE drusen elevations).
- 🧠 **Multi-Architecture Comparative Study**: Benchmarking a custom scratch CNN against **ResNet18**, **MobileNetV2**, **DenseNet121**, and **VGG16**.
- 🧮 **NumPy First-Principles Grounding**: Analytical backpropagation implementation with comparative empirical benchmark of **ReLU**, **Sigmoid**, and **Tanh** activations.
- 💾 **Ready-to-Deploy Serialization**: Pickled model packages (`.pkl`) bundling trained weights, preprocessing transformations, and evaluation metrics for immediate deployment.
- 🖥️ **Full-Featured Interactive Web App**: A responsive Streamlit dashboard featuring multi-model inference, sample galleries, and live Grad-CAM inspection.

---

## 🩺 Clinical Pathology & Biomarkers

<div align="center">

| Condition | Diagnostic Status | OCT Biomarkers & Morphological Hallmarks | Clinical Risk & Treatment Protocol |
| :--- | :---: | :--- | :--- |
| **CNV**<br/><sub>Choroidal Neovascularization</sub> | <span style="background-color: #FFEBEE; color: #C62828; padding: 4px 10px; border-radius: 12px; font-weight: bold;">🔴 Urgent</span> | Disruption of the RPE layer, subretinal / sub-RPE fluid accumulation, and hyperreflective fibrovascular membranes. | Leading cause of rapid, severe central vision loss in wet AMD. Requires urgent intravitreal anti-VEGF injection therapy. |
| **DME**<br/><sub>Diabetic Macular Edema</sub> | <span style="background-color: #FFF3E0; color: #E65100; padding: 4px 10px; border-radius: 12px; font-weight: bold;">🟠 Priority</span> | Hyporeflective intraretinal cystoid fluid spaces, widespread neurosensory retinal thickening, loss of foveal depression. | Primary cause of legal blindness in working-age diabetic populations. Managed via anti-VEGF, focal laser, and systemic glycemic control. |
| **DRUSEN**<br/><sub>Basal Laminar Deposits</sub> | <span style="background-color: #EDE7F6; color: #4A148C; padding: 4px 10px; border-radius: 12px; font-weight: bold;">🟣 Monitor</span> | Convex dome-shaped or undulating elevations of the hyperreflective RPE layer over acellular lipid deposits. | Hallmark biomarker of early/intermediate dry AMD. Monitored regularly to detect progression to geographic atrophy or conversion to wet AMD. |
| **NORMAL**<br/><sub>Healthy Macula</sub> | <span style="background-color: #E8F5E9; color: #2E7D32; padding: 4px 10px; border-radius: 12px; font-weight: bold;">🟢 Healthy</span> | Preserved symmetric foveal pit contour, intact retinal layers (RNFL, GCL, IPL, INL, OPL, ONL, ELM, EZ, RPE), zero fluid. | Normal baseline ocular architecture with preserved photoreceptor integrity and no clinical intervention required. |

</div>

<br/>

### 🖼️ Real Clinical OCT B-Scan Gallery

<div align="center">

| 🔴 **CNV Sample** | 🟠 **DME Sample** | 🟣 **DRUSEN Sample** | 🟢 **NORMAL Sample** |
| :---: | :---: | :---: | :---: |
| <img src="assets/sample_images/CNV_sample_1.jpeg" width="220" alt="CNV OCT Sample" /> | <img src="assets/sample_images/DME_sample_1.jpeg" width="220" alt="DME OCT Sample" /> | <img src="assets/sample_images/DRUSEN_sample_1.jpeg" width="220" alt="Drusen OCT Sample" /> | <img src="assets/sample_images/NORMAL_sample_1.jpeg" width="220" alt="Normal OCT Sample" /> |
| *Subretinal Fluid & Neovascular Membrane* | *Intraretinal Cystoid Fluid Cavities* | *Sub-RPE Nodular Elevations* | *Intact Foveal Pit & Layer Stratification* |

</div>

---

## 🏗️ End-to-End Architecture & Pipeline

```mermaid
flowchart TD
    %% Styling Classes
    classDef inputStyle fill:#1E88E5,stroke:#0D47A1,stroke-width:2px,color:#fff;
    classDef prepStyle fill:#7E57C2,stroke:#4527A0,stroke-width:2px,color:#fff;
    classDef modelStyle fill:#00897B,stroke:#004D40,stroke-width:2px,color:#fff;
    classDef optStyle fill:#F57C00,stroke:#E65100,stroke-width:2px,color:#fff;
    classDef xaiStyle fill:#D81B60,stroke:#880E4F,stroke-width:2px,color:#fff;
    classDef deployStyle fill:#43A047,stroke:#1B5E20,stroke-width:2px,color:#fff;

    subgraph INP ["1. Data Ingestion & Stratification"]
        A["Raw Clinical OCT B-Scans<br/>(Kermany2017 Dataset)"]:::inputStyle --> B["Stratified 70/15/15 Split<br/>Train: 700 | Val: 150 | Test: 150"]:::inputStyle
    end

    subgraph PREP ["2. Preprocessing & Augmentation Engine"]
        B --> C1["Spatial Resampling<br/>(224 x 224 Bilinear)"]:::prepStyle
        C1 --> C2["3-Channel Conversion &<br/>ImageNet Normalization"]:::prepStyle
        C2 --> C3["Data Augmentation<br/>(Flips, Scalings, Rotations, Jitter)"]:::prepStyle
    end

    subgraph MOD ["3. Deep Learning Architecture Zoo"]
        C3 --> M1["NumPy 2-Layer MLP<br/>(First-Principles Benchmark)"]:::modelStyle
        C3 --> M2["Custom 4-Stage CNN<br/>(Scratch Baseline)"]:::modelStyle
        C3 --> M3["ResNet18<br/>(Residual Learning)"]:::modelStyle
        C3 --> M4["MobileNetV2<br/>(Edge Inverted Bottlenecks)"]:::modelStyle
        C3 --> M5["DenseNet121<br/>(Dense Feature Concatenation)"]:::modelStyle
        C3 --> M6["VGG16 Transfer<br/>(Deep 3x3 Convolutions)"]:::modelStyle
    end

    subgraph OPT ["4. Training & Optimization Regime"]
        M2 & M3 & M4 & M5 & M6 --> O1["AdamW Optimizer<br/>(L2 Weight Decay = 1e-4)"]:::optStyle
        O1 --> O2["Categorical Cross-Entropy Loss"]:::optStyle
        O2 --> O3["ReduceLROnPlateau Scheduler<br/>(factor=0.5, patience=2)"]:::optStyle
        O3 --> O4["Early Stopping & Weight Checkpointing"]:::optStyle
    end

    subgraph EVAL ["5. Diagnostic Inference & Explainability"]
        O4 --> E1["Softmax Diagnostic Probabilities<br/>[CNV, DME, DRUSEN, NORMAL]"]:::xaiStyle
        O4 --> E2["Grad-CAM Gradient Engine<br/>Target Feature Layer Attribution"]:::xaiStyle
        E2 --> E3["Visual Heatmap Fusion &<br/>Biomarker Overlay"]:::xaiStyle
    end

    subgraph OUT ["6. Production Delivery & Packaging"]
        E1 & E3 --> D1["Interactive Streamlit Web Dashboard<br/>(app.py)"]:::deployStyle
        E1 & E3 --> D2["Serialized Model Packages (.pkl)<br/>(models/best_retinal_oct_model.pkl)"]:::deployStyle
        D1 --> D3["Dockerized CPU/GPU Deployment<br/>(docker-compose.yml)"]:::deployStyle
    end
```

---

## 📊 Comprehensive Model Benchmarks

Each architecture was rigorously evaluated on the **independent, unseen test cohort** ($N = 150$ scans) across standard multi-class clinical diagnostic metrics:

<div align="center">

| Rank | Architecture | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Misclassified (/150) | Key Architectural Characteristics | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 🥇 | **VGG16** *(Best Overall)* | <b style="color: #00C853;">81.33%</b> | **0.837** | **0.814** | **0.810** | **28 / 150** | Uniform $3 \times 3$ receptive fields; exceptional sensitivity for subtle cystic fluid spaces. | 🟢 **Primary** |
| 🥈 | **DenseNet121** | **78.67%** | **0.828** | **0.787** | **0.783** | **32 / 150** | Dense feature concatenation; robust preservation of low-level RPE edge transitions. | 🟢 Ready |
| 🥉 | **MobileNetV2** | **76.00%** | **0.784** | **0.761** | **0.757** | **36 / 150** | Depthwise separable convolutions; ultra-lightweight footprint optimal for edge devices. | 🟢 Ready |
| 4 | **ResNet18** | **72.00%** | **0.724** | **0.720** | **0.720** | **42 / 150** | Residual shortcut connections; rapid convergence and steady gradient propagation. | 🟢 Ready |
| 5 | **Custom CNN** *(Scratch)* | **65.33%** | **0.687** | **0.656** | **0.622** | **52 / 150** | 4-stage convolutional design trained from scratch without ImageNet priors. | 🟢 Ready |

</div>

<br/>

### 📈 Diagnostic Metric Comparisons & Training Curves

<div align="center">

| 📊 Multi-Model Metric Benchmark Comparison | 📉 Training & Validation Dynamics |
| :---: | :---: |
| <img src="outputs/model_comparison_metrics.png" width="480" alt="Model Comparison Metrics" /> | <img src="outputs/training_curves.png" width="480" alt="Training Curves" /> |
| *Grouped comparison of Accuracy, Precision, Recall, and F1 across all 5 models.* | *Epoch-by-epoch loss reduction and validation accuracy tracking.* |

</div>

<br/>

### 🎯 Multi-Model Confusion Matrix Grid

<div align="center">
<img src="outputs/confusion_matrices.png" width="92%" alt="Confusion Matrices" />
<p><i>Figure 2: Comprehensive 5-model test cohort confusion matrices illustrating true vs. predicted disease classifications.</i></p>
</div>

---

## 🔬 Explainable AI & Grad-CAM Biomarker Localization

In clinical ophthalmology, black-box predictions are unacceptable. **Grad-CAM (Gradient-weighted Class Activation Mapping)** provides visual evidence by calculating the gradients of the target class score $y^c$ with respect to the activation feature maps $A^k$ of the terminal convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

The computed heatmaps are upsampled and fused onto the original grayscale OCT B-scan, highlighting the precise pathological regions influencing the prediction.

<div align="center">
<img src="outputs/gradcam_examples.png" width="95%" alt="Grad-CAM Explainability Gallery" />
<p><i>Figure 3: 4x3 Explainability Gallery displaying Raw OCT Scans (left), Diagnostic Grad-CAM Heatmaps (center), and Biomarker Fusion Overlays (right).</i></p>
</div>

> [!TIP]
> **Clinical Biomarker Correlation**: Notice how the model focuses attention directly on:
> - **CNV**: Hyperreflective membranes and subretinal fluid breaches in the RPE layer.
> - **DME**: Hyporeflective intraretinal cystoid fluid bubbles and neurosensory thickening.
> - **DRUSEN**: Basal undulating deposits elevating the hyperreflective retinal pigment epithelium.
> - **NORMAL**: The intact, symmetrical foveal depression and stratified parallel retinal layers.

---

## 🧮 Neural Network Fundamentals (Pure NumPy MLP)

To establish first-principles rigor, a complete 2-layer Multi-Layer Perceptron (MLP) was developed strictly in **pure NumPy** with zero framework dependencies (no PyTorch, no autograd).

### 📐 Mathematical Formulation

1. **Forward Propagation**:
   $$Z_1 = X W_1 + b_1, \quad A_1 = f(Z_1)$$
   $$Z_2 = A_1 W_2 + b_2, \quad \hat{Y} = \text{Softmax}(Z_2)$$

2. **Categorical Cross-Entropy Loss with $L_2$ Regularization**:
   $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C Y_{i,c} \log(\hat{Y}_{i,c}) + \frac{\lambda}{2N} \left(\|W_1\|_F^2 + \|W_2\|_F^2\right)$$

3. **Analytical Backpropagation (Exact Matrix Gradients)**:
   $$\frac{\partial \mathcal{L}}{\partial Z_2} = \frac{1}{N}(\hat{Y} - Y)$$
   $$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \frac{\partial \mathcal{L}}{\partial Z_2} + \frac{\lambda}{N} W_2, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum \frac{\partial \mathcal{L}}{\partial Z_2}$$
   $$\frac{\partial \mathcal{L}}{\partial Z_1} = \left(\frac{\partial \mathcal{L}}{\partial Z_2} W_2^T\right) \odot f'(Z_1)$$
   $$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \frac{\partial \mathcal{L}}{\partial Z_1} + \frac{\lambda}{N} W_1, \quad \frac{\partial \mathcal{L}}{\partial b_1} = \sum \frac{\partial \mathcal{L}}{\partial Z_1}$$

<br/>

### 🧪 Activation Function Empirical Benchmark

<div align="center">

| Activation Function | Non-Linearity Form $f(z)$ | Gradient Form $f'(z)$ | Test Loss | Test Accuracy | Convergence & Behavioral Dynamics |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **ReLU** | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ | <b style="color: #00C853;">0.7656</b> | <b style="color: #00C853;">77.20%</b> | 🟢 **Superior Convergence**: Constant gradient prevents vanishing; fastest optimization. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - f(z)^2$ | **1.0828** | **56.20%** | 🟡 **Moderate Saturation**: Zero-centered, but suffers from asymptotic gradient dampening. |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $f(z)(1 - f(z))$ | **1.2780** | **44.20%** | 🔴 **Severe Vanishing Gradient**: Maximum gradient bounded at 0.25; slow backpropagation. |

</div>

---

## 💻 Interactive Streamlit Diagnostic Suite

The interactive clinical web application (`app.py`) provides an intuitive decision-support environment for clinicians, researchers, and students:

<div align="center">

```
========================================================================================
 👁️ RETINAL OCT DISEASE AI DIAGNOSTIC PROTOTYPE (STREAMLIT SUITE)
========================================================================================
 [Sidebar: Model Selector]  │  [Main Panel: Diagnostic Workspace]
   • Best Model (VGG16)     │  
   • DenseNet121            │  [Upload Scan / Select Clinical Sample]
   • MobileNetV2            │   ├── Original OCT Image
   • ResNet18               │   ├── Diagnostic Prediction (e.g. DME, 98.4% Confidence)
   • Custom Scratch CNN     │   └── Interactive Softmax Probability Distribution Bar Chart
                            │
 [Grad-CAM Overlay Engine]  │  [Explainable AI Biomarker Heatmap]
   • Enable/Disable XAI     │   ├── Target Layer Selection & Grad-CAM Attention Map
   • Heatmap Alpha Slider   │   └── Fused Pathological Overlay with Structural Correlation
========================================================================================
```

</div>

### ✨ Core Application Features
- 🎛️ **Dynamic Multi-Model Switching**: Seamlessly toggle between all 5 trained deep architectures to contrast their diagnostic consensus.
- 📂 **Dual Acquisition Modes**: Upload any clinical OCT image (`.png`, `.jpg`, `.jpeg`) or choose from the built-in clinical pathology gallery (`CNV`, `DME`, `Drusen`, `Normal`).
- 📊 **Confidence & Softmax Distributions**: Real-time multi-class probability breakdown with intuitive risk color badges.
- 🔬 **Real-Time Grad-CAM Inspection**: Interactive heatmap overlays with configurable opacity sliders to interrogate anatomical layers.
- 📚 **Integrated Knowledge Base**: Immediate on-screen access to pathology descriptions, pathognomonic hallmarks, and recommended clinical next steps.

---

## 💾 Production Model Serialization (.pkl)

All trained models, checkpoints, and evaluation metrics are saved in standard Python `.pkl` format in `models/best_retinal_oct_model.pkl`:

```python
import pickle
import torch

# 1. Load the standalone production model package
with open("models/best_retinal_oct_model.pkl", "rb") as f:
    package = pickle.load(f)

# 2. Extract components
model = package["model"]               # Fully instantiated PyTorch architecture
state_dict = package["state_dict"]     # Optimized tensor weights
metrics = package["metrics"]           # Independent test set evaluation scores
class_names = package["class_names"]   # ['CNV', 'DME', 'DRUSEN', 'NORMAL']

model.load_state_dict(state_dict)
model.eval()

print(f"Loaded: {package['model_name']} | Test Accuracy: {metrics['Accuracy']*100:.2f}%")
```

---

## 🚀 Quick Start

### Option A: Local Python Environment

```bash
# 1. Clone the repository
git clone https://github.com/your-username/retinal-oct-detection.git
cd "retinal-oct-detection"

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full training and benchmarking pipeline
python run_pipeline.py

# 5. Launch the Streamlit web dashboard
streamlit run app.py
```

---

## 🐳 Docker Container Deployment

The repository includes a production-grade containerization setup supporting both CPU and NVIDIA GPU hardware.

### Quick Start with Docker Compose

```bash
# Clone and enter directory
cd "retinal-oct-detection"

# Start the application (CPU profile by default)
docker compose up -d --build
```
> Access the live application at: **`http://localhost:8501`**

### NVIDIA GPU Acceleration Profile

```bash
# Launch with NVIDIA GPU acceleration
docker compose -f docker-compose.gpu.yml up -d --build
```

### One-Click Helper Scripts
- **Windows PowerShell**: `.\docker-start.ps1`
- **Windows Command Prompt**: `docker-start.bat`
- **Linux / macOS Terminal**: `chmod +x docker-start.sh && ./docker-start.sh`

> For complete container configuration details, volume mounts, and troubleshooting, refer to [DOCKER.md](DOCKER.md).

---

## 📂 Repository Architecture

```
Retinal Disease Detection Using Deep Learning on OCT Images/
│
├── app.py                             # Interactive Streamlit Web Application
├── run_pipeline.py                    # End-to-end training & evaluation orchestrator
├── create_presentation.py             # Automated slide generation script
├── Retinal_Disease_Detection_OCT.pptx # 12-slide comprehensive project slide deck
├── Dockerfile                         # Production multi-stage container build
├── docker-compose.yml                 # Multi-platform container configuration
├── docker-compose.gpu.yml             # GPU-accelerated Docker Compose profile
├── docker-start.bat / .ps1 / .sh      # Cross-platform one-click launch scripts
├── DOCKER.md                          # Full Docker deployment & operational guide
├── DOCUMENTATION.md                   # In-depth clinical & technical whitepaper
├── README.md                          # Visual repository presentation
├── requirements.txt                   # Verified Python environment dependencies
│
├── assets/
│   └── sample_images/                 # Real clinical OCT B-scans (CNV, DME, Drusen, Normal)
│
├── data/                              # Local dataset cache and stratified split manifests
│
├── models/                            # Serialized model packages (.pkl) & checkpoints (.pt)
│   ├── best_retinal_oct_model.pkl     # Primary best model package (VGG16)
│   ├── best_retinal_oct_model.pt      # Best model PyTorch state_dict
│   ├── custom_cnn.pkl / .pt           # Scratch 4-stage CNN package
│   ├── resnet18.pkl / .pt             # ResNet18 transfer package
│   ├── mobilenet_v2.pkl / .pt         # MobileNetV2 edge package
│   ├── densenet121.pkl / .pt          # DenseNet121 dense feature package
│   ├── vgg16.pkl / .pt                # VGG16 deep transfer package
│   └── evaluation_metrics.pkl         # Pickled test metrics & confusion matrices
│
├── outputs/                           # Diagnostic evaluation visualizations & reports
│   ├── class_distribution.png         # Dataset split class distribution plot
│   ├── training_curves.png            # Loss and accuracy convergence curves
│   ├── confusion_matrices.png         # 5-model test cohort confusion matrix grid
│   ├── model_comparison_metrics.png   # Benchmark comparative bar chart
│   ├── gradcam_examples.png           # 4x3 Grad-CAM explainability gallery
│   └── model_benchmark_results.csv    # Benchmark metric evaluation CSV
│
└── src/                               # Modular engineering package
    ├── __init__.py                    # Module export definitions
    ├── dataset.py                     # Hugging Face loader, preprocessing & transforms
    ├── nn_fundamentals.py             # Pure NumPy MLP & activation benchmark
    ├── models.py                      # Custom CNN & Transfer Learning builders
    ├── train.py                       # Training loop, AdamW, Early Stopping, L2 reg
    ├── evaluate.py                    # Multi-class test evaluation & classification metrics
    ├── gradcam.py                     # Grad-CAM forward/backward hook implementation
    └── utils.py                       # PKL serialization & plotting utilities
```

---

## 👥 Contributing & License

Contributions, issues, and feature proposals are warmly welcomed! Please feel free to open a pull request or submit an issue.

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">

### 👁️ Retinal OCT Disease AI Diagnostic Framework
*Engineered with precision for explainable, trustworthy deep learning in clinical ophthalmology.*

[![Back to Top](https://img.shields.io/badge/Back%20to%20Top-000000?style=for-the-badge&logo=accenture&logoColor=white)](#-retinal-disease-detection-using-deep-learning-on-oct-images)

</div>

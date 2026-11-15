"""
Script to generate the professional Project Presentation (Retinal_Disease_Detection_OCT.pptx)
using python-pptx according to all requirements in the project specification.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE


def apply_background(slide, color_rgb):
    """Sets a solid background color for a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color_rgb


def add_header(slide, title_text, category_text="RETINAL DISEASE DETECTION USING DEEP LEARNING"):
    """Adds a standardized corporate medical header to a slide."""
    # Category / Breadcrumb
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = RGBColor(30, 136, 229)  # Medical Blue

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.7))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(26, 35, 126)  # Deep Navy


def add_card(slide, left, top, width, height, title, items, bg_color=RGBColor(245, 247, 250), border_color=RGBColor(200, 215, 235)):
    """Creates a structured rounded-box card with bullet points."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    tf.margin_top = Inches(0.15)
    tf.margin_bottom = Inches(0.15)

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(13, 71, 161)
    p0.space_after = Pt(8)

    for item in items:
        p = tf.add_paragraph()
        p.text = f"• {item}"
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(55, 71, 79)
        p.space_after = Pt(4)


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    apply_background(slide1, RGBColor(240, 244, 250))

    # Decorative accent bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.15), Inches(3.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor(30, 136, 229)
    bar.line.fill.background()

    tb = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(3.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p_sub0 = tf.paragraphs[0]
    p_sub0.text = "HEALTHCARE & LIFE SCIENCES — MEDICAL IMAGING (OPHTHALMOLOGY)"
    p_sub0.font.size = Pt(12)
    p_sub0.font.bold = True
    p_sub0.font.color.rgb = RGBColor(30, 136, 229)
    p_sub0.space_after = Pt(10)

    p_title = tf.add_paragraph()
    p_title.text = "Retinal Disease Detection Using Deep Learning on OCT Images"
    p_title.font.size = Pt(32)
    p_title.font.bold = True
    p_title.font.color.rgb = RGBColor(13, 71, 161)
    p_title.space_after = Pt(12)

    p_desc = tf.add_paragraph()
    p_desc.text = "Automated Diagnostic Classification (CNV, DME, Drusen, Normal) with PyTorch, Transfer Learning, Explainable AI (Grad-CAM), and Streamlit Application"
    p_desc.font.size = Pt(15)
    p_desc.font.color.rgb = RGBColor(69, 90, 100)
    p_desc.space_after = Pt(24)

    p_meta = tf.add_paragraph()
    p_meta.text = "Deliverable: Complete Python Pipeline | Saved PKL Models | Grad-CAM Visualizations | Streamlit App"
    p_meta.font.size = Pt(12)
    p_meta.font.bold = True
    p_meta.font.color.rgb = RGBColor(55, 71, 79)

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Clinical Urgency
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    apply_background(slide2, RGBColor(255, 255, 255))
    add_header(slide2, "Problem Statement & Clinical Motivation")

    add_card(slide2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "The Clinical Challenge", [
                 "Optical Coherence Tomography (OCT) is the gold-standard non-invasive modality producing micrometer-resolution retinal cross-sections.",
                 "Global shortage of trained ophthalmologists leads to severe diagnostic delays.",
                 "Conditions like Wet AMD (CNV) and Diabetic Macular Edema (DME) cause irreversible blindness if untreated within weeks.",
                 "Drusen deposits represent the earliest precursor to Age-related Macular Degeneration requiring continuous surveillance.",
                 "Manual review of hundreds of B-scans per patient is labor-intensive and prone to fatigue and subjective inter-observer variability."
             ])

    add_card(slide2, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Project Objectives & Proposed Solution", [
                 "Develop an intelligent end-to-end deep learning diagnostic system.",
                 "Accurately classify scans into 4 target conditions: CNV, DME, Drusen, and Normal.",
                 "Implement Custom CNN from scratch and 4 Pretrained Transfer Learning models (ResNet18, MobileNetV2, DenseNet121, VGG16).",
                 "Provide Explainable AI (Grad-CAM) to localize pathological retinal layer biomarkers for clinical transparency.",
                 "Deploy an interactive Streamlit application with model export to PKL format for clinical workflows."
             ])

    # -------------------------------------------------------------
    # SLIDE 3: Business & Healthcare Use Cases
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    apply_background(slide3, RGBColor(255, 255, 255))
    add_header(slide3, "Real-World Healthcare & Business Use Cases")

    add_card(slide3, Inches(0.8), Inches(1.6), Inches(3.6), Inches(2.5),
             "1. Hospitals & Eye Care Centers", [
                 "Streamline triage workflows.",
                 "Prioritize urgent CNV/DME scans for immediate specialist intervention.",
                 "Reduce clinical diagnostic backlog."
             ])

    add_card(slide3, Inches(4.8), Inches(1.6), Inches(3.6), Inches(2.5),
             "2. Teleophthalmology Platforms", [
                 "Remote retinal assessment.",
                 "Provide instant expert-level screening in rural and underserved clinics.",
                 "Decentralize ophthalmic care."
             ])

    add_card(slide3, Inches(8.8), Inches(1.6), Inches(3.6), Inches(2.5),
             "3. Population Screening", [
                 "Large-scale diabetic retinopathy screening.",
                 "Automated categorization of routine checkups.",
                 "Flag early drusen before severe vision loss."
             ])

    add_card(slide3, Inches(0.8), Inches(4.4), Inches(3.6), Inches(2.5),
             "4. Diagnostic Imaging Centers", [
                 "Automate preliminary image indexing.",
                 "Reduce manual review overhead.",
                 "Standardize cross-site diagnostic accuracy."
             ])

    add_card(slide3, Inches(4.8), Inches(4.4), Inches(3.6), Inches(2.5),
             "5. Clinical Decision Support", [
                 "Act as an objective second opinion.",
                 "Grad-CAM visual validation builds clinician trust and prevents misdiagnoses."
             ])

    add_card(slide3, Inches(8.8), Inches(4.4), Inches(3.6), Inches(2.5),
             "6. Health-Tech Software", [
                 "Embed modular PKL model pipelines into Electronic Health Record (EHR) platforms.",
                 "Edge-compatible deployment via MobileNetV2."
             ])

    # -------------------------------------------------------------
    # SLIDE 4: Dataset Architecture & Preprocessing Pipeline
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    apply_background(slide4, RGBColor(255, 255, 255))
    add_header(slide4, "OCT Dataset Architecture & Data Pipeline")

    add_card(slide4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Kermany2017 Retinal Dataset", [
                 "Source: Verified clinical cohort from Hugging Face / Mendeley Data.",
                 "Modality: High-resolution spectral-domain OCT B-scans.",
                 "4 Well-Defined Clinical Categories:",
                 "  • CNV: Choroidal Neovascularization (Wet AMD)",
                 "  • DME: Diabetic Macular Edema (Diabetic Retinopathy)",
                 "  • DRUSEN: Lipid/protein extracellular deposits (Dry AMD)",
                 "  • NORMAL: Intact retinal layers with preserved foveal pit",
                 "Balanced multi-class distribution ensuring zero class-imbalance bias.",
                 "Stratified partitioning into Train (70%), Validation (15%), and Test (15%)."
             ])

    add_card(slide4, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Preprocessing & Augmentation Pipeline", [
                 "Spatial Standardization: Resized to 224x224 pixels with bilinear interpolation.",
                 "Channel Formatting: Conversion of single-channel grayscale OCT to 3-channel RGB for ImageNet weight compatibility.",
                 "Pixel Normalization: ImageNet channel statistics (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]).",
                 "Data Augmentation (Train Split Only):",
                 "  • Random Horizontal Flip (p = 0.5)",
                 "  • Random Rotation (±10°) mimicking patient head tilt",
                 "  • Random Affine Scale (0.95 - 1.05) simulating zoom variance",
                 "  • Color Jitter (Brightness & Contrast ±15%) simulating scan intensity differences."
             ])

    # -------------------------------------------------------------
    # SLIDE 5: Neural Network Fundamentals (NumPy from Scratch)
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    apply_background(slide5, RGBColor(255, 255, 255))
    add_header(slide5, "Neural Network Fundamentals (First Principles in NumPy)")

    add_card(slide5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Architecture & Mathematical Implementation", [
                 "Implemented 2-layer Multi-Layer Perceptron (MLP) from scratch without PyTorch autograd.",
                 "Full analytical forward propagation: Z1 = X·W1 + b1, A1 = f(Z1), Z2 = A1·W2 + b2, P = Softmax(Z2).",
                 "Cross-Entropy Loss computation with numerical stability clipping.",
                 "Exact backpropagation using chain rule: dZ2 = P - Y, dW2 = A1ᵀ·dZ2, dA1 = dZ2·W2ᵀ, dZ1 = dA1 ⊙ f'(Z1).",
                 "Optimizers: Implemented Stochastic Gradient Descent (SGD) with Momentum and L2 weight decay."
             ])

    add_card(slide5, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Activation Function Comparative Experiment", [
                 "Evaluated ReLU vs. Sigmoid vs. Tanh under identical multi-class conditions:",
                 "• ReLU (Rectified Linear Unit):",
                 "    - Final Loss: 0.7656 | Accuracy: 77.2%",
                 "    - Solved vanishing gradient problem, accelerated learning convergence.",
                 "• Tanh (Hyperbolic Tangent):",
                 "    - Final Loss: 1.0828 | Accuracy: 56.2%",
                 "    - Zero-centered but suffered from saturation in deeper layers.",
                 "• Sigmoid Function:",
                 "    - Final Loss: 1.2780 | Accuracy: 44.2%",
                 "    - Severe gradient vanishing at extreme values hindered optimization.",
                 "Empirical finding directly guided ReLU adoption across all deep CNN layers."
             ])

    # -------------------------------------------------------------
    # SLIDE 6: Custom CNN Architecture (PyTorch from Scratch)
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    apply_background(slide6, RGBColor(255, 255, 255))
    add_header(slide6, "Custom CNN Architecture Designed for OCT Analysis")

    add_card(slide6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Convolutional Feature Extractor", [
                 "Stage 1: Conv2d(3 -> 32, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2) [Output: 112x112]",
                 "Stage 2: Conv2d(32 -> 64, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2) [Output: 56x56]",
                 "Stage 3: Conv2d(64 -> 128, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2) [Output: 28x28]",
                 "Stage 4: Conv2d(128 -> 256, 3x3) + BatchNorm + ReLU + AdaptiveAvgPool2d(4x4) [Output: 4x4]",
                 "Hierarchical feature learning: captures low-level retinal layer edges up to high-level cystic fluid shapes."
             ])

    add_card(slide6, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Regularization & Classification Head", [
                 "Flatten: Transforms 256 x 4 x 4 = 4,096 feature maps into dense representation.",
                 "Dense Layer 1: Linear(4096 -> 256) + BatchNorm1d(256) + ReLU.",
                 "Dropout Layer 1 (Rate: 0.4): Randomly zeroes out 40% of neurons to break co-adaptation.",
                 "Dense Layer 2: Linear(256 -> 64) + ReLU.",
                 "Dropout Layer 2 (Rate: 0.3): Secondary regularization barrier.",
                 "Output Layer: Linear(64 -> 4) producing raw logits for CNV, DME, Drusen, and Normal.",
                 "End-to-end trained using AdamW with Cross-Entropy Loss."
             ])

    # -------------------------------------------------------------
    # SLIDE 7: Transfer Learning Architectures
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    apply_background(slide7, RGBColor(255, 255, 255))
    add_header(slide7, "Pretrained Transfer Learning Architectures")

    add_card(slide7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.5),
             "ResNet18 (Residual Learning)", [
                 "Residual skip-connections (F(x) + x) eliminate vanishing gradients.",
                 "512-dim feature embedding adapted to 4-class retinal classifier.",
                 "Outstanding balance of high accuracy and low parameter count."
             ])

    add_card(slide7, Inches(6.8), Inches(1.6), Inches(5.6), Inches(2.5),
             "MobileNetV2 (Edge Optimized)", [
                 "Depthwise separable convolutions + inverted residual bottlenecks.",
                 "Low memory footprint (~14MB), ultra-fast inference speed.",
                 "Ideal for point-of-care embedded devices and mobile teleophthalmology."
             ])

    add_card(slide7, Inches(0.8), Inches(4.4), Inches(5.6), Inches(2.5),
             "DenseNet121 (Dense Feature Reuse)", [
                 "Direct connections from every layer to all subsequent layers.",
                 "1024-dim dense representation with continuous gradient propagation.",
                 "Excels at detecting subtle sub-RPE drusen and fine fluid borders."
             ])

    add_card(slide7, Inches(6.8), Inches(4.4), Inches(5.6), Inches(2.5),
             "VGG16 (Deep Classical Baseline)", [
                 "Deep uniform architecture with 13 convolutional and 3 dense layers.",
                 "Replaced final classification head with 4-output dropout block.",
                 "Serves as comprehensive classical deep learning benchmark."
             ])

    # -------------------------------------------------------------
    # SLIDE 8: Optimization & Regularization Strategies
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    apply_background(slide8, RGBColor(255, 255, 255))
    add_header(slide8, "Optimization, Regularization & Training Controls")

    add_card(slide8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Regularization Techniques", [
                 "Dropout Regularization: 30% to 40% dropout injected in classification heads prevents feature co-adaptation.",
                 "L2 Regularization (Weight Decay): Set to 1e-4 in AdamW optimizer to prevent weight explosion and overparameterization.",
                 "Batch Normalization: Normalizes activation distributions across mini-batches, stabilizing learning rates.",
                 "Early Stopping: Monitored validation loss with patience of 3 epochs; automatically restores the best historical model weights to eliminate overfitted epochs.",
                 "Stratified Validation: Guarantees exact class parity during model selection."
             ])

    add_card(slide8, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Hyperparameters & Learning Rate Dynamics", [
                 "Optimizer: AdamW (Adaptive Moment Estimation with decoupled weight decay).",
                 "Initial Learning Rate: 1e-4 for transfer learning models; 1e-3 for custom scratch CNN.",
                 "LR Scheduler: ReduceLROnPlateau (factor: 0.5, patience: 2) to dynamically anneal learning rates upon loss stagnation.",
                 "Loss Criterion: Multi-class Cross-Entropy Loss with Softmax normalization.",
                 "Batch Size: 32 scans per batch balancing gradient fidelity with memory efficiency.",
                 "Hardware Acceleration: Automatic device detection (CUDA GPU fallback to CPU)."
             ])

    # -------------------------------------------------------------
    # SLIDE 9: Model Evaluation & Benchmark Comparison
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    apply_background(slide9, RGBColor(255, 255, 255))
    add_header(slide9, "Comprehensive Model Benchmark on Unseen Test Scans")

    add_card(slide9, Inches(0.8), Inches(1.6), Inches(11.6), Inches(5.2),
             "Empirical Benchmark Across 5 Architectures (150 Unseen Test Scans)", [
                 "Evaluation Cohort: 150 unseen, held-out clinical OCT images strictly separated from training/validation.",
                 "• VGG16 (Top Performer): Accuracy: 81.33% | Precision: 0.837 | Recall: 0.814 | F1-Score: 0.810 | Misclassified: 28/150",
                 "• DenseNet121: Accuracy: 78.67% | Precision: 0.828 | Recall: 0.787 | F1-Score: 0.783 | Misclassified: 32/150",
                 "• MobileNetV2: Accuracy: 76.00% | Precision: 0.784 | Recall: 0.761 | F1-Score: 0.757 | Misclassified: 36/150",
                 "• ResNet18: Accuracy: 72.00% | Precision: 0.724 | Recall: 0.720 | F1-Score: 0.720 | Misclassified: 42/150",
                 "• Custom CNN (Scratch): Accuracy: 65.33% | Precision: 0.687 | Recall: 0.656 | F1-Score: 0.622 | Misclassified: 52/150",
                 "Key Insight: Classical deep stacking with uniform 3x3 convolutions (VGG16) and dense reuse (DenseNet121) proved most effective at capturing subtle textural anomalies across retinal layers.",
                 "Diagnostic Confusion: Normal and CNV categories achieved highest distinctiveness; minor confusion observed between early Drusen and small DME cysts due to visual boundary similarity."
             ])

    # -------------------------------------------------------------
    # SLIDE 10: Model Explainability with Grad-CAM
    # -------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    apply_background(slide10, RGBColor(255, 255, 255))
    add_header(slide10, "Model Explainability: Grad-CAM Retinal Layer Heatmaps")

    add_card(slide10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Grad-CAM Mathematical Methodology", [
                 "Gradient-weighted Class Activation Mapping computes gradients of target class score yc w.r.t feature activation map Ak.",
                 "Global Average Pooling computes importance neuron weights: αkc = (1/Z) ∑∑ (∂yc / ∂Aki,j).",
                 "Linear combination of forward activations weighted by αkc.",
                 "Rectified Linear Unit (ReLU) ensures only features positively correlating with the diagnosis are highlighted.",
                 "Bilinear upsampling and JET colormap blending directly over the original grayscale OCT B-scan."
             ])

    add_card(slide10, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Clinical Anatomical Correlation", [
                 "CNV Verification: Grad-CAM consistently focuses on subretinal fluid accumulation and hyperreflective membranes beneath the RPE.",
                 "DME Verification: Activations precisely isolate intraretinal hyporeflective cystoid spaces in the inner nuclear layer.",
                 "Drusen Verification: Focuses on undulating dome-shaped elevations of the RPE layer.",
                 "Normal Verification: Shows distributed attention across intact foveal contour and continuous retinal strata without focal hotspots.",
                 "Crucial for clinician confidence, algorithmic auditing, and regulatory approval."
             ])

    # -------------------------------------------------------------
    # SLIDE 11: Streamlit Web Application & Serialization (.pkl)
    # -------------------------------------------------------------
    slide11 = prs.slides.add_slide(blank_layout)
    apply_background(slide11, RGBColor(255, 255, 255))
    add_header(slide11, "Streamlit Web Application & Model Serialization (.pkl)")

    add_card(slide11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Interactive Application Capabilities (app.py)", [
                 "Dual Input Modality: Upload custom OCT scans (PNG/JPG/JPEG) or select from pre-loaded clinical sample library.",
                 "Model Switcher: Real-time dropdown to compare inference across Best Model, ResNet18, MobileNetV2, DenseNet121, Custom CNN, and VGG16.",
                 "Diagnostic Readout: Clear class badge with percentage confidence score and full probability distribution.",
                 "Real-time Grad-CAM Heatmap: Interactive toggle to view original scan, raw heatmap, and diagnostic overlay side-by-side.",
                 "Clinical Knowledge Base: Built-in reference guides on pathology, biomarkers, and clinical treatments.",
                 "Prominent Disclaimer: Alerts users to educational prototype scope."
             ])

    add_card(slide11, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Model Serialization (.pkl Format)", [
                 "Full Pipeline Serialized to PKL: models/best_retinal_oct_model.pkl.",
                 "Package Contents:",
                 "  • Model Architecture Definition & Weights (state_dict)",
                 "  • Complete Pickled PyTorch Model Object",
                 "  • Test Performance Metrics (Accuracy, Precision, Recall, F1)",
                 "  • Preprocessing, Normalization Parameters & Class Names",
                 "  • Full Benchmark Comparison Table",
                 "Dual format support: also exports standard PyTorch .pt weights for developer compatibility.",
                 "Ready for immediate loading into web services, REST APIs, or edge clinical applications."
             ])

    # -------------------------------------------------------------
    # SLIDE 12: Challenges, Key Findings & Future Scope
    # -------------------------------------------------------------
    slide12 = prs.slides.add_slide(blank_layout)
    apply_background(slide12, RGBColor(255, 255, 255))
    add_header(slide12, "Challenges, Key Findings & Future Scope")

    add_card(slide12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Challenges & Key Findings", [
                 "Visual Similarity: Distinguishing early drusen deposits from small diabetic cysts required deep multi-scale features.",
                 "Transfer Learning Advantage: ImageNet pretrained weights converged 4x faster with markedly higher sensitivity than scratch CNNs.",
                 "Explainability Necessity: Grad-CAM proved indispensable in confirming the model learned real retinal anatomy rather than peripheral image artifacts.",
                 "Efficiency: MobileNetV2 achieved near-ResNet performance with minimal latency, unlocking real-time teleophthalmology."
             ])

    add_card(slide12, Inches(6.8), Inches(1.6), Inches(5.6), Inches(5.2),
             "Future Engineering Scope", [
                 "Volumetric 3D OCT: Expand from 2D B-scans to 3D volumetric macular cube processing.",
                 "Multi-Modal AI: Fuse OCT B-scans with Color Fundus Photography and patient clinical history.",
                 "Fluid Segmentation: Incorporate U-Net architectures for exact pixel-wise quantification of subretinal and intraretinal fluid volume.",
                 "Clinical Trials & Regulatory Compliance: Conduct prospective clinical validation for FDA / CE-mark SaMD certification."
             ])

    out_path = "Retinal_Disease_Detection_OCT.pptx"
    prs.save(out_path)
    print(f"Presentation successfully created and saved to: {out_path}")


if __name__ == "__main__":
    create_presentation()

"""
Retinal Disease Detection Using Deep Learning on OCT Images
Interactive Clinical & Educational Streamlit Application.
Features:
- Multi-model inference (Best Model, ResNet18, MobileNetV2, DenseNet121, Custom CNN, VGG16)
- File uploader & pre-loaded real clinical OCT sample selector
- Disease classification with confidence scores and full probability distribution
- Real-time Explainable AI (Grad-CAM) retinal layer biomarker heatmap overlay
- Comprehensive clinical knowledge base & model benchmark dashboard
- Educational prototype disclaimer
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st

from src.dataset import CLASS_NAMES, CLASS_DESCRIPTIONS, get_transforms
from src.gradcam import GradCAM
from src.models import build_model


# Page Configuration
st.set_page_config(
    page_title="Retinal OCT Disease AI Diagnostic Prototype",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .disease-badge-cnv {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 10px;
        border-radius: 15px;
        font-weight: 700;
    }
    .disease-badge-dme {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 10px;
        border-radius: 15px;
        font-weight: 700;
    }
    .disease-badge-drusen {
        background-color: #EDE7F6;
        color: #4A148C;
        padding: 4px 10px;
        border-radius: 15px;
        font-weight: 700;
    }
    .disease-badge-normal {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 15px;
        font-weight: 700;
    }
    .disclaimer-box {
        background-color: #FFF8E1;
        border-left: 5px solid #FF8F00;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #6D4C41;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model_pkl(model_path):
    """Loads model from .pkl package or PyTorch checkpoint with caching."""
    if not os.path.exists(model_path):
        return None

    # Try loading as PKL package
    try:
        with open(model_path, "rb") as f:
            pkg = pickle.load(f)
            if isinstance(pkg, dict) and "model" in pkg:
                model = pkg["model"]
                model.eval()
                return model, pkg.get("model_name", "model"), pkg.get("metrics", {})
    except Exception:
        pass

    # Fallback to PyTorch checkpoint
    try:
        base_name = os.path.splitext(os.path.basename(model_path))[0].lower()
        if "best" in base_name or "resnet18" in base_name:
            arch = "resnet18"
        elif "mobilenet" in base_name:
            arch = "mobilenet_v2"
        elif "dense" in base_name:
            arch = "densenet121"
        elif "vgg" in base_name:
            arch = "vgg16"
        else:
            arch = "custom_cnn"

        model = build_model(arch, num_classes=4, pretrained=False)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, arch, {}
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return None


def run_inference(model, image):
    """Runs model forward pass and returns prediction details."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    _, eval_tf = get_transforms(img_size=224)
    tensor = eval_tf(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

    return pred_idx, confidence, probs, tensor


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/visible.png", width=70)
    st.sidebar.title("Configuration")

    # Available Model Files
    model_options = {
        "Best Performing Model (PKL)": "models/best_retinal_oct_model.pkl",
        "ResNet18 (Transfer Learning)": "models/resnet18.pkl",
        "MobileNetV2 (Edge Optimized)": "models/mobilenet_v2.pkl",
        "DenseNet121 (Dense Feature Reuse)": "models/densenet121.pkl",
        "Custom CNN (Trained from Scratch)": "models/custom_cnn.pkl",
        "VGG16 (Deep Architecture)": "models/vgg16.pkl",
    }

    selected_model_label = st.sidebar.selectbox("Select Model Architecture", list(model_options.keys()))
    selected_model_path = model_options[selected_model_label]

    model_bundle = load_cached_model_pkl(selected_model_path)
    if model_bundle is None:
        # Fallback to .pt if .pkl still finalizing
        pt_path = selected_model_path.replace(".pkl", ".pt")
        model_bundle = load_cached_model_pkl(pt_path)

    if model_bundle is not None:
        model, m_name, m_metrics = model_bundle
        st.sidebar.success(f"Loaded: **{m_name.upper()}**")
        if m_metrics and "Accuracy" in m_metrics:
            st.sidebar.metric("Test Accuracy", f"{m_metrics['Accuracy']*100:.1f}%")
            st.sidebar.metric("Macro F1-Score", f"{m_metrics['F1-Score']:.3f}")
    else:
        st.sidebar.warning("Model file not yet generated. Please wait for training pipeline to complete.")
        model = None

    # Input Mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("OCT Input Source")
    input_source = st.sidebar.radio("Choose Input Method", ["Pre-loaded Clinical Samples", "Upload OCT Scan"])

    sample_dir = "assets/sample_images"
    selected_image = None
    image_title = ""

    if input_source == "Pre-loaded Clinical Samples":
        if os.path.exists(sample_dir):
            sample_files = sorted(os.listdir(sample_dir))
            sample_choice = st.sidebar.selectbox("Choose Sample Image", sample_files)
            if sample_choice:
                img_path = os.path.join(sample_dir, sample_choice)
                selected_image = Image.open(img_path).convert("RGB")
                image_title = sample_choice
        else:
            st.sidebar.error("Sample images folder not found.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Retinal OCT scan (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_title = uploaded_file.name
            except Exception as e:
                st.sidebar.error(f"Invalid image file: {e}")

    # Main UI Tabs
    tab_predict, tab_knowledge, tab_benchmarks = st.tabs([
        "🔬 OCT Analysis & Explainability",
        "📖 Clinical Knowledge Base",
        "📊 Model Benchmarks & Evaluation"
    ])

    with tab_predict:
        st.markdown('<div class="main-header">Retinal Disease Detection Using Deep Learning</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Automated identification and classification of retinal abnormalities: '
            '<b>CNV</b>, <b>DME</b>, <b>Drusen</b>, and <b>Normal</b> scans with <b>Grad-CAM Explainable AI</b>.</div>',
            unsafe_allow_html=True
        )

        if selected_image is None:
            st.info("👈 Please select a pre-loaded sample scan or upload an OCT scan from the sidebar to begin analysis.")
        elif model is None:
            st.error("No trained model loaded. Please run the training pipeline first.")
        else:
            # Inference execution
            pred_idx, conf, probs, tensor = run_inference(model, selected_image)
            predicted_class = CLASS_NAMES[pred_idx]

            badge_styles = {
                "CNV": "disease-badge-cnv",
                "DME": "disease-badge-dme",
                "DRUSEN": "disease-badge-drusen",
                "NORMAL": "disease-badge-normal"
            }

            col_img, col_pred = st.columns([1, 1.2])

            with col_img:
                st.subheader("Input OCT B-Scan")
                st.image(selected_image, caption=f"{image_title} ({selected_image.size[0]}x{selected_image.size[1]} px)", use_container_width=True)

            with col_pred:
                st.subheader("Diagnostic Classification")
                badge_class = badge_styles.get(predicted_class, "disease-badge-normal")
                
                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size:0.9rem; color:#78909C; text-transform:uppercase;">Predicted Diagnosis</span><br>
                    <span style="font-size:1.8rem; font-weight:800; color:#1A237E;">{predicted_class}</span>
                    <span class="{badge_class}" style="margin-left:12px; font-size:1rem;">{conf*100:.1f}% Confidence</span>
                    <p style="margin-top:10px; font-size:0.95rem; color:#37474F;">
                        {CLASS_DESCRIPTIONS.get(predicted_class, "")}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### Prediction Probability Distribution")
                prob_df = pd.DataFrame({
                    "Condition": CLASS_NAMES,
                    "Probability": probs
                })
                for idx, row in prob_df.iterrows():
                    c_name = row["Condition"]
                    p_val = float(row["Probability"])
                    st.write(f"**{c_name}**: `{p_val*100:.1f}%`")
                    st.progress(p_val)

            # Grad-CAM Explainability Section
            st.markdown("---")
            st.subheader("🔍 Explainable AI: Grad-CAM Retinal Layer Localization")
            st.caption("Visualizes the specific spatial regions within the OCT scan (e.g., subretinal fluid, retinal pigment epithelium detachment, or hyperreflective foci) that most strongly influenced the model's decision.")

            enable_gradcam = st.checkbox("Generate Grad-CAM Heatmap Visualization", value=True)
            if enable_gradcam:
                try:
                    gradcam = GradCAM(model)
                    heatmap, _, _, _ = gradcam.generate_heatmap(tensor, target_class=pred_idx)
                    blended, heatmap_img = GradCAM.overlay_heatmap_on_image(selected_image, heatmap, alpha=0.45)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(selected_image, caption="Original OCT B-Scan", use_container_width=True)
                    with c2:
                        st.image(heatmap_img, caption=f"Grad-CAM Heatmap ({predicted_class})", use_container_width=True)
                    with c3:
                        st.image(blended, caption="Diagnostic Region Overlay", use_container_width=True)

                    st.info(f"💡 **Biomarker Interpretation**: The warm red/yellow zones highlight the anatomical areas that guided the model to classify this scan as **{predicted_class}**.")
                except Exception as e:
                    st.error(f"Grad-CAM calculation failed: {e}")

        # Prominent Mandatory Disclaimer Banner
        st.markdown("""
        <div class="disclaimer-box">
            <b>⚠️ MANDATORY CLINICAL PROTOTYPE DISCLAIMER:</b><br>
            This software is an <b>educational research prototype</b> developed for demonstrating deep learning algorithms in ophthalmic computer vision.
            It is <b>NOT</b> a cleared or approved medical device and must <b>NEVER</b> be used as a substitute for professional clinical diagnosis, treatment decisions, or patient care.
        </div>
        """, unsafe_allow_html=True)

    with tab_knowledge:
        st.header("Ophthalmic Pathology & OCT Biomarkers")
        st.write("Overview of the four retinal categories analyzed by this system:")

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            with st.expander("🔴 Choroidal Neovascularization (CNV)", expanded=True):
                st.write("""
                - **Definition**: Formation of abnormal capillary networks originating from the choroid that break through Bruch's membrane into the sub-RPE or subretinal space.
                - **Key Risk**: Major cause of rapid, severe central vision loss in wet (neovascular) Age-related Macular Degeneration (AMD).
                - **OCT Hallmarks**:
                  - Disruption of the Retinal Pigment Epithelium (RPE)
                  - Subretinal fluid (SRF) accumulation
                  - Sub-RPE hyperreflective fibrovascular membranes
                - **Clinical Treatment**: Urgent intravitreal injections of Anti-VEGF agents (e.g., Aflibercept, Ranibizumab).
                """)

            with st.expander("🟠 Diabetic Macular Edema (DME)", expanded=True):
                st.write("""
                - **Definition**: Leakage and fluid accumulation within the central macula caused by chronic breakdown of the blood-retinal barrier in diabetic patients.
                - **Key Risk**: Leading cause of moderate-to-severe visual impairment in working-age diabetic adults.
                - **OCT Hallmarks**:
                  - Intraretinal cystoid fluid spaces (sponge-like thickening)
                  - Subretinal fluid neurosensory detachments
                  - Loss of foveal depression / foveal architectural distortion
                - **Clinical Treatment**: Anti-VEGF pharmacotherapy, focal laser photocoagulation, and strict glycemic control.
                """)

        with col_k2:
            with st.expander("🟣 Drusen Deposits", expanded=True):
                st.write("""
                - **Definition**: Extracellular accumulation of lipid and protein debris located between the basal lamina of the RPE and the inner collagenous layer of Bruch's membrane.
                - **Key Risk**: Primary clinical hallmark of dry (non-neovascular) Age-related Macular Degeneration (AMD); high risk of progressing to geographic atrophy or wet AMD.
                - **OCT Hallmarks**:
                  - Dome-shaped or undulating elevations of the hyperreflective RPE layer
                  - Variable internal reflectivity (homogeneous or heterogeneous)
                  - Intact neurosensory retina in early stages
                - **Clinical Management**: Regular OCT surveillance, Amsler grid self-monitoring, AREDS-2 antioxidant supplementation.
                """)

            with st.expander("🟢 Normal Retinal Anatomy", expanded=True):
                st.write("""
                - **Definition**: Healthy human retina with preserved structural integrity and cellular stratification.
                - **OCT Hallmarks**:
                  - Distinct, symmetrical foveal depression (foveal pit)
                  - Continuous, well-defined hyperreflective RPE/Bruch's complex
                  - Intact Ellipsoid Zone (EZ) and External Limiting Membrane (ELM)
                  - Completely free of intraretinal or subretinal fluid, cystic spaces, or elevations
                """)

    with tab_benchmarks:
        st.header("Model Performance & Comprehensive Benchmarks")
        st.write("Detailed comparative analysis of all trained architectures on the unseen Kermany2017 OCT test dataset.")

        csv_path = "outputs/model_benchmark_results.csv"
        if os.path.exists(csv_path):
            bench_df = pd.read_csv(csv_path)
            st.dataframe(bench_df.style.highlight_max(subset=["Accuracy", "F1-Score"], color="#C8E6C9"), use_container_width=True)
        else:
            st.info("Benchmark summary CSV will be displayed once training completes.")

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            st.subheader("Model Comparison Metrics")
            comp_img = "outputs/model_comparison_metrics.png"
            if os.path.exists(comp_img):
                st.image(comp_img, use_container_width=True)
            else:
                st.caption("Benchmark chart will appear upon pipeline completion.")

        with c_b2:
            st.subheader("Training & Validation Curves")
            train_img = "outputs/training_curves.png"
            if os.path.exists(train_img):
                st.image(train_img, use_container_width=True)
            else:
                st.caption("Training curves will appear upon pipeline completion.")

        st.subheader("Confusion Matrices (Test Dataset)")
        cm_img = "outputs/confusion_matrices.png"
        if os.path.exists(cm_img):
            st.image(cm_img, use_container_width=True)

        st.subheader("Grad-CAM Explainability Gallery")
        grad_gallery = "outputs/gradcam_examples.png"
        if os.path.exists(grad_gallery):
            st.image(grad_gallery, use_container_width=True)


if __name__ == "__main__":
    main()

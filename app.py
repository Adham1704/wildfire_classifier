import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------
st.set_page_config(
    page_title="FireGuard AI Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Injection for LIGHT MODE
st.markdown("""
    <style>
    /* 1. Main Background - Clean White */
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
    }
    
    /* 2. Sidebar Background - Light Gray */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    
    /* 3. Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #D93025 100%);
        color: white;
        border-radius: 8px;
        height: 3.5em;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 4. Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    label[data-testid="stMetricLabel"] {
        color: #666666; /* Darker label text for white background */
    }
    div[data-testid="stMetricValue"] {
        color: #333333;
    }
    
    /* 5. Headers */
    h1 {
        background: -webkit-linear-gradient(#D32F2F, #FF5722);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
    }
    h2, h3 {
        color: #333333;
    }
    
    /* 6. Images */
    img {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2. Model Loading (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    # Make sure this matches your actual filename
    model = tf.keras.models.load_model("fire_mobilenetv2_best_finetuned.h5", compile=False)
    return model

with st.spinner("🤖 Initializing Neural Defense System..."):
    try:
        model = load_model()
        LAST_CONV_LAYER = "Conv_1" 
    except Exception as e:
        st.error(f"❌ System Failure: Could not load model. {e}")
        st.stop()

# -----------------------------
# 3. Grad-CAM Logic
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        results = grad_model(img_array)
        
        conv_output = results[0]
        prediction = results[1]

        # Fix for list wrapping in newer TF versions
        if isinstance(conv_output, list): conv_output = conv_output[0]
        if isinstance(prediction, list): prediction = prediction[0]

        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy(), float(prediction[0][0])

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# -----------------------------
# 4. Sidebar Control Panel
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/785/785116.png", width=80)
    st.markdown("## 🛡️ Control Center")
    st.info("System Online: Ready for Analysis")
    
    uploaded_file = st.file_uploader("Upload Surveillance Feed", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Calibration")
    alpha_slider = st.slider("Heatmap Intensity", 0.0, 1.0, 0.4, 0.1)
    
    st.markdown("---")
    with st.expander("ℹ️ Threat Classification Levels"):
        st.markdown("""
        * 🟢 **Normal (0-30%):** Safe environment.
        * 🟡 **Advisory (30-60%):** Suspicious thermal activity.
        * 🟠 **Warning (60-85%):** High probability of fire.
        * 🔴 **Critical (>85%):** Confirmed fire signature.
        """)

# -----------------------------
# 5. Main Dashboard
# -----------------------------
st.title("🔥 FireGuard Intelligence Grid")
st.markdown("### Autonomous Visual Recognition & Threat Assessment")

if uploaded_file is None:
    # Placeholder State
    st.info("👈 Waiting for data stream. Please upload an image.")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Uses your local image '1.jpg' as the placeholder
        try:
            st.image("1.jpg", caption="System Standby Mode", use_column_width=True)
        except:
            st.warning("Placeholder image '1.jpg' not found. Please upload a file.")

else:
    # -----------------------------
    # Processing
    # -----------------------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    IMG_SIZE = (224, 224)
    img_resized = cv2.resize(opencv_image, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # -----------------------------
    # Analysis UI
    # -----------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Source Feed")
        st.image(opencv_image, channels="BGR", use_column_width=True, caption="Raw Input Signal")

    # Trigger Analysis
    analyze_btn = st.button("🚀 INITIATE THREAT SCAN", type="primary")

    if analyze_btn:
        with st.spinner("⚡ Processing Neural Layers..."):
            # Inference
            heatmap, score = make_gradcam_heatmap(img_batch, model, LAST_CONV_LAYER)
            overlay = overlay_heatmap(opencv_image, heatmap, alpha=alpha_slider)

            # -----------------------------
            # 4-LEVEL THREAT LOGIC
            # -----------------------------
            if score < 0.30:
                # Level 1: Normal
                threat_level = "NORMAL"
                threat_color = "#28a745" # Green
                bg_color = "rgba(40, 167, 69, 0.1)"
                message = "No thermal anomalies detected. Environment is safe."
                icon = "✅"
                
            elif 0.30 <= score < 0.60:
                # Level 2: Advisory
                threat_level = "ADVISORY"
                threat_color = "#ffc107" # Amber/Yellow
                bg_color = "rgba(255, 193, 7, 0.1)"
                message = "Slight anomaly detected. Monitor situation closely."
                icon = "⚠️"
                
            elif 0.60 <= score < 0.85:
                # Level 3: Warning
                threat_level = "WARNING"
                threat_color = "#fd7e14" # Orange
                bg_color = "rgba(253, 126, 20, 0.1)"
                message = "High probability of fire detected. Potential hazard."
                icon = "🔥"
                
            else:
                # Level 4: Critical
                threat_level = "CRITICAL"
                threat_color = "#dc3545" # Red
                bg_color = "rgba(220, 53, 69, 0.1)"
                message = "Confirmed fire signature. Immediate action required."
                icon = "🚨"

        # -----------------------------
        # Results View
        # -----------------------------
        with col2:
            st.subheader("🧠 Neural Attention Map")
            st.image(overlay, use_column_width=True, caption="Target Localization")

        st.divider()
        
        # -----------------------------
        # Professional Metrics
        # -----------------------------
        st.markdown("### 📊 Telemetry Report")
        
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric("Threat Assessment", threat_level, delta=icon, delta_color="off")
            
        with m2:
            # Display risk percentage
            st.metric("Risk Probability", f"{score*100:.1f}%", delta="Confidence")
            
        with m3:
            st.metric("System Mode", "Grad-CAM", delta="Active")

        # -----------------------------
        # Status Banner (Light Mode Version)
        # -----------------------------
        st.markdown(f"""
        <div style="
            background-color: {bg_color}; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 10px solid {threat_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: {threat_color}; margin:0;">{icon} STATUS: {threat_level}</h3>
            <p style="color: #333; margin-top: 5px; font-size: 1.1em; font-weight: 500;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
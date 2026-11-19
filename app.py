import os
import sys
import io
import hashlib
from datetime import datetime
import pandas as pd
from PIL import Image
import numpy as np
import streamlit as st

# Resolve project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import ML components
from ml_core import (
    CattleDetector, 
    BreedClassifier, 
    DataPreprocessor, 
    get_target_breeds, 
    IMG_SIZE
)

# Page configuration
st.set_page_config(
    page_title="Livestock Breed ID",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling
st.markdown("""
<style>
    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-mid {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format breed names cleanly
def format_breed_name(raw_name):
    if not raw_name or raw_name == "N/A":
        return raw_name
    return raw_name.replace("_", " ").title()

# -------------------------------------------------------------
# Cached ML Pipeline (Loads models once into memory)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection and classification models...")
def load_ml_pipeline():
    breeds = get_target_breeds()
    detector = CattleDetector()
    classifier = BreedClassifier(num_classes=len(breeds) if breeds else 50)
    preprocessor = DataPreprocessor(target_size=IMG_SIZE)
    return detector, classifier, preprocessor, breeds

detector, classifier, preprocessor, TARGET_BREEDS = load_ml_pipeline()

# -------------------------------------------------------------
# Session State Initialization & Reset Utilities
# -------------------------------------------------------------
if "census_log" not in st.session_state:
    st.session_state["census_log"] = []

if "logged_file_keys" not in st.session_state:
    st.session_state["logged_file_keys"] = set()

if "processed_cache" not in st.session_state:
    st.session_state["processed_cache"] = {}

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "camera_key" not in st.session_state:
    st.session_state["camera_key"] = 0

if "active_sample" not in st.session_state:
    st.session_state["active_sample"] = None

def reset_all_state():
    """Completely resets uploader, camera capture, sample selections, and logs."""
    st.session_state["uploader_key"] += 1
    st.session_state["camera_key"] += 1
    st.session_state["active_sample"] = None
    st.session_state["processed_cache"].clear()
    st.session_state["logged_file_keys"].clear()
    st.session_state["census_log"] = []

# -------------------------------------------------------------
# Sidebar: Breed Directory & Quick Guide
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐄 Cattle Breed ID")
    st.caption("AI-Assisted Livestock Identification")
    st.markdown("---")

    st.subheader(f"Recognized Breeds ({len(TARGET_BREEDS)})")
    search_query = st.text_input("Search breeds", placeholder="e.g. Gir, Sahiwal...")
    
    filtered_breeds = [
        b for b in TARGET_BREEDS 
        if search_query.lower() in format_breed_name(b).lower()
    ]
    
    with st.expander(f"Browse List ({len(filtered_breeds)} found)", expanded=False):
        for b in filtered_breeds:
            st.markdown(f"• **{format_breed_name(b)}**")

    st.markdown("---")
    st.markdown(
        "<small style='color: #6b7280;'>Supports 50 indigenous and exotic cattle breeds. "
        "Images are processed in-memory for instant results.</small>", 
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="header-title">🐄 Livestock Breed Identification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-subtitle">Upload or photograph cattle to automatically locate the animal and identify its breed with confidence scoring.</div>', 
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Input Tabs
# -------------------------------------------------------------
tab_upload, tab_camera, tab_samples = st.tabs([
    "📁 Upload Image(s)", 
    "📸 Camera Capture", 
    "🖼️ Sample Photos"
])

images_to_process = []

with tab_upload:
    uploaded_files = st.file_uploader(
        "Select cattle image(s) to analyze",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
        help="Upload single or multiple cattle photos"
    )
    if uploaded_files:
        # User uploaded files, clear active sample
        st.session_state["active_sample"] = None
        for file in uploaded_files:
            try:
                img = Image.open(file).convert("RGB")
                file_key = f"upload_{file.name}_{file.size}"
                images_to_process.append({
                    "key": file_key,
                    "filename": file.name,
                    "image": img
                })
            except Exception as e:
                st.error(f"Could not open {file.name}: {e}")

with tab_camera:
    camera_file = st.camera_input(
        "Take a photo of cattle in the field",
        key=f"camera_{st.session_state['camera_key']}"
    )
    if camera_file:
        st.session_state["active_sample"] = None
        try:
            cam_bytes = camera_file.getvalue()
            cam_hash = hashlib.md5(cam_bytes).hexdigest()[:10]
            cam_img = Image.open(io.BytesIO(cam_bytes)).convert("RGB")
            filename = f"camera_capture_{cam_hash}.jpg"
            file_key = f"cam_{cam_hash}"
            images_to_process.append({
                "key": file_key,
                "filename": filename,
                "image": cam_img
            })
        except Exception as e:
            st.error(f"Camera error: {e}")

with tab_samples:
    st.write("Click an example image below to test the classifier:")
    
    sample_items = [
        {"title": "Gir Cow", "file": "sample_gir.jpg"},
        {"title": "Sahiwal Cow", "file": "sample_sahiwal.jpg"},
        {"title": "Amritmahal Cow", "file": "sample_amritmahal.jpg"},
    ]
    
    cols = st.columns(len(sample_items))
    for i, item in enumerate(sample_items):
        with cols[i]:
            p = os.path.join(DATA_DIR, "sample_images", item["file"])
            if not os.path.exists(p):
                p = os.path.join(BASE_DIR, "sample_images", item["file"])
            
            if os.path.exists(p):
                thumb = Image.open(p)
                st.image(thumb, caption=item["title"], use_container_width=True)
                if st.button(f"Test {item['title'].split()[0]}", key=f"btn_sample_{i}"):
                    st.session_state["active_sample"] = item["file"]
                    st.rerun()
            else:
                st.info(f"Sample not found: {item['file']}")

    if st.session_state["active_sample"] and not uploaded_files:
        active_f = st.session_state["active_sample"]
        sample_path = os.path.join(DATA_DIR, "sample_images", active_f)
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "sample_images", active_f)
        if os.path.exists(sample_path):
            s_img = Image.open(sample_path).convert("RGB")
            images_to_process.append({
                "key": f"sample_{active_f}",
                "filename": active_f,
                "image": s_img
            })
            if st.button("✕ Deselect Sample", key="btn_clear_sample"):
                st.session_state["active_sample"] = None
                st.rerun()

# When all inputs are removed, reset logged keys so re-uploading works
if not images_to_process:
    st.session_state["logged_file_keys"].clear()

# -------------------------------------------------------------
# Analysis & Results Section
# -------------------------------------------------------------
if images_to_process:
    st.markdown("---")
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader(f"Analysis Results ({len(images_to_process)})")
    with header_col2:
        if st.button("🔄 Clear & Reset All", key="btn_clear_all_top"):
            reset_all_state()
            st.rerun()

    for item in images_to_process:
        item_key = item["key"]
        filename = item["filename"]
        pil_img = item["image"]
        img_np = np.array(pil_img)

        # Check cache or run inference
        if item_key in st.session_state["processed_cache"]:
            result = st.session_state["processed_cache"][item_key]
        else:
            # Run Detection
            crop_rgb, bbox_info, annotated_img = detector.detect_full(img_np)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if crop_rgb is None:
                result = {
                    "crop_rgb": None,
                    "annotated_img": annotated_img,
                    "top_breed": "N/A",
                    "clean_breed": "N/A",
                    "top_conf": None,
                    "top_3": [],
                    "detected": False,
                    "timestamp": timestamp
                }
            else:
                # Run Classification
                tensor = preprocessor.preprocess(crop_rgb)
                top_breed, top_conf = classifier.predict(tensor, TARGET_BREEDS)
                top_3 = classifier.predict_top_k(tensor, TARGET_BREEDS, k=3)
                result = {
                    "crop_rgb": crop_rgb,
                    "annotated_img": annotated_img,
                    "top_breed": top_breed,
                    "clean_breed": format_breed_name(top_breed),
                    "top_conf": top_conf,
                    "top_3": top_3,
                    "detected": True,
                    "timestamp": timestamp
                }

            # Cache the result
            st.session_state["processed_cache"][item_key] = result

            # Append to census log ONLY ONCE per unique image
            if item_key not in st.session_state["logged_file_keys"]:
                st.session_state["logged_file_keys"].add(item_key)
                st.session_state["census_log"].append({
                    "Image": filename,
                    "Status": "Cattle Confirmed" if result["detected"] else "No Cattle Detected",
                    "Identified Breed": result["clean_breed"],
                    "Confidence": f"{result['top_conf']}%" if result["detected"] else "N/A",
                    "Timestamp": result["timestamp"]
                })

        # Display Result Card
        with st.container():
            col_orig, col_crop, col_pred = st.columns([1.2, 1, 1.4])

            with col_orig:
                st.markdown(f"**Source: `{filename}`**")
                st.image(result["annotated_img"], caption="Detection Preview", use_container_width=True)

            with col_crop:
                st.markdown("**Isolated Cattle**")
                if result["crop_rgb"] is not None:
                    st.image(result["crop_rgb"], caption="Cropped Subject", use_container_width=True)
                else:
                    st.info("No cattle region found.")

            with col_pred:
                st.markdown("**Identification Result**")

                if not result["detected"]:
                    st.markdown('<div class="badge-low">⚠️ No Cattle Detected</div>', unsafe_allow_html=True)
                    st.warning("No cattle were found in this image. Please provide a clear photo containing cattle.")
                else:
                    top_conf = result["top_conf"]
                    clean_name = result["clean_breed"]

                    if top_conf >= 70.0:
                        badge_html = f'<div class="badge-high">✅ High Confidence ({top_conf}%)</div>'
                    elif top_conf >= 40.0:
                        badge_html = f'<div class="badge-mid">ℹ️ Moderate Confidence ({top_conf}%)</div>'
                    else:
                        badge_html = f'<div class="badge-low">⚠️ Low Confidence ({top_conf}%)</div>'
                    
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"### **{clean_name}**")
                    st.metric(label="Match Confidence", value=f"{top_conf}%")

                    st.markdown("**Candidate Breeds:**")
                    for cand in result["top_3"]:
                        cand_name = format_breed_name(cand["breed"])
                        c_col1, c_col2 = st.columns([2, 1])
                        with c_col1:
                            st.write(cand_name)
                        with c_col2:
                            st.write(f"**{cand['confidence']}%**")
                        st.progress(float(min(cand["confidence"] / 100.0, 1.0)))

            st.markdown("---")

# -------------------------------------------------------------
# Digital Census Record & Export
# -------------------------------------------------------------
if st.session_state["census_log"]:
    st.subheader("📋 Session Records")
    st.caption("Live log of analyzed livestock images during this session.")

    df_log = pd.DataFrame(st.session_state["census_log"])
    st.dataframe(df_log, use_container_width=True)

    csv_buffer = io.StringIO()
    df_log.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
    with btn_col1:
        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name=f"livestock_census_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with btn_col2:
        if st.button("🗑️ Clear Records", key="btn_clear_records"):
            st.session_state["census_log"] = []
            st.rerun()
    with btn_col3:
        if st.button("🔄 Reset All (Inputs & Records)", key="btn_reset_all_bottom"):
            reset_all_state()
            st.rerun()

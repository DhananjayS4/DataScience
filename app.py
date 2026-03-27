import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import glob
import json
from models import ImageCaptioningModel
from utils import load_image, CaptionPredictor
from blip_predictor import BlipPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Visionary Captioning AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Glassmorphism & Premium Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@400;700&display=swap');

    :root {
        --primary-color: #6366f1;
        --secondary-color: #a855f7;
        --bg-color: #0f172a;
        --text-color: #f8fafc;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
        transform: scale(1.05);
    }

    .upload-container {
        border: 2px dashed var(--glass-border);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
    }

    .caption-box {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e2e8f0;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        background: rgba(0, 0, 0, 0.2);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("Visionary Settings")
    
    st.markdown("### 🤖 Model Selection")
    model_mode = st.radio("Choose Engine:", ["Custom Model (Training)", "Pro AI (SOTA)"])
    st.markdown("---")
    
    if model_mode == "Custom Model (Training)":
        beam_width = st.slider("Beam Width", 1, 10, 3)
        st.info("Higher beam width increases accuracy but takes more time.")
    else:
        st.success("✅ Using Pre-trained SOTA BLIP")
    st.markdown("---")
    st.markdown("Developed by Dhananjay S.")

# --- Main Page ---
st.title("👁️ Visionary Image Captioning")
st.markdown("Transform images into descriptive, natural language using advanced Transformer architectures.")

# Load Custom Model
@st.cache_resource
def get_model():
    from models import ImageCaptioningModel
    # Clear any previous model names from global space to avoid collisions
    tf.keras.backend.clear_session()
    
    # Model Hyperparameters (should match training)
    NUM_LAYERS = 4
    EMBED_DIM = 256
    NUM_HEADS = 8
    FF_DIM = 512
    VOCAB_SIZE = 10000
    MAX_LEN = 40
    
    model = ImageCaptioningModel(NUM_LAYERS, EMBED_DIM, NUM_HEADS, FF_DIM, VOCAB_SIZE, MAX_LEN)
    
    # Dummy call to build the model so weights can be loaded
    dummy_img = tf.zeros((1, 299, 299, 3))
    dummy_cap = tf.zeros((1, 1), dtype=tf.int64)
    model(dummy_img, dummy_cap, training=False, look_ahead_mask=None, padding_mask=None)
    
    # Find the latest weights
    weight_files = glob.glob('weights_epoch_*.weights.h5')
    latest_weights = 'weights.weights.h5' if os.path.exists('weights.weights.h5') else None
    
    if weight_files:
        weight_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
        latest_weights = weight_files[0]

    if latest_weights:
        st.sidebar.success(f"Loaded: `{latest_weights}`")
        model.load_weights(latest_weights)
    else:
        st.sidebar.warning("No weights found yet. Waiting for training...")
        
    return model

# Load Tokenizer with saved vocabulary
@st.cache_resource
def get_tokenizer():
    if os.path.exists('vocab.json'):
        with open('vocab.json', 'r') as f:
            vocab = json.load(f)
        tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=10000,
            output_sequence_length=40,
            vocabulary=vocab
        )
    else:
        tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=10000,
            output_sequence_length=40
        )
    return tokenizer

@st.cache_resource
def get_blip_predictor():
    from blip_predictor import BlipPredictor
    return BlipPredictor()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a scenic image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")
        img_path = "temp_image.jpg"
        image.save(img_path)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📝 Generated Caption")
    
    if uploaded_file is not None:
        if st.button("Generate Magic ✨"):
            with st.spinner("AI is dreaming up a description..."):
                try:
                    # Preprocess image
                    if model_mode == "Pro AI (SOTA)":
                        blip_predictor = get_blip_predictor()
                        caption = blip_predictor.generate_caption(img_path)
                    else:
                        model = get_model()
                        tokenizer = get_tokenizer()
                        predictor = CaptionPredictor(model, tokenizer, max_length=40)
                        
                        processed_img = load_image(img_path)
                        # Since we don't have real weights, show a fallback message or mock result
                        # Check if any weights exist in the directory
                        weight_exists = any(glob.glob('*.weights.h5'))
                        
                        if not weight_exists:
                            st.warning("⚠️ No trained weights found. Showing demo caption.")
                            caption = "a beautiful sunset over a calm lake with mountains in the background"
                        else:
                            caption = predictor.beam_search(processed_img, beam_width=beam_width)
                    
                    st.markdown(f'<div class="caption-box">{caption.capitalize()}</div>', unsafe_allow_html=True)
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload an image to start generating captions.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 💡 How it works")
st.write("""
1. **CNN Encoder**: An InceptionV3 model extracts high-level features from your image.
2. **Transformer Encoder**: Refines these features using self-attention to understand spatial relationships.
3. **Transformer Decoder**: Generates text word-by-word by attending to the image features and previously generated words.
4. **Beam Search**: Evaluates multiple sentence paths simultaneously to find the most coherent and accurate description.
""")

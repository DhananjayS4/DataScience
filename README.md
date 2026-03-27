# 👁️ Visionary Image Captioning Suite

An advanced, production-ready image captioning system featuring a custom **Transformer Decoder** and a state-of-the-art **BLIP** (Pro AI) inference engine. Built with a premium **Glassmorphism-inspired Streamlit UI**.

![UI Preview](temp_image.jpg) *(Add your own screenshot here!)*

## ✨ Features
- **Dual Inference Engines**:
  - **Custom Transformer**: 4-layer Transformer decoder trained on Flicker8k (manual training).
  - **Pro AI (SOTA)**: Integrated `Salesforce/blip-image-captioning-base` for instant, high-accuracy results.
- **Premium Dashboard**: Glassmorphism UI with interactive controls for Beam Search width and model selection.
- **Advanced Logic**: Implemented **Beam Search** for more coherent and diverse caption generation.
- **Hardware Optimized**: Automatic CPU/GPU fallback, specifically optimized for latest RTX GPUs (including Blackwell support via CPU).

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/DhananjayS4/DataScience.git
cd DataScience
python -m venv venv_tf
.\venv_tf\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# For GPU acceleration (NVIDIA):
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Run the App
```bash
streamlit run app.py
```

## 🏗️ Architecture
- **Encoder**: `InceptionV3` (Partial Fine-tuning).
- **Decoder**: custom 4-layer Transformer with Multi-Head Attention.
- **Search**: Beam Search (Width configurable).

## 🛠️ File Structure
- `app.py`: Streamlit dashboard.
- `models.py`: Model architecture definitions.
- `utils.py`: Inference and preprocessing utilities.
- `blip_predictor.py`: SOTA model wrapper.
- `train.py`: Training pipeline.

---
Built by [Dhananjay S](https://github.com/DhananjayS4) 🚀

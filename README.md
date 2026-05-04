# 🩺 WoundLens — AI-Powered Wound Staging & Clinical Decision Support

> **Hackathon Build:** 9:00 PM → 8:00 AM | **Team Size:** 4

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Team Roles & Responsibilities](#team-roles--responsibilities)
4. [Project Structure](#project-structure)
5. [Phase-by-Phase Roadmap](#phase-by-phase-roadmap)
6. [Setup Instructions](#setup-instructions)
7. [Feature Breakdown](#feature-breakdown)
8. [Dataset Resources](#dataset-resources)
9. [Pitch Strategy](#pitch-strategy)
10. [Emergency Fallbacks](#emergency-fallbacks)

---

## Project Overview

**WoundLens** is a computer-vision-powered clinical decision support system that:

| Step | What It Does |
|------|-------------|
| 📸 **Capture** | User uploads or takes a photo of a chronic wound (diabetic ulcer, pressure sore) |
| 🔬 **Segment** | AI identifies the wound bed boundary vs. healthy surrounding skin |
| 🎨 **Classify Tissue** | Calculates % of **Necrotic** (black/dead), **Slough** (yellow/infected), and **Granulation** (red/healing) tissue |
| 📊 **Stage** | Assigns a clinical wound stage (Stage 1–4) based on tissue composition and depth markers |
| 💊 **Recommend** | Suggests evidence-based treatment (dressings, debridement, referral) |
| 📈 **Track** | Shows wound healing progression over time via a mock healing tracker |

**Why it matters:** Chronic wounds cost global healthcare **$25B+ annually**. WoundLens upskills junior nurses, empowers home-care patients, and prevents amputations through early, accurate staging.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **CV Model** | YOLOv8 (Ultralytics) / Roboflow | Pre-trained wound detection, fast fine-tuning |
| **Segmentation** | OpenCV + Color-space analysis (HSV) | Tissue classification via pixel color thresholds |
| **Backend Logic** | Python 3.10+ | Tissue ratio calculation, staging logic, treatment rules |
| **Web UI** | Streamlit | Rapid prototyping, live camera feed, interactive widgets |
| **Data Viz** | Plotly / Matplotlib | Healing tracker graphs, tissue pie charts |
| **Deployment** | Streamlit Cloud / localhost | Quick demo-ready deployment |

---

## Team Roles & Responsibilities

### 👤 Member 1 — **ML Engineer** (Model & Detection)
| Phase | Tasks |
|-------|-------|
| Phase 1 | Source wound datasets (Kaggle/Roboflow), explore and clean data |
| Phase 2 | Fine-tune YOLOv8 on wound detection, validate on test set |
| Phase 3 | Export model, integrate inference pipeline with the app |
| Phase 4 | Optimize model speed, help with demo preparation |

### 👤 Member 2 — **CV Engineer** (Segmentation & Tissue Analysis)
| Phase | Tasks |
|-------|-------|
| Phase 1 | Research HSV/LAB color thresholds for Necrotic, Slough, Granulation tissue |
| Phase 2 | Build tissue segmentation module using OpenCV color-space analysis |
| Phase 3 | Build tissue ratio calculator (% Necrotic, % Slough, % Granulation) |
| Phase 4 | Generate tissue overlay visualizations (color-coded wound map) |

### 👤 Member 3 — **Full-Stack / UI Developer** (Streamlit App)
| Phase | Tasks |
|-------|-------|
| Phase 1 | Set up project structure, install dependencies, build basic Streamlit layout |
| Phase 2 | Build image upload + camera capture UI, results display panel |
| Phase 3 | Integrate model inference + tissue analysis into UI pipeline |
| Phase 4 | Build Healing Tracker page, polish UI, add animations & branding |

### 👤 Member 4 — **Domain Logic + Pitch Lead** (Staging, Treatment & Presentation)
| Phase | Tasks |
|-------|-------|
| Phase 1 | Research clinical wound staging criteria (NPUAP/EPUAP guidelines) |
| Phase 2 | Build staging engine (rule-based: tissue ratios → Stage 1–4) |
| Phase 3 | Build treatment recommendation engine, write treatment database |
| Phase 4 | Prepare pitch deck (5 slides max), rehearse 3-min pitch, prepare demo script |

---

## Project Structure

```
OVERNIGHT_HACKATHON/
│
├── README.md                    # This file — project workflow
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── app/
│   ├── app.py                   # Main Streamlit entry point
│   ├── pages/
│   │   ├── 1_📸_Analyze.py      # Wound analysis page
│   │   ├── 2_📈_Tracker.py      # Healing tracker page
│   │   └── 3_ℹ️_About.py        # About / how it works
│   └── components/
│       ├── sidebar.py           # Sidebar navigation & branding
│       └── result_card.py       # Styled result display component
│
├── models/
│   └── predict.py               # Roboflow API inference wrapper
│
├── core/
│   ├── segmentation.py          # Wound bed segmentation (OpenCV)
│   ├── tissue_classifier.py     # HSV-based tissue classification
│   ├── staging_engine.py        # Rule-based wound staging (Stage 1–4)
│   └── treatment_engine.py      # Treatment recommendation logic
│
├── data/
│   ├── dataset/                 # YOLOv8-ready structure
│   │   ├── images/
│   │   │   ├── train/           # Training images (80%)
│   │   │   └── val/             # Validation images (20%)
│   │   ├── labels/
│   │   │   ├── train/           # Matching .txt label files
│   │   │   └── val/             # (same filename as image)
│   │   └── data.yaml            # Dataset config for YOLOv8
│   ├── raw/                     # Original dataset download (backup)
│   └── sample_wounds/           # Demo images for live presentation
│
├── assets/
│   ├── logo.png                 # WoundLens logo
│   └── demo_images/             # Screenshots for README / pitch
│
├── pitch/
│   ├── pitch_deck.pptx          # 5-slide pitch deck
│   └── script.md                # Pitch talking points
│
└── notebooks/
    └── exploration.ipynb        # Quick data exploration (optional)
```

---

## Phase-by-Phase Roadmap

### ⏰ Phase 1: Foundation (9:00 PM – 11:00 PM) — 2 hours

> **Goal:** Everyone is set up, data is sourced, and the skeleton app is running.

| Who | Task | Deliverable | Done? |
|-----|------|-------------|-------|
| **All** | Git repo init, clone, install deps | Everyone can run `streamlit run app/app.py` | ☐ |
| **Member 1** | Download wound dataset from Kaggle/Roboflow | `data/raw/` populated with labeled images | ☐ |
| **Member 2** | Research & document HSV ranges for tissue types | Color thresholds doc / constants in `tissue_classifier.py` | ☐ |
| **Member 3** | Build Streamlit skeleton: Home, Analyze, Tracker, About pages | Multi-page app with navigation working | ☐ |
| **Member 4** | Research NPUAP staging criteria, document rules | Staging rules documented in `staging_engine.py` comments | ☐ |

**🚩 Checkpoint @ 11:00 PM:** Skeleton app running, dataset ready, team aligned.

---

### ⏰ Phase 2: Core Build (11:00 PM – 2:00 AM) — 3 hours

> **Goal:** Model training started, tissue analysis working, staging logic coded.

| Who | Task | Deliverable | Done? |
|-----|------|-------------|-------|
| **Member 1** | Train model using Roboflow UI on wound dataset | Model trained on Roboflow dashboard | ☐ |
| **Member 2** | Build tissue segmentation pipeline (HSV masking) | `segmentation.py` outputs wound mask, `tissue_classifier.py` outputs tissue % | ☐ |
| **Member 3** | Build Analyze page: image upload, camera capture, display results | Functional upload → display pipeline | ☐ |
| **Member 4** | Code staging engine + treatment recommendation engine | `staging_engine.py` and `treatment_engine.py` return stage + treatment | ☐ |

**🚩 Checkpoint @ 2:00 AM:** Can upload an image → get tissue breakdown → get stage + treatment (even if model is still training).

---

### ⏰ Phase 3: Integration (2:00 AM – 5:00 AM) — 3 hours

> **Goal:** End-to-end pipeline works. Upload a photo → full analysis.

| Who | Task | Deliverable | Done? |
|-----|------|-------------|-------|
| **Member 1** | Download .pt weights or get API key, build `predict.py` wrapper | `predict.py` takes image → returns bounding box via Roboflow | ☐ |
| **Member 2** | Apply tissue analysis only within detected wound region (crop to bbox) | Pipeline: detect wound → segment → classify tissue | ☐ |
| **Member 3** | Wire everything together in the UI: model → segmentation → stage → treatment | Full end-to-end working in Streamlit | ☐ |
| **Member 4** | Build treatment database (JSON), start pitch deck | `treatments.json`, first 3 slides of pitch | ☐ |

**🚩 Checkpoint @ 5:00 AM: Full demo works end-to-end.** This is the **CRITICAL** milestone.

---

### ⏰ Phase 4: Polish & Pitch (5:00 AM – 8:00 AM) — 3 hours

> **Goal:** Make it demo-ready, beautiful, and pitch-perfect.

| Who | Task | Deliverable | Done? |
|-----|------|-------------|-------|
| **Member 1** | Add confidence scores, handle edge cases (no wound detected) | Robust error handling | ☐ |
| **Member 2** | Generate beautiful tissue overlay visualization (color-coded wound map) | Overlay image: red=granulation, yellow=slough, black=necrotic | ☐ |
| **Member 3** | Build Healing Tracker (mock data showing wound shrinkage over time), polish UI | Plotly chart showing healing progression | ☐ |
| **Member 4** | Finalize pitch deck (5 slides), rehearse 3-min pitch, prepare demo script | Polished pitch + demo flow | ☐ |
| **All** | Run full demo 3 times end-to-end, fix any bugs | Bug-free demo | ☐ |

**🚩 Checkpoint @ 7:00 AM:** Full rehearsal. Everyone knows the demo flow.

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- pip
- Webcam (optional, for live demo)
- GPU recommended for training (can use Google Colab as fallback)

### Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd OVERNIGHT_HACKATHON

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/app.py
```

### Google Colab (for model training)
If no local GPU is available:
1. Upload `models/train.py` and dataset to Colab
2. Train on Colab's free GPU
3. Download `best.pt` weights
4. Place in `models/weights/best.pt`

---

## Feature Breakdown

### 🔍 Feature 1: Wound Detection (YOLOv8)
- Detects wound region in an image
- Draws bounding box around the wound
- Returns confidence score
- **Fallback:** If model isn't ready, use manual ROI selection (user draws a box)

### 🎨 Feature 2: Tissue Classification
Using HSV color-space analysis on the detected wound region:

| Tissue Type | Color in Image | HSV Range (approx) | Clinical Meaning |
|-------------|---------------|---------------------|-----------------|
| **Granulation** | Red/Pink | H: 0-10, 160-180 | Healing — good sign |
| **Slough** | Yellow/Cream | H: 20-40 | Infection risk — needs debridement |
| **Necrotic** | Black/Brown | S: low, V: low | Dead tissue — urgent intervention |
| **Healthy Skin** | Normal tones | Excluded from wound analysis | Surrounding skin |

### 📊 Feature 3: Automated Staging

| Stage | Criteria | Tissue Profile |
|-------|----------|----------------|
| **Stage 1** | Intact skin, non-blanchable redness | >80% Granulation, no depth |
| **Stage 2** | Partial-thickness skin loss | >60% Granulation, <20% Slough |
| **Stage 3** | Full-thickness skin loss | Mixed Slough + Granulation, <30% Necrotic |
| **Stage 4** | Full-thickness tissue loss | >40% Necrotic, exposed deep structures |

### 💊 Feature 4: Treatment Recommendations

| Stage | Recommended Treatment |
|-------|----------------------|
| Stage 1 | Transparent film dressing, pressure relief, skin moisturizer |
| Stage 2 | Hydrocolloid or foam dressing, saline irrigation |
| Stage 3 | Alginate dressing, wound debridement consult, infection monitoring |
| Stage 4 | Surgical consult, negative-pressure wound therapy, IV antibiotics assessment |

### 📈 Feature 5: Healing Tracker (Mock Data)
- Line chart showing wound area (cm²) decreasing over time
- Tissue composition stacked bar chart over time
- Stage progression timeline
- **Purpose:** Shows judges you're thinking about the longitudinal patient journey

---

## Dataset Resources

| Dataset | Source | Notes |
|---------|--------|-------|
| Diabetic Foot Ulcer (DFU) | [Kaggle](https://www.kaggle.com/datasets) — search "diabetic foot ulcer" | Best for fine-tuning |
| DFUC2022 Challenge | [DFUC 2022](https://dfu-challenge.github.io/) | Competition dataset, well-labeled |
| Medetec Wound Database | [Medetec](http://www.medetec.co.uk/files/medetec-image-databases.html) | Free wound images |
| Roboflow Wound Detection | [Roboflow Universe](https://universe.roboflow.com/) — search "wound" | Pre-labeled for YOLO, easiest to use |
| AZH Wound Care Dataset | [Kaggle](https://www.kaggle.com/) — search "wound care" | Mixed wound types |

> **💡 Pro Tip for Member 1:** Start with Roboflow Universe — datasets there come pre-formatted for YOLOv8, saving 30+ minutes of data prep.

---

## Pitch Strategy

### The 5-Slide Pitch Deck

| Slide | Content | Time |
|-------|---------|------|
| **1. The Problem** | "28M people suffer from chronic wounds. Misdiagnosis → amputation. Current method: rulers & paper." | 30 sec |
| **2. The Solution** | Live demo screenshot. "WoundLens: Point, Scan, Stage, Treat." | 30 sec |
| **3. How It Works** | Architecture diagram: Camera → Detection → Segmentation → Staging → Treatment | 45 sec |
| **4. Live Demo** | **LIVE DEMO** — upload a wound image, show full pipeline | 60 sec |
| **5. Impact & Future** | "$25B market. Upskills nurses. Prevents amputations. Future: FDA-cleared SaMD." | 15 sec |

### Demo Flow Script
```
1. Open app → show clean, professional UI
2. Upload a sample wound image
3. AI detects wound region (bounding box appears) — "Look at that detection"
4. Tissue overlay appears — "Red is healing, yellow needs attention"
5. Staging result pops up — "Stage 3 — the AI recommends an Alginate dressing"
6. Switch to Healing Tracker — "Over time, we track the patient's journey"
7. Close with: "This is WoundLens. One scan. Accurate staging. Better outcomes."
```

---

## Emergency Fallbacks

> Things go wrong at 3 AM. Here's the Plan B for each critical component.

| Component | If It Fails... | Fallback |
|-----------|---------------|----------|
| **YOLOv8 Training** | Model won't converge or no GPU | Use Roboflow's hosted API (free tier) for inference |
| **Tissue Segmentation** | HSV thresholds don't work well | Use K-Means clustering (k=3) on wound region pixels |
| **Camera Feed** | Webcam not working in Streamlit | Pre-loaded sample images with file upload |
| **Full Pipeline** | Integration breaks | Demo each module independently with screenshots |
| **Streamlit Cloud** | Deployment fails | Demo on localhost — it's a hackathon, that's fine |
| **Dataset Issues** | Can't find good labeled data | Use Roboflow Universe, or create 20 manual annotations |

---

## Key Commands Reference

```bash
# Run the app
streamlit run app/app.py

# Train YOLOv8
python models/train.py

# Test tissue classification on a single image
python core/tissue_classifier.py --image data/sample_wounds/test1.jpg

# Generate mock healing data
python core/healing_tracker_data.py
```

---

## Communication Plan

| Tool | Purpose |
|------|---------|
| **WhatsApp/Discord** | Quick messages, status updates |
| **Git + GitHub** | Code sharing — **push every 30 minutes** |
| **Shared Google Doc** | Pitch script, staging rules reference |

### Git Branching Strategy (Keep It Simple)
```
main          ← stable, demo-ready code only
├── feat/model       ← Member 1's model work
├── feat/cv          ← Member 2's segmentation work
├── feat/ui          ← Member 3's Streamlit work
└── feat/logic       ← Member 4's staging/treatment logic
```
**Merge to `main` at each checkpoint (11 PM, 2 AM, 5 AM).**

---

## ⚡ Critical Success Rules

1. **✅ Working > Perfect.** A rough working demo beats a polished broken one.
2. **✅ Demo at 5 AM.** Everything after 5 AM is polish. If it doesn't work by 5 AM, activate fallbacks.
3. **✅ Commit often.** Push every 30 min. Losing code at 4 AM is a disaster.
4. **✅ Test with real images early.** Don't wait until 6 AM to see if the pipeline works on actual wound photos.
5. **✅ Sleep in shifts if needed.** One person can nap 30 min while others code. Tired devs write terrible code.
6. **✅ The pitch wins the hackathon, not the code.** Spend real time on the pitch after 5 AM.

---

## License

MIT License — Built with ❤️ at [Hackathon Name] 2026

---

> **"One scan. Accurate staging. Better outcomes."**
> — WoundLens Team

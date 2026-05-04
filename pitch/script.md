# WoundLens — Pitch Deck Script & Talking Points
*Member 4 Deliverable*

## Slide 1: The Problem (The Hook)
**Title:** The Hidden Epidemic of Chronic Wounds
**Visual:** Impact metrics (28M+ patients, $25B+ market, amputation risks)
**Talking Points:**
- "Good morning judges. Every year, over 28 million people suffer from chronic wounds like diabetic foot ulcers and pressure injuries."
- "The standard of care today is shockingly primitive: nurses assessing wounds subjectively with paper rulers."
- "This subjectivity leads to a 30% misdiagnosis rate. When a wound is mis-staged, the wrong treatment is applied. The result? Infections, amputations, and skyrocketing hospital costs."

## Slide 2: The Solution (WoundLens)
**Title:** WoundLens: AI-Powered Clinical Decision Support
**Visual:** The 3-step pipeline (Upload ➔ AI Analysis ➔ Treatment Plan)
**Talking Points:**
- "We built **WoundLens**, a computer-vision pipeline that brings objective, reproducible science to wound care."
- "With a single smartphone photo, WoundLens does three things in seconds:"
  1. "It automatically **detects** the wound bed using a YOLOv8 object detection model."
  2. "It **segments and classifies** the tissue at a pixel level—distinguishing between healthy granulation tissue, risky slough, and dangerous necrotic eschar using HSV and LAB color spaces."
  3. "It feeds those exact tissue ratios into a clinical rule engine based on NPIAP guidelines to output a **standardized stage and an evidence-based treatment plan**."

## Slide 3: The Demo (The "Wow" Moment)
**Title:** Live Demonstration
**Visual:** Have the Streamlit app open on the "Analyze" page.
**Talking Points:**
- *(Upload a challenging image, like `abrasions-30` or a Stage 3 ulcer)*
- "Let's look at a live example. I'm uploading a photo of a complex wound."
- "Notice how fast it runs. Our YOLO model found the wound and cropped out the background noise perfectly."
- "Look at this Dashboard: Our OpenCV pipeline has color-coded the tissues—red for healing, yellow for slough, black for necrosis. It objectively calculated exactly 35% slough."
- "Because of that 35% slough, our rules engine automatically staged this and recommended an Alginate Dressing to absorb the exudate."

## Slide 4: Architecture & Tech Stack
**Title:** Built for Clinical Robustness
**Visual:** Simple block diagram (YOLOv8 ➔ OpenCV GrabCut ➔ Rule Engine ➔ Streamlit)
**Talking Points:**
- "We designed this for the real world."
- "We used **Roboflow & YOLOv8** for state-of-the-art bounding box detection."
- "We wrote a custom, zero-dependency **OpenCV GrabCut algorithm** for pixel-perfect segmentation—meaning it runs instantly without needing heavy GPUs."
- "The entire UI is built in **Streamlit** for rapid prototyping, and our treatment logic is decoupled into a scalable **JSON database**."

## Slide 5: Business Impact & Future
**Title:** Better Outcomes, Lower Costs
**Visual:** Arrow going up (Healing Rates), Arrow going down (Hospital Costs)
**Talking Points:**
- "WoundLens isn't just an AI toy. By standardizing wound staging, we ensure patients get the right dressing on day one."
- "This reduces healing times, prevents life-threatening sepsis, and saves hospitals millions in un-reimbursed care costs."
- "Next steps? Integration with EPIC and Cerner EHRs, and adding 3D depth-sensing for volumetric wound measurement."
- "Thank you. We are ready for your questions."

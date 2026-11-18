import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Smart Waste Segregation", page_icon="♻️", layout="wide")

MODEL_NAME = "yolo11n-seg.pt"
WASTE_CATEGORIES = ["Plastic", "Metal", "Paper", "Cloth", "Glass", "Organic", "Other / Unsure"]

st.title("♻️ AI-Based Smart Waste Segregation System")
st.caption("Image-based object segmentation and visible pixel-area analysis")

with st.sidebar:
    st.header("Project settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    st.info(
        "This demo uses a general pre-trained YOLO segmentation model. "
        "It detects common objects, not your custom 5 waste classes. "
        "Review the suggested category for each object before calculating results."
    )

@st.cache_resource
def load_model():
    # Ultralytics downloads this small pretrained segmentation model on first run.
    return YOLO(MODEL_NAME)

uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.markdown("""
    ### How to use
    1. Upload an image containing waste objects.
    2. The model detects objects and draws segmentation masks.
    3. Confirm or change the waste category for each detected object.
    4. View category-wise visible pixel-area percentages.
    """)
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

with st.spinner("Loading segmentation model and analysing image..."):
    try:
        model = load_model()
        result = model.predict(
            source=np.array(image),
            conf=confidence,
            imgsz=640,
            device="cpu",
            verbose=False
        )[0]
    except Exception as exc:
        st.error("Model could not run. Check your internet connection on first run and install requirements.")
        st.code(str(exc))
        st.stop()

height, width = np.array(image).shape[:2]
annotated = np.array(image).copy()
detections = []

if result.masks is not None and result.boxes is not None:
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    names = result.names

    for idx in range(len(boxes)):
        mask = masks[idx]
        # Resize model mask to the original uploaded image dimensions.
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
        mask_bool = np.array(mask_img) > 127
        pixel_count = int(mask_bool.sum())
        if pixel_count == 0:
            continue

        cls_id = int(boxes.cls[idx].item())
        conf = float(boxes.conf[idx].item())
        label = names.get(cls_id, str(cls_id))
        xyxy = boxes.xyxy[idx].cpu().numpy().astype(int).tolist()

        # Conservative suggestions only; users can change these below.
        suggested = "Other / Unsure"
        lower = label.lower()
        if lower in {"bottle"}:
            suggested = "Plastic"
        elif lower in {"book", "paper", "newspaper"}:
            suggested = "Paper"
        elif lower in {"scissors", "knife", "spoon", "fork"}:
            suggested = "Metal"

        detections.append({
            "index": len(detections),
            "label": label,
            "confidence": conf,
            "mask": mask_bool,
            "pixels": pixel_count,
            "suggested": suggested,
            "box": xyxy
        })

if not detections:
    st.warning("No segmented objects were detected. Try another image or lower the confidence threshold.")
    st.stop()

# Render segmentation overlay.
overlay = annotated.astype(np.float32)
palette = [
    (0, 220, 80), (255, 170, 0), (0, 160, 255), (220, 0, 220),
    (255, 230, 0), (0, 220, 220), (255, 80, 80), (150, 150, 255)
]
for i, det in enumerate(detections):
    color = np.array(palette[i % len(palette)], dtype=np.float32)
    mask = det["mask"]
    overlay[mask] = overlay[mask] * 0.55 + color * 0.45
    x1, y1, x2, y2 = det["box"]
    # Draw box and label using PIL to avoid an OpenCV dependency.
    from PIL import ImageDraw
    temp = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(temp)
    draw.rectangle((x1, y1, x2, y2), outline=tuple(int(v) for v in color), width=3)
    draw.text((x1, max(0, y1 - 18)), f"{det['label']} {det['confidence']:.2f}", fill=(255, 255, 255))
    overlay = np.array(temp).astype(np.float32)

with right:
    st.subheader("AI segmentation result")
    st.image(np.clip(overlay, 0, 255).astype(np.uint8), use_container_width=True)

st.divider()
st.subheader("Confirm detected waste categories")
st.caption("The model's object labels are automatic. Category suggestions are approximate; please correct them where needed.")

assignments = []
cols = st.columns(2)
for i, det in enumerate(detections):
    with cols[i % 2]:
        choice = st.selectbox(
            f"Object {i+1}: {det['label']} ({det['confidence']:.0%})",
            WASTE_CATEGORIES,
            index=WASTE_CATEGORIES.index(det["suggested"]),
            key=f"category_{i}"
        )
        assignments.append(choice)

# Build a per-pixel category map. Earlier/high-confidence detections claim overlapping pixels first.
category_map = np.full((height, width), -1, dtype=np.int16)
order = sorted(range(len(detections)), key=lambda k: detections[k]["confidence"], reverse=True)
for i in order:
    mask = detections[i]["mask"]
    category_id = WASTE_CATEGORIES.index(assignments[i])
    category_map[(mask) & (category_map == -1)] = category_id

valid_pixels = int((category_map >= 0).sum())
if valid_pixels == 0:
    st.warning("No visible mask pixels available for area calculation.")
    st.stop()

rows = []
for category_id, category in enumerate(WASTE_CATEGORIES):
    count = int((category_map == category_id).sum())
    if count:
        rows.append({
            "Waste category": category,
            "Visible area (pixels)": count,
            "Share of detected waste area (%)": round(count / valid_pixels * 100, 2),
            "Share of full image (%)": round(count / (height * width) * 100, 2)
        })

df = pd.DataFrame(rows).sort_values("Visible area (pixels)", ascending=False)
st.subheader("Category-wise visible pixel-area analysis")
m1, m2, m3 = st.columns(3)
m1.metric("Detected objects", len(detections))
m2.metric("Image resolution", f"{width} × {height}")
m3.metric("Detected waste pixels", f"{valid_pixels:,}")

st.dataframe(df, use_container_width=True, hide_index=True)
st.bar_chart(df.set_index("Waste category")["Share of detected waste area (%)"])

st.caption(
    "Note: These are visible pixel-area estimates from segmentation masks, not physical weight or volume. "
    "Overlapping masks are assigned to the higher-confidence detection to avoid double-counting pixels."
)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download area report (CSV)", csv, "waste_area_report.csv", "text/csv")

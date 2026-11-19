import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import os
from ultralytics import YOLO

SPUR_CLS  = 2
TRUNK_CLS = 3

st.set_page_config(page_title="Vineyard Pruning Detection", layout="wide")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


def get_tip_and_angle(spur_mask, trunk_centroid):
    spur_pixels = np.argwhere(spur_mask > 0).astype(float)
    if len(spur_pixels) < 5:
        return None, None
    coords = spur_pixels[:, ::-1]
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ main_axis
    tip_a = (mean + main_axis * projections.max()).astype(int)
    tip_b = (mean + main_axis * projections.min()).astype(int)
    dist_a = np.linalg.norm(tip_a - trunk_centroid)
    dist_b = np.linalg.norm(tip_b - trunk_centroid)
    tip = tuple(tip_a) if dist_a > dist_b else tuple(tip_b)
    axis_angle = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
    cut_angle = (axis_angle + 90) % 180
    return tip, cut_angle


def process_frame(img_bgr, model):
    h, w = img_bgr.shape[:2]
    results = model.predict(img_bgr, conf=0.15, iou=0.4, verbose=False)[0]

    trunk_mask = np.zeros((h, w), dtype=np.uint8)
    spur_list = []

    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            cls = int(cls)
            m = cv2.resize(mask.cpu().numpy(), (w, h))
            m = (m > 0.5).astype(np.uint8)
            if cls == TRUNK_CLS:
                trunk_mask = cv2.bitwise_or(trunk_mask, m)
            elif cls == SPUR_CLS:
                spur_list.append(m)

    if trunk_mask.sum() > 0:
        trunk_centroid = np.argwhere(trunk_mask > 0)[:, ::-1].astype(float).mean(axis=0)
    else:
        trunk_centroid = np.array([w / 2, h / 2])

    output_img = img_bgr.copy()
    overlay = output_img.copy()
    overlay[trunk_mask > 0] = [0, 200, 0]
    output_img = cv2.addWeighted(output_img, 0.75, overlay, 0.25, 0)

    scale = max(h, w) / 640
    r = max(6, int(8 * scale))
    line_len = max(25, int(35 * scale))
    thickness = max(2, int(2 * scale))
    font_scale = max(0.5, 0.5 * scale)

    pruning_data = []
    for i, spur_mask in enumerate(spur_list):
        ov = output_img.copy()
        ov[spur_mask > 0] = [0, 140, 255]
        output_img = cv2.addWeighted(output_img, 0.75, ov, 0.25, 0)
        tip, cut_angle = get_tip_and_angle(spur_mask, trunk_centroid)
        if tip is None:
            continue
        cv2.circle(output_img, tip, r, (0, 0, 255), -1)
        rad = np.radians(cut_angle)
        dx, dy = int(line_len * np.cos(rad)), int(line_len * np.sin(rad))
        cv2.line(output_img, (tip[0]-dx, tip[1]-dy), (tip[0]+dx, tip[1]+dy), (255, 0, 0), thickness)
        cv2.putText(output_img, f"{cut_angle:.0f}", (tip[0]+r, tip[1]-r),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        pruning_data.append({
            "spur_id": i,
            "x": int(tip[0]),
            "y": int(tip[1]),
            "cut_angle_deg": round(float(cut_angle), 1)
        })

    out_rgb = cv2.cvtColor(output_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    detected_classes = [model.names[int(c)] for c in results.boxes.cls]
    return out_rgb, pruning_data, detected_classes


# --- UI ---
st.title("Vineyard Pruning Point Detection")
st.markdown("Upload an image or video of a vineyard ready for pruning.")

model = load_model()

tab_img, tab_vid = st.tabs(["Image", "Video"])

with tab_img:
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting..."):
            out_rgb, data, detected = process_frame(img_bgr, model)

        st.caption(f"Model classes: {model.names} | Detected: {detected}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original")
        with col2:
            st.image(out_rgb, caption=f"Result — {len(data)} pruning points")

        if data:
            json_str = json.dumps(data, indent=2, default=lambda x: int(x))
            st.download_button("Download JSON", json_str, "pruning_data.json", "application/json")
            with st.expander("View JSON"):
                st.code(json_str, language="json")
        else:
            st.warning("No spurs detected. Check model classes above.")

with tab_vid:
    uploaded_vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.mktemp(suffix=".mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        all_data = []
        progress = st.progress(0, text="Processing video...")

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_rgb, data, _ = process_frame(frame, model)
            for d in data:
                d["frame_id"] = frame_id
            all_data.extend(data)
            writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            frame_id += 1
            if total > 0:
                progress.progress(min(frame_id / total, 1.0), text=f"Frame {frame_id}/{total}")

        cap.release()
        writer.release()
        progress.empty()

        st.video(out_path)
        json_str = json.dumps(all_data, indent=2, default=lambda x: int(x))
        st.download_button("Download JSON", json_str, "pruning_data.json", "application/json")
        st.markdown(f"**{len(all_data)} pruning points across {frame_id} frames**")
        os.unlink(tfile.name)

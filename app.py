import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# =========================================================
# 1. 기본 설정
# =========================================================
IMG_SIZE = 224
CNN_MODEL_PATH = "model/battery_model_fixed.keras"
YOLO_MODEL_PATH = "yoloModel/battery_yolo11n_best_30epochs.pt"
SAMPLE_DIR = "sample_images"
LAST_CONV_LAYER_NAME = "conv2d_2"

st.set_page_config(
    page_title="배터리 외관 품질 검사 시스템",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. 다크 HMI CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp { background: #070b14 !important; color: #e5e7eb !important; }
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { background: #070b14 !important; }
    .block-container { padding-top: 2.4rem; padding-bottom: 2.2rem; max-width: 1500px; background: #070b14 !important; }
    section[data-testid="stSidebar"] { background-color: #0b1220 !important; border-right: 1px solid #1f2937; }
    section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
    h1, h2, h3, h4, h5, h6, p, span, div, label { font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", sans-serif; }
    h1, h2, h3 { color: #f9fafb !important; }
    a { color: #38bdf8 !important; }
    div[data-testid="stMetric"] { background-color: #0f172a; border: 1px solid #1f2937; padding: 16px; border-radius: 12px; min-height: 110px; }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 13px !important; white-space: nowrap; }
    div[data-testid="stMetricValue"] { color: #f9fafb !important; font-size: 26px !important; white-space: nowrap; }
    section[data-testid="stFileUploader"] { background-color: #0f172a !important; border: 1px dashed #334155; border-radius: 14px; padding: 12px; }
    div[data-testid="stDataFrame"] { background-color: #0f172a !important; border: 1px solid #1f2937; border-radius: 12px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    .hmi-header { background: linear-gradient(90deg, #0f172a, #111827 50%, #1e293b); padding: 30px 32px; border-radius: 16px; border: 1px solid #334155; margin-bottom: 22px; box-shadow: 0 0 24px rgba(15, 23, 42, 0.75); }
    .hmi-title { color: #f9fafb; font-size: 30px; font-weight: 900; letter-spacing: -0.5px; line-height: 1.35; word-break: keep-all; margin-bottom: 8px; }
    .hmi-subtitle { color: #cbd5e1; font-size: 15px; line-height: 1.6; word-break: keep-all; }
    .status-card { background: #0f172a; padding: 18px 14px; border-radius: 14px; border: 1px solid #334155; min-height: 112px; height: 112px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; overflow: hidden; }
    .status-label { color: #94a3b8; font-size: 13px; font-weight: 700; margin-bottom: 8px; white-space: nowrap; }
    .status-value { color: #f9fafb; font-size: 22px; font-weight: 900; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
    .status-running { color: #22c55e; font-size: 22px; font-weight: 900; white-space: nowrap; }
    .status-waiting { color: #facc15; font-size: 22px; font-weight: 900; white-space: nowrap; }
    .control-panel { background: #0f172a; border: 1px solid #334155; border-radius: 16px; padding: 20px; margin-bottom: 18px; }
    .control-title { color: #f9fafb; font-size: 20px; font-weight: 900; margin-bottom: 6px; }
    .control-desc { color: #94a3b8; font-size: 14px; line-height: 1.6; }
    .pass-box { background: linear-gradient(135deg, #064e3b, #047857); color: white; padding: 34px 20px; border-radius: 18px; text-align: center; border: 3px solid #10b981; box-shadow: 0 0 26px rgba(16, 185, 129, 0.25); min-height: 190px; display: flex; flex-direction: column; justify-content: center; }
    .ng-box { background: linear-gradient(135deg, #7f1d1d, #b91c1c); color: white; padding: 34px 20px; border-radius: 18px; text-align: center; border: 3px solid #ef4444; box-shadow: 0 0 26px rgba(239, 68, 68, 0.25); min-height: 190px; display: flex; flex-direction: column; justify-content: center; }
    .result-big { font-size: 68px; font-weight: 1000; line-height: 1.0; margin-bottom: 14px; letter-spacing: 1px; }
    .result-small { font-size: 18px; font-weight: 800; line-height: 1.5; word-break: keep-all; }
    .action-box { background-color: #111827; border-left: 6px solid #64748b; padding: 16px; border-radius: 10px; margin-top: 14px; color: #e5e7eb; font-weight: 600; line-height: 1.65; }
    .roadmap-box { background-color: #0f172a; padding: 18px; border-radius: 14px; border: 1px dashed #475569; line-height: 1.9; color: #e5e7eb; min-height: 230px; }
    .notice-box { background-color: #0f172a; padding: 18px; border-radius: 14px; border: 1px solid #334155; line-height: 1.75; color: #cbd5e1; }
    .queue-box { background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 16px; text-align: center; min-height: 108px; }
    .queue-label { color: #94a3b8; font-size: 13px; font-weight: 700; margin-bottom: 8px; }
    .queue-value { color: #f9fafb; font-size: 26px; font-weight: 900; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. 모델 로드
# =========================================================
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(CNN_MODEL_PATH):
        return None
    return tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)


@st.cache_resource
def load_yolo_model():
    if YOLO is None:
        return None
    if not os.path.exists(YOLO_MODEL_PATH):
        return None
    return YOLO(YOLO_MODEL_PATH)


cnn_model = load_cnn_model()
yolo_model = load_yolo_model()

# =========================================================
# 4. 세션 상태 초기화
# =========================================================
if "inspection_logs" not in st.session_state:
    st.session_state.inspection_logs = []
if "inspection_count" not in st.session_state:
    st.session_state.inspection_count = 0
if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0
if "current_image_path" not in st.session_state:
    st.session_state.current_image_path = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "자동 검사"

# =========================================================
# 5. 이미지 목록 / 로딩 / 전처리
# =========================================================
def get_sample_images():
    if not os.path.exists(SAMPLE_DIR):
        return []
    files = [
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return sorted(files)


def preprocess_image(image):
    original_image = np.array(image.convert("RGB"))
    resized_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))
    normalized_image = resized_image.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized_image, axis=0)
    return original_image, normalized_image, input_tensor


def load_image_from_path(image_path):
    image = Image.open(image_path)
    return preprocess_image(image)


def load_image_from_upload(uploaded_file):
    image = Image.open(uploaded_file)
    return preprocess_image(image)


sample_files = get_sample_images()

# =========================================================
# 6. Grad-CAM 함수
# =========================================================
def make_gradcam_heatmap_binary(img_array, model, last_conv_layer_name="conv2d_2", target_class="defect"):
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    last_conv_output = None

    for layer in model.layers:
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    if last_conv_output is None:
        raise ValueError(f"{last_conv_layer_name} 레이어를 찾지 못했습니다.")

    grad_model = tf.keras.Model(inputs=inputs, outputs=[last_conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        defect_score = predictions[:, 0]
        if target_class == "defect":
            target_score = defect_score
        elif target_class == "normal":
            target_score = 1.0 - defect_score
        else:
            raise ValueError("target_class는 'normal' 또는 'defect'만 가능합니다.")

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        return heatmap.numpy()
    return (heatmap / max_value).numpy()


def create_gradcam_overlay(image_array, heatmap, alpha=0.42):
    img = np.uint8(image_array * 255)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return heatmap_resized, overlay

# =========================================================
# 7. YOLO Detection 함수
# =========================================================
def run_yolo_detection(original_image, conf_threshold=0.25):
    """원본 RGB 이미지에 YOLO 탐지를 수행하고, 박스가 그려진 RGB 이미지와 결과 DataFrame을 반환."""
    if yolo_model is None:
        return None, pd.DataFrame(), 0, "탐지 모델 없음"

    results = yolo_model.predict(
        source=original_image,
        conf=conf_threshold,
        verbose=False,
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    rows = []
    if result.boxes is not None and len(result.boxes) > 0:
        names = result.names
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for i, (xyxy, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, classes), start=1):
            x1, y1, x2, y2 = xyxy
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            rows.append(
                {
                    "No": i,
                    "불량 유형": class_name,
                    "Confidence": f"{conf * 100:.2f}%",
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            )

    detection_df = pd.DataFrame(rows)
    detection_count = len(rows)
    detection_summary = "탐지 없음" if detection_count == 0 else ", ".join(detection_df["불량 유형"].astype(str).tolist())
    return annotated_rgb, detection_df, detection_count, detection_summary

# =========================================================
# 8. 통합 검사 실행 함수
# =========================================================
def run_inspection(
    original_image,
    image_norm,
    input_tensor,
    image_name,
    threshold,
    yolo_conf_threshold,
    use_yolo,
    line_name,
    inspection_mode,
    source_mode,
):
    if cnn_model is None:
        return None

    pred_prob = float(cnn_model.predict(input_tensor, verbose=0)[0][0])
    pass_score = 1.0 - pred_prob
    pred_label = "defect" if pred_prob >= threshold else "normal"
    cnn_result = "NG" if pred_label == "defect" else "PASS"

    yolo_annotated = None
    yolo_df = pd.DataFrame()
    yolo_count = 0
    yolo_summary = "미실행"

    if use_yolo:
        yolo_annotated, yolo_df, yolo_count, yolo_summary = run_yolo_detection(
            original_image=original_image,
            conf_threshold=yolo_conf_threshold,
        )

    # 최종 판정 정책: CNN이 NG이거나 YOLO가 1개 이상 탐지하면 최종 NG
    final_result = "NG" if (cnn_result == "NG" or yolo_count > 0) else "PASS"

    if final_result == "NG":
        action = "재검사 필요"
        if yolo_count > 0:
            action_kr = f"YOLO가 불량 위치를 {yolo_count}건 탐지했습니다. 작업자 확인 또는 재검사가 필요합니다."
        else:
            action_kr = "CNN이 불량 의심 제품으로 분류했습니다. 작업자 확인 또는 재검사가 필요합니다."
    else:
        action = "다음 공정 이동"
        action_kr = "정상 제품으로 분류되었습니다. 다음 공정으로 이동 가능합니다."

    inspection_id = datetime.now().strftime("INSP-%Y%m%d-%H%M%S")
    st.session_state.inspection_count += 1

    log = {
        "검사 ID": inspection_id,
        "검사 시간": datetime.now().strftime("%H:%M:%S"),
        "투입 방식": source_mode,
        "이미지 파일": image_name,
        "최종 판정": final_result,
        "CNN 판정": cnn_result,
        "CNN NG Score": f"{pred_prob * 100:.2f}%",
        "YOLO 탐지 수": yolo_count,
        "YOLO 탐지 유형": yolo_summary,
        "기준값": f"{threshold:.2f}",
        "조치": action,
    }

    st.session_state.inspection_logs.insert(0, log)
    st.session_state.inspection_logs = st.session_state.inspection_logs[:30]

    return {
        "inspection_id": inspection_id,
        "image_name": image_name,
        "original_image": original_image,
        "image_norm": image_norm,
        "input_tensor": input_tensor,
        "pred_prob": pred_prob,
        "pass_score": pass_score,
        "cnn_result": cnn_result,
        "final_result": final_result,
        "action": action,
        "action_kr": action_kr,
        "line_name": line_name,
        "inspection_mode": inspection_mode,
        "threshold": threshold,
        "source_mode": source_mode,
        "yolo_annotated": yolo_annotated,
        "yolo_df": yolo_df,
        "yolo_count": yolo_count,
        "yolo_summary": yolo_summary,
        "yolo_conf_threshold": yolo_conf_threshold,
        "use_yolo": use_yolo,
    }

# =========================================================
# 9. Sidebar
# =========================================================
with st.sidebar:
    st.header("🔧 검사 조건 설정")

    st.markdown("### 라인 정보")
    line_name = st.selectbox(
        "검사 라인",
        ["배터리 셀 검사 라인 A", "배터리 셀 검사 라인 B", "파일럿 검사 라인"],
        index=0,
    )

    inspection_mode = st.selectbox(
        "검사 모드",
        ["불량 검출 우선", "균형 판정", "정상 과검출 최소화"],
        index=0,
    )

    threshold = st.slider(
        "CNN NG 판정 기준값",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    st.caption("기준값이 낮을수록 NG를 민감하게 검출하고, 높을수록 정상 과검출을 줄입니다.")

    st.markdown("---")
    st.markdown("### YOLO Detection 설정")
    use_yolo = st.checkbox("YOLO Detection 함께 실행", value=True)
    yolo_conf_threshold = st.slider(
        "YOLO Confidence 기준값",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )
    st.write("YOLO 모델:")
    st.code(YOLO_MODEL_PATH)
    if YOLO is None:
        st.error("ultralytics가 설치되어 있지 않습니다. pip install ultralytics 필요")
    elif yolo_model is None:
        st.warning("YOLO 모델 파일을 찾지 못했습니다.")
    else:
        st.success("YOLO 모델 연결 완료")
        try:
            st.caption(f"클래스: {yolo_model.names}")
        except Exception:
            pass

    st.markdown("---")
    st.markdown("### CNN 모델 설정")
    st.write("검사 모델: CNN Vision V1")
    st.write("입력 크기: 224 × 224")
    st.write("판정 방식: PASS / NG 분류")
    st.write("검사 대상: 원통형 배터리 셀")
    if cnn_model is None:
        st.warning("CNN 모델 파일을 찾지 못했습니다.")
    else:
        st.success("CNN 모델 연결 완료")

    st.markdown("---")
    if st.button("검사 로그 초기화"):
        st.session_state.inspection_logs = []
        st.session_state.inspection_count = 0
        st.session_state.sample_index = 0
        st.success("검사 로그가 초기화되었습니다.")

# =========================================================
# 10. Header / Status Bar
# =========================================================
st.markdown(
    """
    <div class="hmi-header">
        <div class="hmi-title">🔋 딥러닝 기반 배터리 외관 품질 검사 시스템</div>
        <div class="hmi-subtitle">
            CNN 기반 PASS / NG 판정과 YOLO 기반 Damaged / Pollution 위치 탐지를 함께 수행하는 AI 비전 검사 대시보드
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

system_status = "운영 중" if cnn_model is not None else "모델 대기"
yolo_status = "연결 완료" if yolo_model is not None else "대기"
status_class = "status-running" if cnn_model is not None else "status-waiting"
yolo_status_class = "status-running" if yolo_model is not None else "status-waiting"

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.markdown(f"""<div class="status-card"><div class="status-label">시스템 상태</div><div class="{status_class}">{system_status}</div></div>""", unsafe_allow_html=True)
with status_col2:
    st.markdown("""<div class="status-card"><div class="status-label">CNN 모델</div><div class="status-value">PASS / NG</div></div>""", unsafe_allow_html=True)
with status_col3:
    st.markdown(f"""<div class="status-card"><div class="status-label">YOLO 모델</div><div class="{yolo_status_class}">{yolo_status}</div></div>""", unsafe_allow_html=True)
with status_col4:
    st.markdown(f"""<div class="status-card"><div class="status-label">누적 검사 수</div><div class="status-value">{st.session_state.inspection_count}건</div></div>""", unsafe_allow_html=True)

st.markdown("---")

if cnn_model is None:
    st.warning(f"CNN 모델 파일이 없습니다. `{CNN_MODEL_PATH}` 파일을 확인하세요.")
if use_yolo and yolo_model is None:
    st.warning(f"YOLO 모델 파일이 없거나 ultralytics가 설치되지 않았습니다. `{YOLO_MODEL_PATH}` 파일과 패키지를 확인하세요.")

# =========================================================
# 11. 자동 검사 제어 패널
# =========================================================
st.subheader("🏭 자동 검사 라인 시뮬레이션")
st.markdown(
    """
    <div class="control-panel">
        <div class="control-title">검사 이미지 자동 투입 모드</div>
        <div class="control-desc">
            실제 카메라 또는 검사기 입력을 대신하여 <b>sample_images</b> 폴더의 이미지를 검사 대기열로 사용합니다.<br>
            <b>다음 제품 검사</b> 버튼을 누르면 CNN 판정과 YOLO Detection이 함께 실행됩니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

queue_col1, queue_col2, queue_col3, queue_col4 = st.columns(4)
with queue_col1:
    st.markdown(f"""<div class="queue-box"><div class="queue-label">대기 이미지 수</div><div class="queue-value">{len(sample_files)}장</div></div>""", unsafe_allow_html=True)
with queue_col2:
    current_no = st.session_state.sample_index + 1 if len(sample_files) > 0 else 0
    st.markdown(f"""<div class="queue-box"><div class="queue-label">현재 대기 순번</div><div class="queue-value">{current_no}</div></div>""", unsafe_allow_html=True)
with queue_col3:
    st.markdown(f"""<div class="queue-box"><div class="queue-label">누적 검사 수</div><div class="queue-value">{st.session_state.inspection_count}건</div></div>""", unsafe_allow_html=True)
with queue_col4:
    st.markdown("""<div class="queue-box"><div class="queue-label">검사 방식</div><div class="queue-value">CNN+YOLO</div></div>""", unsafe_allow_html=True)

st.write("")
auto_col1, auto_col2, auto_col3 = st.columns([1, 1, 2])
with auto_col1:
    next_clicked = st.button("▶ 다음 제품 검사", use_container_width=True)
with auto_col2:
    repeat_clicked = st.button("↻ 현재 제품 재검사", use_container_width=True)
with auto_col3:
    st.caption("자동 검사 모드는 sample_images 폴더의 이미지를 순차적으로 불러옵니다.")

inspection_result_data = None

if next_clicked:
    if len(sample_files) == 0:
        st.error("sample_images 폴더에 검사 이미지가 없습니다. 샘플 이미지를 먼저 넣어주세요.")
    else:
        current_path = sample_files[st.session_state.sample_index]
        st.session_state.current_image_path = current_path
        st.session_state.current_mode = "자동 검사"
        st.session_state.sample_index = (st.session_state.sample_index + 1) % len(sample_files)

        original_image, image_norm, input_tensor = load_image_from_path(current_path)
        image_name = os.path.basename(current_path)

        inspection_result_data = run_inspection(
            original_image=original_image,
            image_norm=image_norm,
            input_tensor=input_tensor,
            image_name=image_name,
            threshold=threshold,
            yolo_conf_threshold=yolo_conf_threshold,
            use_yolo=use_yolo,
            line_name=line_name,
            inspection_mode=inspection_mode,
            source_mode="자동 투입",
        )

if repeat_clicked:
    if st.session_state.current_image_path is None:
        st.warning("재검사할 현재 제품이 없습니다. 먼저 다음 제품 검사를 실행하세요.")
    else:
        current_path = st.session_state.current_image_path
        st.session_state.current_mode = "자동 재검사"

        original_image, image_norm, input_tensor = load_image_from_path(current_path)
        image_name = os.path.basename(current_path)

        inspection_result_data = run_inspection(
            original_image=original_image,
            image_norm=image_norm,
            input_tensor=input_tensor,
            image_name=image_name,
            threshold=threshold,
            yolo_conf_threshold=yolo_conf_threshold,
            use_yolo=use_yolo,
            line_name=line_name,
            inspection_mode=inspection_mode,
            source_mode="자동 재검사",
        )

# =========================================================
# 12. 수동 검사 모드
# =========================================================
with st.expander("🧪 수동 검사 모드 열기"):
    st.caption("자동 검사 대기열 외의 이미지를 별도로 확인할 때 사용하는 모드입니다.")
    uploaded_file = st.file_uploader("수동으로 검사할 배터리 셀 이미지를 선택하세요.", type=["png", "jpg", "jpeg"])
    manual_clicked = st.button("수동 이미지 검사 실행", use_container_width=True)

    if uploaded_file is not None and manual_clicked:
        st.session_state.current_mode = "수동 검사"
        original_image, image_norm, input_tensor = load_image_from_upload(uploaded_file)

        inspection_result_data = run_inspection(
            original_image=original_image,
            image_norm=image_norm,
            input_tensor=input_tensor,
            image_name=uploaded_file.name,
            threshold=threshold,
            yolo_conf_threshold=yolo_conf_threshold,
            use_yolo=use_yolo,
            line_name=line_name,
            inspection_mode=inspection_mode,
            source_mode="수동 검사",
        )

# =========================================================
# 13. 검사 결과 표시
# =========================================================
if inspection_result_data is not None:
    st.markdown("---")
    st.subheader("🧪 검사 판정 결과")

    original_image = inspection_result_data["original_image"]
    image_norm = inspection_result_data["image_norm"]
    input_tensor = inspection_result_data["input_tensor"]
    final_result = inspection_result_data["final_result"]
    cnn_result = inspection_result_data["cnn_result"]
    pred_prob = inspection_result_data["pred_prob"]
    pass_score = inspection_result_data["pass_score"]
    inspection_id = inspection_result_data["inspection_id"]
    image_name = inspection_result_data["image_name"]
    action = inspection_result_data["action"]
    action_kr = inspection_result_data["action_kr"]
    source_mode = inspection_result_data["source_mode"]
    yolo_annotated = inspection_result_data["yolo_annotated"]
    yolo_df = inspection_result_data["yolo_df"]
    yolo_count = inspection_result_data["yolo_count"]
    yolo_summary = inspection_result_data["yolo_summary"]

    left_col, right_col = st.columns([1.15, 1])
    with left_col:
        st.markdown("**투입 이미지**")
        st.image(original_image, caption=f"검사 ID: {inspection_id} | 파일: {image_name}", use_container_width=True)

    with right_col:
        st.markdown("**최종 AI 판정**")
        if final_result == "NG":
            st.markdown("""<div class="ng-box"><div class="result-big">NG</div><div class="result-small">불량 의심 — 작업자 확인 필요</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="pass-box"><div class="result-big">PASS</div><div class="result-small">정상 통과 가능</div></div>""", unsafe_allow_html=True)

        st.write("")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("CNN NG Score", f"{pred_prob * 100:.2f}%")
        with metric_col2:
            st.metric("CNN PASS Score", f"{pass_score * 100:.2f}%")
        with metric_col3:
            st.metric("YOLO 탐지 수", f"{yolo_count}건")

        st.progress(min(max(pred_prob, 0.0), 1.0))
        st.markdown(f"""<div class="action-box"><b>조치:</b> {action}<br><span>{action_kr}</span></div>""", unsafe_allow_html=True)

        st.write(f"CNN 판정: `{cnn_result}`")
        st.write(f"YOLO 탐지 유형: `{yolo_summary}`")
        st.write(f"투입 방식: `{source_mode}`")
        st.write(f"검사 라인: `{line_name}`")
        st.write(f"검사 모드: `{inspection_mode}`")
        st.write(f"CNN 판정 기준값: `{threshold:.2f}`")
        st.write(f"YOLO Confidence 기준값: `{yolo_conf_threshold:.2f}`")

    # YOLO Detection 결과
    st.markdown("---")
    st.subheader("📦 YOLO Detection 결과")
    if use_yolo and yolo_model is not None:
        yolo_col1, yolo_col2 = st.columns([1.15, 1])
        with yolo_col1:
            if yolo_annotated is not None:
                st.image(yolo_annotated, caption="YOLO Bounding Box 탐지 결과", use_container_width=True)
            else:
                st.info("YOLO 시각화 결과가 없습니다.")
        with yolo_col2:
            st.markdown("**탐지 상세**")
            if len(yolo_df) > 0:
                st.dataframe(yolo_df, use_container_width=True, hide_index=True)
            else:
                st.success("YOLO 기준으로 탐지된 불량 위치가 없습니다.")
    else:
        st.info("YOLO Detection이 꺼져 있거나 모델이 연결되지 않았습니다.")

    # Grad-CAM
    if cnn_model is not None:
        st.markdown("---")
        st.subheader("🔥 CNN 주목 영역 시각화")
        target_class = "defect" if cnn_result == "NG" else "normal"
        try:
            heatmap = make_gradcam_heatmap_binary(
                input_tensor,
                cnn_model,
                last_conv_layer_name=LAST_CONV_LAYER_NAME,
                target_class=target_class,
            )
            heatmap_resized, overlay = create_gradcam_overlay(image_norm, heatmap, alpha=0.42)
            cam_col1, cam_col2, cam_col3 = st.columns(3)
            with cam_col1:
                st.image(image_norm, caption="투입 이미지", use_container_width=True)
            with cam_col2:
                st.image(heatmap_resized, caption="CNN Heatmap", use_container_width=True)
            with cam_col3:
                st.image(overlay, caption="CNN 판정 근거 Overlay", use_container_width=True)
            st.caption("Grad-CAM은 CNN이 PASS/NG 판정 시 상대적으로 크게 참고한 영역입니다. 실제 위치 검출은 YOLO Bounding Box를 기준으로 해석하면 됩니다.")
        except Exception as e:
            st.error("AI 주목 영역 시각화 중 오류가 발생했습니다.")
            st.code(str(e))
else:
    st.info("자동 검사 버튼을 누르면 sample_images 폴더의 이미지가 검사 라인에 투입됩니다.")

# =========================================================
# 14. 품질 검사 KPI
# =========================================================
st.markdown("---")
st.subheader("📊 품질 검사 KPI")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric("CNN 검증 정확도", "98.45%")
    st.caption("검증 데이터 기준 전체 판정 정확도")
with kpi_col2:
    st.metric("YOLO 탐지 대상", "Damaged / Pollution")
    st.caption("Bounding Box 기반 세부 불량 위치 탐지")
with kpi_col3:
    st.metric("최종 판정 정책", "CNN 또는 YOLO NG")
    st.caption("둘 중 하나라도 불량이면 최종 NG")
with kpi_col4:
    st.metric("검사 정책", "불량 검출 우선")
    st.caption("불량 유출 방지를 우선하는 판정 기준")

# =========================================================
# 15. 검사 이력 로그
# =========================================================
st.markdown("---")
st.subheader("🧾 검사 이력 로그")
if len(st.session_state.inspection_logs) > 0:
    log_df = pd.DataFrame(st.session_state.inspection_logs)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.caption("아직 검사 이력이 없습니다. 이 로그는 별도 DB 없이 현재 세션에서만 임시 저장됩니다.")

# =========================================================
# 16. 시스템 모듈 구성
# =========================================================
st.markdown("---")
st.subheader("🧩 시스템 모듈 구성")
roadmap_col1, roadmap_col2 = st.columns(2)
with roadmap_col1:
    st.markdown(
        """
        <div class="roadmap-box">
            <b>현재 적용 모듈</b><br><br>
            ✅ 자동 검사 이미지 투입 시뮬레이션<br>
            ✅ CNN 기반 PASS / NG 판정<br>
            ✅ CNN NG Score 산출<br>
            ✅ Grad-CAM 주목 영역 시각화<br>
            ✅ YOLO 기반 Damaged / Pollution 위치 탐지<br>
            ✅ Bounding Box 시각화<br>
            ✅ 검사 이력 로그 기록
        </div>
        """,
        unsafe_allow_html=True,
    )
with roadmap_col2:
    st.markdown(
        """
        <div class="roadmap-box">
            <b>향후 고도화 모듈</b><br><br>
            🔄 실제 카메라 입력 연동<br>
            🔄 검사 결과 DB 저장<br>
            🔄 불량 유형별 통계 대시보드<br>
            🔄 FP / FN 사례 분석 화면<br>
            🔄 검사 리포트 자동 생성
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 17. Footer
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div class="notice-box">
    본 시스템은 배터리 셀 외관 이미지를 기반으로 PASS / NG 판정과 불량 위치 탐지를 수행하는
    딥러닝 기반 품질 검사 프로토타입입니다.<br>
    CNN 모델은 제품 단위의 정상/불량 판정을 수행하고, YOLO 모델은 Damaged, Pollution 등 세부 불량 위치를 Bounding Box로 표시합니다.<br>
    현재 버전은 실제 카메라 입력을 대신하여 <b>sample_images</b> 폴더를 검사 이미지 대기열로 사용하며,
    검사 이력은 현재 Streamlit 세션에서 임시 관리됩니다.
    </div>
    """,
    unsafe_allow_html=True,
)

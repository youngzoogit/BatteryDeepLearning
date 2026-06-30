# -*- coding: utf-8 -*-
"""
====================================================================
  JSON -> YOLO .txt 변환 + data.yaml 생성 + model.val() 실행
====================================================================
  1) battery_label/{training,validation}/label/*.json  ->  YOLO .txt
  2) data.yaml 생성
  3) model.val() 실행 -> mAP@0.5, mAP@0.5:0.95 등 출력
  4) 결과 시각화 (한글 폰트 적용)
====================================================================
"""
import json
import os
import glob
import shutil
import numpy as np
from pathlib import Path

# ==================================================================
# 0. 한글 폰트 설정
# ==================================================================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ==================================================================
# 1. 경로 설정
# ==================================================================
ROOT = Path("d:/DeepLearning")
LABEL_ROOT = ROOT / "dataset" / "battery_label"
DATASET_ROOT = ROOT / "dataset" / "battery_subset_binary_train10000"
IMAGE_DIR = DATASET_ROOT / "images" / "defect"   # 불량 이미지 폴더
OUTPUT_DIR = ROOT / "yolo_eval"
OUTPUT_DIR.mkdir(exist_ok=True)

# 클래스 매핑: JSON 라벨의 name -> YOLO class id
CLASS_MAP = {"Damaged": 0, "Pollution": 1}
CLASS_NAMES = ["Damaged", "Pollution"]

# ==================================================================
# 2. JSON -> YOLO .txt 변환 함수
# ==================================================================
def polygon_to_bbox(points, img_w, img_h):
    """
    폴리곤 좌표 리스트 [x1,y1,x2,y2,...] -> YOLO 형식 (x_center, y_center, w, h) 정규화
    """
    xs = points[0::2]  # 짝수 인덱스 = x
    ys = points[1::2]  # 홀수 인덱스 = y
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # YOLO 정규화 좌표 (0~1)
    x_center = ((x_min + x_max) / 2.0) / img_w
    y_center = ((y_min + y_max) / 2.0) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    
    # 클리핑 (0~1 범위)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.001, min(1.0, w))
    h = max(0.001, min(1.0, h))
    
    return x_center, y_center, w, h


def convert_json_to_yolo(json_dir, output_label_dir):
    """
    JSON 라벨 파일들을 YOLO .txt 형식으로 변환
    """
    os.makedirs(output_label_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    converted = 0
    skipped = 0
    
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        img_info = data.get("image_info", {})
        img_w = img_info.get("width", 1920)
        img_h = img_info.get("height", 1080)
        file_name = img_info.get("file_name", "")
        
        defects = data.get("defects", None) or []
        
        # 파일명에서 확장자 제거 -> .txt
        base_name = os.path.splitext(os.path.basename(jf))[0]
        txt_path = os.path.join(output_label_dir, f"{base_name}.txt")
        
        lines = []
        for defect in defects:
            name = defect.get("name", "")
            points = defect.get("points", [])
            
            if name not in CLASS_MAP:
                continue
            if len(points) < 4:  # 최소 2개 점 (4개 좌표값)
                continue
            
            class_id = CLASS_MAP[name]
            x_c, y_c, w, h = polygon_to_bbox(points, img_w, img_h)
            lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
        # normal 이미지면 빈 파일 생성 (라벨 없음)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        converted += 1
    
    return converted

# ==================================================================
# 3. 변환 실행
# ==================================================================
# YOLO는 images/ 옆에 labels/ 폴더를 기대함
# images/defect/  <->  labels/defect/
YOLO_LABEL_DIR = DATASET_ROOT / "labels" / "defect"
YOLO_LABEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  JSON -> YOLO .txt 라벨 변환")
print("=" * 60)

# validation 라벨 변환
val_json_dir = LABEL_ROOT / "validation" / "label"
val_count = convert_json_to_yolo(str(val_json_dir), str(YOLO_LABEL_DIR))
print(f"  validation 변환 완료: {val_count}개 파일")

# training 라벨 변환
train_json_dir = LABEL_ROOT / "training" / "label"
train_count = convert_json_to_yolo(str(train_json_dir), str(YOLO_LABEL_DIR))
print(f"  training 변환 완료: {train_count}개 파일")

print(f"\n  총 {val_count + train_count}개 YOLO .txt 파일 -> {YOLO_LABEL_DIR}")

# 변환된 .txt 파일 중 이미지와 매칭되는 것 확인
txt_files = list(YOLO_LABEL_DIR.glob("*.txt"))
img_files = list(IMAGE_DIR.glob("*.png"))
img_names = {f.stem for f in img_files}
matched = [f for f in txt_files if f.stem in img_names]
print(f"  이미지와 매칭된 라벨: {len(matched)}개 / 전체 이미지: {len(img_files)}개")

# ==================================================================
# 4. data.yaml 생성
# ==================================================================
data_yaml_path = OUTPUT_DIR / "data.yaml"
yaml_content = f"""path: {DATASET_ROOT}
train: images/defect
val: images/defect
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n  data.yaml 생성 완료 -> {data_yaml_path}")

# ==================================================================
# 5. YOLO model.val() 실행
# ==================================================================
print("\n" + "=" * 60)
print("  YOLO11n 모델 검증 (model.val()) 실행 중...")
print("=" * 60)

from ultralytics import YOLO

MODEL_PATH = ROOT / "yoloModel" / "battery_yolo11n_best_30epochs.pt"
model = YOLO(str(MODEL_PATH))

# val 실행
results = model.val(
    data=str(data_yaml_path),
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.6,
    save_json=False,
    project=str(OUTPUT_DIR),
    name="val_results",
    exist_ok=True,
    workers=0,
)

# ==================================================================
# 6. 결과 출력 + 저장
# ==================================================================
print("\n" + "=" * 60)
print("  YOLO11n 성능 평가 결과")
print("=" * 60)

# results.box 에서 지표 추출
box = results.box
mp = float(box.mp)    # mean precision
mr = float(box.mr)    # mean recall
map50 = float(box.map50)   # mAP@0.5
map50_95 = float(box.map)  # mAP@0.5:0.95

print(f"  Precision (평균)     : {mp:.4f}")
print(f"  Recall (평균)        : {mr:.4f}")
print(f"  mAP@0.5             : {map50:.4f}")
print(f"  mAP@0.5:0.95        : {map50_95:.4f}")

# 클래스별 AP도 출력
if hasattr(box, 'ap_class_index') and box.ap_class_index is not None:
    print(f"\n  {'클래스':<15} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>13}")
    print("  " + "-" * 60)
    for i, cls_idx in enumerate(box.ap_class_index):
        cls_name = CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else f"class_{cls_idx}"
        p_val = float(box.p[i])
        r_val = float(box.r[i])
        ap50_val = float(box.ap50[i])
        ap_val = float(box.ap[i])
        print(f"  {cls_name:<15} {p_val:>10.4f} {r_val:>10.4f} {ap50_val:>10.4f} {ap_val:>13.4f}")

# 결과를 텍스트 파일로도 저장
metrics_path = OUTPUT_DIR / "yolo_metrics.txt"
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("YOLO11n Battery Defect Detection - Validation Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Precision (mean)  : {mp:.4f}\n")
    f.write(f"Recall (mean)     : {mr:.4f}\n")
    f.write(f"mAP@0.5           : {map50:.4f}\n")
    f.write(f"mAP@0.5:0.95      : {map50_95:.4f}\n")
    f.write("=" * 60 + "\n")
    if hasattr(box, 'ap_class_index') and box.ap_class_index is not None:
        for i, cls_idx in enumerate(box.ap_class_index):
            cls_name = CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else f"class_{cls_idx}"
            f.write(f"{cls_name}: P={float(box.p[i]):.4f} R={float(box.r[i]):.4f} mAP50={float(box.ap50[i]):.4f} mAP50-95={float(box.ap[i]):.4f}\n")

print(f"\n  metrics 저장 완료 -> {metrics_path}")

# ==================================================================
# 7. 시각화 (한글 적용)
# ==================================================================
import seaborn as sns

# -- 결과 요약 바 차트 --
fig, ax = plt.subplots(figsize=(8, 5))
metrics_names = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
metrics_values = [mp, mr, map50, map50_95]
colors = ["#3b82f6", "#22c55e", "#f97316", "#ef4444"]
bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_title("YOLO11n 배터리 불량 탐지 - 성능 지표", fontsize=14, fontweight="bold")
ax.set_ylabel("점수", fontsize=12)
ax.set_facecolor("#f8fafc")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
chart_path = OUTPUT_DIR / "yolo_metrics_chart.png"
fig.savefig(chart_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  metrics 차트 저장 -> {chart_path}")

print("\n" + "=" * 60)
print("  YOLO 성능 검증 완료!")
print("=" * 60)
print(f"  결과 폴더: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.rglob("*")):
    if f.is_file():
        print(f"    {f.relative_to(OUTPUT_DIR)}")

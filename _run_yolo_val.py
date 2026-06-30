import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from ultralytics import YOLO

def main():
    # 0. 한글 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    ROOT = Path("d:/DeepLearning")
    OUTPUT_DIR = ROOT / "yolo_eval"
    data_yaml_path = OUTPUT_DIR / "data.yaml"
    MODEL_PATH = ROOT / "yoloModel" / "battery_yolo11n_best_30epochs.pt"
    CLASS_NAMES = ["Damaged", "Pollution"]
    
    print("=" * 60)
    print("  YOLO11n 모델 검증 (model.val()) 실행 중...")
    print("=" * 60)

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
        workers=0,  # 윈도우 멀티프로세싱 에러 방지
    )

    print("\n" + "=" * 60)
    print("  YOLO11n 성능 평가 결과")
    print("=" * 60)

    box = results.box
    mp = float(box.mp)
    mr = float(box.mr)
    map50 = float(box.map50)
    map50_95 = float(box.map)

    print(f"  Precision (평균)     : {mp:.4f}")
    print(f"  Recall (평균)        : {mr:.4f}")
    print(f"  mAP@0.5             : {map50:.4f}")
    print(f"  mAP@0.5:0.95        : {map50_95:.4f}")

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

    metrics_path = OUTPUT_DIR / "yolo_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("YOLO11n Battery Defect Detection - Validation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Precision (mean)  : {mp:.4f}\n")
        f.write(f"Recall (mean)     : {mr:.4f}\n")
        f.write(f"mAP@0.5           : {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95      : {map50_95:.4f}\n")
        
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

if __name__ == '__main__':
    main()

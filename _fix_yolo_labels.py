import os
from pathlib import Path

def main():
    root = Path("d:/DeepLearning/dataset/battery_subset_binary_train10000")
    img_dir = root / "images" / "defect"
    lbl_dir = root / "labels" / "defect"

    if not img_dir.exists() or not lbl_dir.exists():
        print("Directory not found")
        return

    # 이미지 파일 리스트 확보
    images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    print(f"Total images: {len(images)}")

    rename_count = 0
    missing_labels = 0

    for img_path in images:
        img_name = img_path.stem  # e.g., Damaged_000017_RGB_cell_cylindrical_0408_029
        
        # 원본 라벨 이름 찾기 (Damaged_000017_ 부분 제거)
        # 이미지 이름에 RGB_가 포함되어 있다면 그 이후를 취함
        if "RGB_cell_cylindrical" in img_name:
            original_stem = img_name[img_name.find("RGB_cell_cylindrical"):]
            
            old_label_path = lbl_dir / f"{original_stem}.txt"
            new_label_path = lbl_dir / f"{img_name}.txt"
            
            if new_label_path.exists():
                continue # 이미 변환됨
            
            if old_label_path.exists():
                old_label_path.rename(new_label_path)
                rename_count += 1
            else:
                missing_labels += 1

    print(f"Renamed {rename_count} labels to match image names.")
    print(f"Missing labels for {missing_labels} images.")

if __name__ == '__main__':
    main()

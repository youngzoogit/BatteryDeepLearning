import zipfile
import json
import shutil
from pathlib import Path

ORIGINAL_MODEL = "battery_cnn_best.keras"       # 네 모델 파일명으로 수정
FIXED_MODEL = "battery_model_fixed.keras"    # 새로 저장될 모델명

original_path = Path(ORIGINAL_MODEL)
fixed_path = Path(FIXED_MODEL)

temp_dir = Path("temp_keras_model")

if temp_dir.exists():
    shutil.rmtree(temp_dir)

temp_dir.mkdir()

# 1. .keras 파일 압축 해제
with zipfile.ZipFile(original_path, "r") as z:
    z.extractall(temp_dir)

config_path = temp_dir / "config.json"

# 2. config.json 읽기
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 3. quantization_config 제거 함수
def remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            remove_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)

remove_quantization_config(config)

# 4. 수정된 config.json 저장
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f)

# 5. 다시 .keras 파일로 압축
with zipfile.ZipFile(fixed_path, "w", zipfile.ZIP_DEFLATED) as z:
    for file in temp_dir.rglob("*"):
        z.write(file, file.relative_to(temp_dir))

shutil.rmtree(temp_dir)

print(f"수정 완료: {fixed_path}")
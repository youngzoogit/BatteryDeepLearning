import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.join(os.getcwd(), 'dataset', 'battery_subset_binary_train10000')
VALID_CSV = os.path.join(ROOT, 'valid.csv')

val_df = pd.read_csv(VALID_CSV)
val_df['new_relative_path'] = val_df['new_relative_path'].str.replace('\\', '/', regex=False)
val_df['image_path'] = val_df['new_relative_path'].apply(lambda x: os.path.join(ROOT, x))
val_df = val_df[val_df['image_path'].apply(lambda p: os.path.exists(p) and os.path.getsize(p) > 0)].copy()
val_df['label_num'] = val_df['binary_label'].map({'normal': 0, 'defect': 1})

image_paths = val_df['image_path'].tolist()
true_labels = val_df['label_num'].tolist()

IMG_SIZE = 224
BATCH_SIZE = 32

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

def load_and_preprocess_cnn(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_and_preprocess_resnet(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = resnet_preprocess(img)
    return img

def load_and_preprocess_effnet(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = effnet_preprocess(img)
    return img

models_info = {
    'CNN': ('battery_cnn_local_best.keras', load_and_preprocess_cnn),
    'ResNet': ('battery_resnet_finetune_best.keras', load_and_preprocess_resnet),
    'EfficientNet': ('battery_efficientnet_finetune_best.keras', load_and_preprocess_effnet)
}

output_dir = os.path.abspath('confusion_matrices')
os.makedirs(output_dir, exist_ok=True)

for name, (filename, preprocess_func) in models_info.items():
    print(f'--- Processing {name} model ---')
    model_path = os.path.join('model', filename)
    if not os.path.exists(model_path):
        if name == 'EfficientNet':
            model_path = os.path.join('model', 'battery_efficientnet_best.keras')
        elif name == 'ResNet':
            model_path = os.path.join('model', 'battery_resnet_stage1_best.keras')
    
    if not os.path.exists(model_path):
        print(f"Skipping {name}, model not found: {model_path}")
        continue

    model = tf.keras.models.load_model(model_path)
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)

    preds = model.predict(ds, verbose=0)
    if preds.shape[-1] == 1:
        pred_labels = (preds.squeeze() > 0.5).astype(int)
    else:
        pred_labels = np.argmax(preds, axis=1)

    cm = confusion_matrix(true_labels, pred_labels)
    
    # Calculate Precision, Recall, F1 for Defect (class 1)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_d = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_d = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_d = 2 * (precision_d * recall_d) / (precision_d + recall_d) if (precision_d + recall_d) > 0 else 0

    precision_n = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_n = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_n = 2 * (precision_n * recall_n) / (precision_n + recall_n) if (precision_n + recall_n) > 0 else 0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['정상 (Normal)', '불량 (Defect)'], 
                yticklabels=['정상 (Normal)', '불량 (Defect)'],
                annot_kws={'size': 20, 'weight': 'bold'})
    plt.xlabel('예측 라벨', fontsize=12, fontweight='bold')
    plt.ylabel('실제 라벨', fontsize=12, fontweight='bold')
    plt.title(f'{name} 혼돈 행렬  (정확도: {accuracy*100:.0f}%)', fontsize=14, fontweight='bold')
    
    metrics_text = (f"정상 → Precision: {precision_n:.2f}  |  Recall: {recall_n:.2f}  |  F1: {f1_n:.2f}\n"
                    f"불량 → Precision: {precision_d:.2f}  |  Recall: {recall_d:.2f}  |  F1: {f1_d:.2f}")
    plt.figtext(0.5, -0.05, metrics_text, wrap=True, horizontalalignment='center', fontsize=11, color='dimgrey')
    
    out_path = os.path.join(output_dir, f'confusion_{name.lower()}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

print('Done')

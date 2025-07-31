import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

from model import HumanCharacterClassifier # モデルの定義をインポート
from dataset import load_all_data # class_to_idxを取得するためにインポート

# --- ハイパーパラメータ (train.pyと一致させる) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "efficientnet_v2_s" # 訓練時と同じモデル名

# --- 画像変換 (train.pyと一致させる) ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(num_classes, model_path='model.pth'):
    model = HumanCharacterClassifier(num_classes=num_classes, model_name=MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # 評価モードに設定
    print(f"Model loaded from {model_path}")
    return model

def predict_image(image_path, model, transform, class_names):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE) # バッチ次元を追加し、デバイスへ移動

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    predicted_class_idx = predicted.item()
    predicted_class_name = class_names[predicted_class_idx]
    return predicted_class_name

if __name__ == '__main__':
    # 訓練データセットからクラス名を取得
    # 注意: 訓練時と同じroot_dirを指定する必要があります
    _, _, class_to_idx = load_all_data(root_dir='../dataset')
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    # モデルをロード
    model = load_model(num_classes)

    while True:
        image_file = input("予測したい画像ファイルのパスを入力してください (終了するには 'q' を入力): ")
        if image_file.lower() == 'q':
            break
        
        predicted_class_idx, predicted_class_name = predict_image(image_file, model, image_transform, class_names) # type: ignore
        if predicted_class_idx is not None and predicted_class_name is not None:
            print(f"予測結果: {predicted_class_name}")
        print("\n")

    print("予測プログラムを終了します。")

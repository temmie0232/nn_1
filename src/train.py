import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold # type: ignore

# 作成したモジュールをインポート
from dataset import HumanCharacterDataset, load_all_data # load_all_dataもインポート
from model import SimpleCNN

# --- ハイパーパラメータ ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_FOLDS = 5 # K-Foldの分割数

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train() # モデルを訓練モードに
        running_train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)

        # 検証フェーズ
        model.eval() # モデルを評価モードに
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad(): # 勾配計算を無効化
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_predictions

        print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        # 必要に応じて、ここでベストモデルの保存ロジックを追加
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # torch.save(model.state_dict(), 'best_model.pth')
            # print("Best model saved!")

def main():
    # 1. データの前処理を定義
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 全ての画像パスとラベルを一度読み込む
    all_image_paths, all_labels, class_to_idx = load_all_data(root_dir='../dataset')
    num_classes = len(class_to_idx)
    print(f"認識対象クラス数: {num_classes}")
    print(f"全データ数: {len(all_image_paths)}")
    print(f"Using device: {DEVICE}")

    # 3. Stratified K-Fold Cross-Validationの準備
    # ラベルをnumpy配列に変換（StratifiedKFoldのため）
    all_labels_np = np.array(all_labels) 
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # 各フォールドの結果を保存するリスト
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(skf.split(all_image_paths, all_labels_np)):
        print(f"\n--- Fold {fold+1}/{NUM_FOLDS} ---")
        
        # フォールドごとの訓練データと検証データのパスとラベルを準備
        train_fold_image_paths = [all_image_paths[i] for i in train_indices]
        train_fold_labels = [all_labels[i] for i in train_indices]
        val_fold_image_paths = [all_image_paths[i] for i in val_indices]
        val_fold_labels = [all_labels[i] for i in val_indices]

        # フォールドごとのデータセットとデータローダーを作成
        train_dataset = HumanCharacterDataset(train_fold_image_paths, train_fold_labels, transform=image_transform)
        val_dataset = HumanCharacterDataset(val_fold_image_paths, val_fold_labels, transform=image_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"  訓練データ数 (Fold {fold+1}): {len(train_dataset)}")
        print(f"  検証データ数 (Fold {fold+1}): {len(val_dataset)}")

        # モデル、損失関数、オプティマイザを定義 (各フォールドで新しいモデルを初期化)
        model = SimpleCNN(num_classes=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 学習と検証の実行
        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)

        # (オプション) フォールドごとの最終評価指標を収集
        # ここでは簡略化のため、最終エポックの検証精度のみを記録
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        fold_accuracy = 100 * correct_predictions / total_predictions
        print(f"Fold {fold+1} Final Validation Accuracy: {fold_accuracy:.2f}%")
        fold_results.append(fold_accuracy)

    print("\n--- K-Fold Cross-Validation 完了 ---")
    print(f"各フォールドの検証精度: {fold_results}")
    print(f"平均検証精度: {np.mean(fold_results):.2f}%")
    print(f"標準偏差: {np.std(fold_results):.2f}%")

if __name__ == '__main__':
    main()
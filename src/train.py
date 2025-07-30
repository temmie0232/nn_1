import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm

# 作成したモジュールをインポート
from dataset import HumanCharacterDataset
from model import SimpleCNN

# --- ハイパーパラメータ ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def main():
    # 1. データの前処理を定義
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. データセットとデータローダーを準備
    train_dataset = HumanCharacterDataset(root_dir='../dataset', train=True, transform=image_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # (オプション) テストデータも同様に準備
    # test_dataset = HumanCharacterDataset(root_dir='../dataset', train=False, transform=image_transform)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Using device: {DEVICE}")
    print(f"訓練データ数: {len(train_dataset)}")

    # 3. モデル、損失関数、オプティマイザを定義
    model = SimpleCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 学習ループ
    for epoch in range(NUM_EPOCHS):
        model.train() # モデルを訓練モードに
        running_loss = 0.0
        
        # tqdmを使って進捗バーを表示
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            # データをデバイスに送る
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 勾配をリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 逆伝播（勾配計算）
            loss.backward()

            # パラメータ更新
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    print("Finished Training")

    # (オプション) モデルの保存
    # torch.save(model.state_dict(), 'simple_cnn.pth')

if __name__ == '__main__':
    main()
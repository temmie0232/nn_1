import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 224x224x3 の画像を入力とする
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # プーリング層を2回通過後の画像サイズを計算
        # 224 -> 112 -> 56
        # 全結合層への入力サイズ: 32 (チャネル数) * 56 * 56
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # 全結合層への入力のためにテンソルをフラット化
        x = x.view(-1, 32 * 56 * 56)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # モデルの動作確認
    import torch # type: ignore
    model = SimpleCNN(num_classes=10)
    # ダミーの入力データ (バッチサイズ4, 3チャネル, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print("モデルの出力形状:", output.shape) # -> torch.Size([4, 10]) になるはず
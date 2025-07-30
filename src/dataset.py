import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore

class HumanCharacterDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): 全てのクラスの画像が入っているディレクトリへのパス (例: 'dataset/')
            train (bool): Trueなら訓練データ、Falseならテストデータを返す
            transform (callable, optional): 画像に適用する変換
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        # 1. 全ての画像パスを取得し、訓練/テストのルールに従って振り分ける
        all_image_paths = sorted(glob.glob(os.path.join(root_dir, "*", "*.jpg")))

        for img_path in all_image_paths:
            # ファイル名（拡張子なし）を取得 (例: '01-001')
            filename_without_ext = os.path.basename(img_path).split('.')[0]
            
            # ハイフンで分割し、連番部分を取得して整数に変換 (例: '01-001' -> 1)
            file_number = int(filename_without_ext.split('-')[1])
            
            # 訓練/テストの分割ルール
            # テストデータ: 連番が4の倍数
            is_test_sample = (file_number % 4 == 0)

            # このインスタンスが訓練用かテスト用かに基づいてリストに追加
            if self.train and not is_test_sample:
                # 訓練データの場合
                self.image_paths.append(img_path)
                class_name = os.path.basename(os.path.dirname(img_path))
                self.labels.append(self.class_to_idx[class_name])
            elif not self.train and is_test_sample:
                # テストデータの場合
                self.image_paths.append(img_path)
                class_name = os.path.basename(os.path.dirname(img_path))
                self.labels.append(self.class_to_idx[class_name])


    def __len__(self):
        """
        データセットのサンプル数を返す
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        指定されたインデックスのサンプル（画像とラベル）を返す
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 2. 画像を読み込む
        # PILを使ってRGB形式で開くのが一般的
        image = Image.open(img_path).convert("RGB")

        # 3. Transformを適用する
        if self.transform:
            image = self.transform(image)
            
        return image, label

if __name__ == '__main__':
    # このスクリプトを直接実行したときに動作確認するためのコード

    # --- ここからが重要 ---
    # 4. 画像の前処理（Transform）を定義する
    #    - 画像サイズがバラバラなので、全て同じサイズにリサイズする (例: 224x224)
    #    - PyTorchのテンソル形式に変換する
    #    - 画像のピクセル値を正規化する
    # 転移学習を見越して、ImageNetの平均と標準偏差で正規化しておくのが定石
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 画像サイズを統一
        transforms.ToTensor(),         # テンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正規化
    ])

    # 訓練データセットのインスタンスを作成
    train_dataset = HumanCharacterDataset(root_dir='../dataset', train=True, transform=image_transform)
    # テストデータセットのインスタンスを作
    test_dataset = HumanCharacterDataset(root_dir='../dataset', train=False, transform=image_transform)

    print(f"訓練データ数: {len(train_dataset)}")
    print(f"テストデータ数: {len(test_dataset)}")
    
    # DataLoaderを使って、データがバッチで取り出せるか確認
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 1バッチ分のデータを取り出してみる
    images, labels = next(iter(train_loader))
    
    print("\n--- DataLoaderからの出力確認 ---")
    print(f"画像のバッチの形状: {images.shape}") # -> torch.Size([4, 3, 224, 224]) になるはず
    print(f"ラベルのバッチの形状: {labels.shape}") # -> torch.Size([4]) になるはず
    print(f"ラベル: {labels}")
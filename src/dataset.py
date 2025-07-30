import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore

class HumanCharacterDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): 画像ファイルパスのリスト
            labels (list): 各画像に対応するラベルのリスト
            transform (callable, optional): 画像に適用する変換
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

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
        
        # 画像を読み込む
        image = Image.open(img_path).convert("RGB")

        # Transformを適用する
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_all_data(root_dir):
    """
    指定されたルートディレクトリから全ての画像パスとラベルを読み込む
    """
    all_image_paths = sorted(glob.glob(os.path.join(root_dir, "*", "*.jpg")))
    
    # クラス名とインデックスのマッピングを生成
    # 'dataset/01/', 'dataset/02/', ... のようにディレクトリ名がクラス名となる
    class_names = sorted(os.listdir(root_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    all_labels = []
    for img_path in all_image_paths:
        class_name = os.path.basename(os.path.dirname(img_path))
        all_labels.append(class_to_idx[class_name])
            
    return all_image_paths, all_labels, class_to_idx

if __name__ == '__main__':
    # このスクリプトを直接実行したときに動作確認するためのコード

    # 画像の前処理（Transform）を定義する
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 画像サイズを統一
        transforms.ToTensor(),         # テンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正規化
    ])

    # 全ての画像パスとラベルを読み込む
    all_image_paths, all_labels, class_to_idx = load_all_data(root_dir='../dataset')

    # ダミーでデータセットを作成し、DataLoaderで確認
    full_dataset = HumanCharacterDataset(all_image_paths, all_labels, transform=image_transform)

    print(f"全データ数: {len(full_dataset)}")
    
    # DataLoaderを使って、データがバッチで取り出せるか確認
    data_loader = DataLoader(full_dataset, batch_size=4, shuffle=True)
    
    # 1バッチ分のデータを取り出してみる
    images, labels = next(iter(data_loader))
    
    print("\n--- DataLoaderからの出力確認 ---")
    print(f"画像のバッチの形状: {images.shape}") # -> torch.Size([4, 3, 224, 224]) になるはず
    print(f"ラベルのバッチの形状: {labels.shape}") # -> torch.Size([4]) になるはず
    print(f"ラベル: {labels}")
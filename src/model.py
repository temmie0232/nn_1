import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torchvision.models as models # type: ignore

class HumanCharacterClassifier(nn.Module):
    def __init__(self, num_classes=10, model_name="convnext_tiny"):
        super(HumanCharacterClassifier, self).__init__()
        
        if model_name == "convnext_tiny":
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            # ConvNeXt_Tinyの最終層 (classifier) を変更
            # ConvNeXt_Tinyのclassifierはnn.Sequentialでfeaturesとlinear_layerを持つ
            # 最後のnn.Linear層のin_featuresを取得
            num_ftrs = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "efficientnet_v2_s":
            self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            # EfficientNet_V2_Sの最終層 (classifier) を変更
            # EfficientNet_V2_Sのclassifierはnn.Sequentialでdropoutとlinear_layerを持つ
            # 最後のnn.Linear層のin_featuresを取得
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # モデルの動作確認
    # ConvNeXt_Tinyの確認
    model_convnext = HumanCharacterClassifier(num_classes=10, model_name="convnext_tiny")
    dummy_input = torch.randn(4, 3, 224, 224)
    output_convnext = model_convnext(dummy_input)
    print("ConvNeXt_Tinyの出力形状:", output_convnext.shape) # -> torch.Size([4, 10]) になるはず

    # EfficientNet_V2_Sの確認
    model_efficientnet = HumanCharacterClassifier(num_classes=10, model_name="efficientnet_v2_s")
    output_efficientnet = model_efficientnet(dummy_input)
    print("EfficientNet_V2_Sの出力形状:", output_efficientnet.shape) # -> torch.Size([4, 10]) になるはず
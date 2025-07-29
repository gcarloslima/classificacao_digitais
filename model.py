import torch.nn as nn
import timm

class SimpleFingerprintClassifier(nn.Module):
    def __init__(self, num_classes=120):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

"""
Definição da arquitetura do modelo de classificação de impressões digitais.

Este módulo contém a implementação da rede neural convolucional baseada
em EfficientNet para classificação de impressões digitais. O modelo utiliza
transfer learning com pesos pré-treinados do ImageNet.
"""

import torch.nn as nn
import timm


class SimpleFingerprintClassifier(nn.Module):
    """
    Classificador de impressões digitais baseado em EfficientNet.

    Esta classe implementa uma rede neural convolucional para classificação
    de impressões digitais usando EfficientNet-B0 como backbone. O modelo
    utiliza transfer learning e uma camada classificadora personalizada.

    Args:
        num_classes (int): Número de classes para classificação (padrão: 120)
    """

    def __init__(self, num_classes=120):
        """
        Inicializa o modelo de classificação.

        Args:
            num_classes (int): Número de classes de saída (pessoas diferentes)
        """
        super().__init__()
        # Cria o modelo base EfficientNet-B0 com pesos pré-treinados
        self.base_model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        )

        # Extrai as features sem a camada classificadora final
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Define uma nova camada classificadora personalizada
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        """
        Executa o forward pass do modelo.

        Args:
            x (torch.Tensor): Tensor de entrada com shape (batch_size, 3, height, width)

        Returns:
            torch.Tensor: Logits de saída com shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

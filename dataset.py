"""
Módulo de dataset para carregamento de impressões digitais.

Este módulo define a classe FingerprintDataset que herda de torch.utils.data.Dataset
e utiliza ImageFolder para carregar as imagens organizadas em pastas por classe.
"""

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class FingerprintDataset(Dataset):
    """
    Dataset personalizado para impressões digitais.

    Esta classe encapsula o ImageFolder do torchvision para facilitar
    o carregamento de imagens de impressões digitais organizadas em
    diretórios separados por pessoa.

    Args:
        data_dir (str): Caminho para o diretório contendo as pastas de cada pessoa
        transform (callable, optional): Transformações a serem aplicadas nas imagens
    """

    def __init__(self, data_dir, transform=None):
        """
        Inicializa o dataset.

        Args:
            data_dir (str): Diretório raiz contendo subpastas para cada classe
            transform (callable, optional): Transformações para as imagens
        """
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        """
        Retorna o número total de amostras no dataset.

        Returns:
            int: Número de amostras
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Recupera uma amostra pelo índice.

        Args:
            idx (int): Índice da amostra

        Returns:
            tuple: (imagem, label) onde imagem é um tensor e label é um inteiro
        """
        return self.data[idx]

    @property
    def classes(self):
        """
        Retorna a lista de nomes das classes.

        Returns:
            list: Lista com os nomes das classes (nomes das pastas)
        """
        return self.data.classes

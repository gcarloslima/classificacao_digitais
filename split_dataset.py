import os
import shutil
import random

# Caminho para o diretório original
base_dir = 'dataset'

# Pastas de saída
train_dir = 'train'
test_dir = 'test'
valid_dir = 'valid'

# Cria as pastas de destino, se não existirem
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Loop pelas pastas de cada pessoa
for i in range(1, 121):
    person_folder = f"pessoa_{i:05d}"
    source_folder = os.path.join(base_dir, person_folder)

    # Lista todas as imagens .tif
    images = [img for img in os.listdir(source_folder) if img.endswith('.tif')]
    
    # Garante que há pelo menos 8 imagens
    if len(images) < 8:
        print(f"[Aviso] {person_folder} tem menos de 8 imagens. Pulando...")
        continue

    # Embaralha e divide
    random.shuffle(images)
    train_images = images[:6]
    test_images = images[6:7]
    valid_images = images[7:8]
    
    # Cria subpastas nas pastas train/test
    train_person_dir = os.path.join(train_dir, person_folder)
    test_person_dir = os.path.join(test_dir, person_folder)
    valid_person_dir = os.path.join(valid_dir, person_folder)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(test_person_dir, exist_ok=True)
    os.makedirs(valid_person_dir, exist_ok=True)

    # Copia os arquivos
    for img in train_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_person_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(test_person_dir, img))
        
    for img in valid_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(valid_person_dir, img))

print("Divisão concluída com sucesso.")

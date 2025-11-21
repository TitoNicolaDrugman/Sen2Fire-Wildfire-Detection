import json

# Границы сцен
scene2_end = 1457  # Scene1 + Scene2: 0000-1457
scene3_end = 1961  # Scene3: 1458-1961

# Создаем списки case IDs
train_cases = [f"Sen2Fire_{i:04d}" for i in range(0, scene2_end + 1)]
val_cases = [f"Sen2Fire_{i:04d}" for i in range(scene2_end + 1, scene3_end + 1)]

# Структура для nnU-Net v2
splits_data = [{"train": train_cases, "val": val_cases}]

# Сохраняем файл
with open("nnUNet_raw/Dataset001_Sen2Fire/splits_final.json", 'w') as f:
    json.dump(splits_data, f, indent=2)

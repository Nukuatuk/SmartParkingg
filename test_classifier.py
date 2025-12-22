import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd

# === Настройки ===
IMG_SIZE = 128
CLASSES = ["free", "occupied"]
MODEL_PATH = "parking_classifier.pt"
TEST_DIR = r"C:\Users\nukua\OneDrive\Рабочий стол\dataset\test"  # путь к папке с тестовыми изображениями
SAVE_CSV = True            # True — сохранять результаты в CSV

# === Преобразование ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Модель ===
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === Список изображений ===
image_paths = list(Path(TEST_DIR).rglob("*.jpg")) + list(Path(TEST_DIR).rglob("*.png"))

if not image_paths:
    print("⚠️ В папке нет изображений (.jpg/.png)")
    exit()

results = []

# === Предсказания ===
for img_path in image_paths:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
        cls_idx = pred.argmax(1).item()
        cls = CLASSES[cls_idx]
    results.append({"filename": img_path.name, "prediction": cls})
    print(f"{img_path.name:<25} → {cls.upper()}")

# === Сохранение в CSV ===
if SAVE_CSV:
    df = pd.DataFrame(results)
    df.to_csv("test_predictions.csv", index=False, encoding="utf-8-sig")
    print("\n📄 Результаты сохранены в файл: test_predictions.csv")

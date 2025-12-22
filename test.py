import json
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# === Настройки ===
IMAGE_PATH = "3.png"
SLOTS_PATH = "saved_slots.json"
OUTPUT_PATH = "parking_result1.jpg"
MODEL_PATH = "parking_classifier.pt"

PAD = 3  # сдвиг внутрь ROI (0 если не нужно)

# === Загрузка разметки ===
with open(SLOTS_PATH, "r", encoding="utf-8") as f:
    slots_data = json.load(f)

slots = slots_data["slots"]

# Если в json есть image_size — используем его (так правильнее)
if "image_size" in slots_data:
    RESIZE_WIDTH, RESIZE_HEIGHT = slots_data["image_size"]
else:
    # fallback, если в json нет image_size
    RESIZE_WIDTH, RESIZE_HEIGHT = 1280, 720

# === Загрузка изображения ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Файл {IMAGE_PATH} не найден!")

img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

# === Загрузка модели (ВАЖНО: та же архитектура, что при обучении) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 2)
)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# === Transform (RGB => 3 канала) ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

CLASSES = ["free", "occupied"]

# === Обработка каждого слота ===
for s in slots:
    pts = np.array(s["polygon"], np.int32)

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    # небольшой отступ внутрь, чтобы не захватывать линии разметки
    x_min += PAD
    y_min += PAD
    x_max -= PAD
    y_max -= PAD

    # защита от выхода за границы
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)

    # если кроп некорректный
    if x_max <= x_min or y_max <= y_min:
        continue

    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        continue

    # инференс
    im_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = transform(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        cls = CLASSES[pred.argmax(1).item()]

    # визуализация
    color = (0, 255, 0) if cls == "free" else (0, 0, 255)
    cv2.polylines(img, [pts], True, color, 2)
    cv2.putText(
        img,
        f"{s['slot_id']} ({cls})",
        (pts[0][0], max(15, pts[0][1] - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

# === Сохранение результата ===
cv2.imwrite(OUTPUT_PATH, img)
print(f"✅ Результат сохранён в {OUTPUT_PATH}")

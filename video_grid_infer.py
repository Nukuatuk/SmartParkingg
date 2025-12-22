import cv2
import json
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# ================== НАСТРОЙКИ ==================
VIDEO_PATH = "parking.mp4"          # входное видео
OUTPUT_VIDEO = "result.mp4"          # выходное видео (опционально)
SLOTS_FILE = "saved_slots.json"
MODEL_PATH = "parking_classifier.pt"

# =================================================

# === Загрузка разметки ===
slots_data = json.load(open(SLOTS_FILE, "r", encoding="utf-8"))
slots = slots_data["slots"]
W, H = slots_data["image_size"]      # 352x607

# === Модель ===
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

CLASSES = ["free", "occupied"]

# === Видео ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

frame_id = 0

# ================== ОСНОВНОЙ ЦИКЛ ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # 1️⃣ resize под разметку
    frame = cv2.resize(frame, (W, H))

    free, occupied = 0, 0

    # 2️⃣ обработка всех слотов
    for s in slots:
        pts = np.array(s["polygon"], np.int32)

        x_min, y_min = pts[:,0].min(), pts[:,1].min()
        x_max, y_max = pts[:,0].max(), pts[:,1].max()

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        x = transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x).argmax(1).item()

        cls = CLASSES[pred]

        if cls == "free":
            free += 1
            color = (0,255,0)
        else:
            occupied += 1
            color = (0,0,255)

        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, cls,
                    (pts[0][0]+3, pts[0][1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 3️⃣ overlay статистики
    cv2.rectangle(frame, (5,5), (200,70), (0,0,0), -1)
    cv2.putText(frame, f"Total: {len(slots)}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Free: {free}", (10,45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Occupied: {occupied}", (10,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # 4️⃣ вывод / сохранение
    cv2.imshow("Parking AI", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ================== ЗАВЕРШЕНИЕ ==================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Видео обработано и сохранено:", OUTPUT_VIDEO)

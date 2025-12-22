import os
import json
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from torchvision import models, transforms
from PIL import Image
import uvicorn

# === Глобальные настройки ===
SLOTS_FILE = "saved_slots.json"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# === Инициализация приложения ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Загрузка модели ===
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("parking_classifier.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
CLASSES = ["free", "occupied"]


# === Маршруты ===

@app.get("/")
async def root():
    return HTMLResponse(open("static/index.html", "r", encoding="utf-8").read())


@app.get("/annotate")
async def annotate_page():
    return HTMLResponse(open("static/annotate.html", "r", encoding="utf-8").read())


@app.post("/save-slots")
async def save_slots(data: dict = Body(...)):
    """Сохранение разметки в файл"""
    with open(SLOTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"status": "ok", "message": "Разметка сохранена"}


@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    if not os.path.exists(SLOTS_FILE):
        return JSONResponse({"error": "Разметка ещё не создана!"}, status_code=400)

    slots_data = json.load(open(SLOTS_FILE, "r", encoding="utf-8"))
    slots = slots_data["slots"]
    w_expected, h_expected = slots_data["image_size"]

    img_bytes = await image.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (w_expected, h_expected))

    total = len(slots)
    occupied = 0
    free = 0
    slot_results = []

    for s in slots:
        pts = np.array(s["polygon"], np.int32)

        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

        crop = img[y_min:y_max, x_min:x_max]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        x = transform(crop_pil).unsqueeze(0)
        with torch.no_grad():
            pred = model(x)

        cls = CLASSES[pred.argmax(1).item()]
        slot_results.append({"slot_id": s["slot_id"], "status": cls})

        if cls == "free":
            free += 1
        else:
            occupied += 1

        color = (0,255,0) if cls == "free" else (0,0,255)
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, f"{s['slot_id']} ({cls})", (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    percent = round(occupied / total * 100, 1) if total > 0 else 0

    result_path = "result.jpg"
    cv2.imwrite(result_path, img)

    return {
        "image_url": "/result.jpg",
        "total": total,
        "occupied": occupied,
        "free": free,
        "percent": percent,
        "slots": slot_results
    }
@app.get("/result.jpg")
async def get_result_image():
    if not os.path.exists("result.jpg"):
        return JSONResponse({"error": "Файл результата ещё не создан"}, status_code=404)
    return FileResponse("result.jpg")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

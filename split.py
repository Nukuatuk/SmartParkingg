import cv2
import os

# ===== НАСТРОЙКИ =====
COLLAGE_PATH = "collage.png"   # имя файла с коллажем
OUT_DIR = "images"
ROWS = 3    # сколько рядов в коллаже
COLS = 3   # сколько колонок в коллаже
# =====================

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(COLLAGE_PATH)
h, w = img.shape[:2]

tile_h = h // ROWS
tile_w = w // COLS

idx = 1
for r in range(ROWS):
    for c in range(COLS):
        crop = img[
            r*tile_h:(r+1)*tile_h,
            c*tile_w:(c+1)*tile_w
        ]
        cv2.imwrite(f"{OUT_DIR}/img_{idx:02d}.jpg", crop)
        idx += 1

print("✅ Готово! Картинки сохранены в папку images/")

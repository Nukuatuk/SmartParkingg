import cv2
import json
import numpy as np

# ============ НАСТРОЙКИ ============
IMAGE_PATH = "0b4b0c8e-eaf5-462a-8ee2-4abb08abba61.png"   # кадр из видео
OUTPUT_JSON = "saved_slots.json"
# ==================================

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Не найден файл {IMAGE_PATH}")

H, W = img.shape[:2]
clone = img.copy()

slots = []
points = []
slot_id = 1
cursor = (0, 0)

def redraw():
    global img
    img = clone.copy()

    # сохранённые слоты
    for s in slots:
        pts = np.array(s["polygon"], np.int32)
        cv2.polylines(img, [pts], True, (0,255,255), 2)
        cv2.putText(
            img, s["slot_id"],
            (pts[0][0]+3, pts[0][1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2
        )

    # текущие точки
    for p in points:
        cv2.circle(img, p, 4, (0,255,0), -1)

    # координатный крест
    x, y = cursor
    cv2.line(img, (x,0), (x,W), (80,80,80), 1)
    cv2.line(img, (0,y), (H,y), (80,80,80), 1)

def mouse(event, x, y, flags, param):
    global points, slot_id, cursor

    if event == cv2.EVENT_MOUSEMOVE:
        cursor = (x,y)
        redraw()
        cv2.imshow("Annotate", img)

    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        redraw()
        print(f"Точка: ({x},{y})")

        if len(points) == 4:
            slots.append({
                "slot_id": f"S{slot_id:02d}",
                "polygon": points.copy()
            })
            print(f"✅ Добавлен слот S{slot_id:02d}")
            slot_id += 1
            points.clear()
            redraw()
            cv2.imshow("Annotate", img)

    elif event == cv2.EVENT_MBUTTONDOWN:
        points.clear()
        redraw()
        cv2.imshow("Annotate", img)
        print("↩ Отмена текущего слота")

    elif event == cv2.EVENT_RBUTTONDOWN:
        data = {
            "image_size": [W, H],
            "slots": slots
        }
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Разметка сохранена: {OUTPUT_JSON}")
        cv2.destroyAllWindows()

# ============ ЗАПУСК ============
cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
cv2.imshow("Annotate", img)
cv2.setMouseCallback("Annotate", mouse)

print("\n🖱 ИНСТРУКЦИЯ:")
print("ЛКМ — поставить точку (4 точки = 1 слот)")
print("СКМ — отменить текущий слот")
print("ПКМ — сохранить и выйти")
print("Крест — помогает точно выбрать координаты\n")

cv2.waitKey(0)
cv2.destroyAllWindows()

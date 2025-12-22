import cv2

VIDEO_PATH = "parking.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

# можно взять, например, 10-й кадр
cap.set(cv2.CAP_PROP_POS_FRAMES, 10)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Не удалось считать кадр")

cv2.imwrite("reference_from_video.jpg", frame)
print("✅ Референс-кадр сохранён: reference_from_video.jpg")

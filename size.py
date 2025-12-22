import cv2, json, numpy as np

img = cv2.imread("reference_from_video.jpg")
data = json.load(open("saved_slots.json"))
for s in data["slots"]:
    pts = np.array(s["polygon"], np.int32)
    cv2.polylines(img, [pts], True, (0,255,0), 2)
    cv2.putText(img, s["slot_id"],
                (pts[0][0]+3, pts[0][1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imshow("check", img)
cv2.waitKey(0)

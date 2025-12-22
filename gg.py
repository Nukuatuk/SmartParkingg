import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np

# === Путь к тестовому датасету ===
TEST_DIR = r"C:\Users\nukua\OneDrive\Рабочий стол\dataset\test"

IMG_SIZE = 128
BATCH_SIZE = 32

# === Трансформация ===
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Датасет ===
test_data = datasets.ImageFolder(TEST_DIR, transform=test_tf)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# === Загрузка модели ===
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("parking_classifier.pt", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === TEST ===
y_true = []
y_pred = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === Confusion Matrix ===
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("\n📌 Итоговая проверка модели на тестовом наборе:")
print(f"Всего изображений: {len(test_data)}")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")

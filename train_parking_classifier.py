import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================== НАСТРОЙКИ ==================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4

TRAIN_DIR = r"C:\Users\nukua\OneDrive\Рабочий стол\dataset\train"
VAL_DIR   = r"C:\Users\nukua\OneDrive\Рабочий стол\dataset\val"

MODEL_PATH = "parking_classifier.pt"
# =================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)

# ================== АУГМЕНТАЦИИ ==================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_data   = datasets.ImageFolder(VAL_DIR, transform=val_tf)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print("Классы:", train_data.classes)

# ================== МОДЕЛЬ ==================
model = models.resnet18(pretrained=True)

# 🔥 Fully Connected с Dropout (против переобучения)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),          # ← Dropout
    nn.Linear(256, 2)
)

model = model.to(device)

# ================== ОБУЧЕНИЕ ==================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(
        f"Train loss: {train_loss/len(train_loader):.4f} | "
        f"Train acc: {train_acc:.3f} | "
        f"Val acc: {val_acc:.3f}"
    )

    # -------- SAVE BEST MODEL --------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Лучшая модель сохранена")

print("\n✅ Обучение завершено")
print(f"Лучшая val_acc = {best_val_acc:.3f}")
print(f"Модель сохранена в {MODEL_PATH}")

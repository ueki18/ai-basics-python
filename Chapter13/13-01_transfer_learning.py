import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------
# データの読み込み
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    "hymenoptera_data/train", transform=transform
)
val_dataset = datasets.ImageFolder(
    "hymenoptera_data/val", transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ---------------------------
# モデル（ResNet-18）
# ---------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 特徴抽出部分の重みをすべて固定
for param in model.parameters():
    param.requires_grad = False

# 最終層を変更（2クラス）
model.fc = nn.Linear(model.fc.in_features, 2)

# ---------------------------
# 学習設定
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ---------------------------
# 学習
# ---------------------------
for epoch in range(3):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---------------------------
# 評価
# ---------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", correct / total)

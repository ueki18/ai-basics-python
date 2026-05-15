import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------
# データの準備（MNIST）
# ---------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ---------------------------
# モデル
# ---------------------------
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*13*13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()

# ---------------------------
# 損失関数と最適化
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 学習
# ---------------------------
for epoch in range(3):

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # 予測
        outputs = model(images)

        # 損失計算
        loss = criterion(outputs, labels)

        # 勾配リセット
        optimizer.zero_grad()

        # 逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()

        # lossを加算
        running_loss += loss.item()

        # 精度計算
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # epochごとの平均loss
    epoch_loss = running_loss / len(train_loader)

    # epochごとの精度
    epoch_accuracy = correct / total

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {epoch_loss:.4f}, "
        f"Accuracy: {epoch_accuracy:.4f}"
    )

# ---------------------------
# 評価
# ---------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy:", correct / total)

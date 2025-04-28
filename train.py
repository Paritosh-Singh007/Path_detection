import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNModel
from data_loader import get_dataloaders

# Hyperparameters
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, val_loader, classes = get_dataloaders(batch_size=batch_size)

# ⚡ Load model with updated number of classes
model = CNNModel(num_classes=len(classes)).to(device)  # ⬅️ UPDATED

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "cnn_model.pth")
print("✅ Model saved as cnn_model.pth")

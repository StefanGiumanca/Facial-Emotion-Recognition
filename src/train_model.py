import os
import ssl
import certifi
ssl._create_default_https_context =  lambda : ssl.create_default_context(cafile=certifi.where())

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader

# Checking the CPU/GPU the user has
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print('Using device: CPU')

train_dir = "../data/train"
test_dir = "../data/test"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

train_dataset = datasets.ImageFolder(train_dir, transform = transform)
test_dataset = datasets.ImageFolder(test_dir, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

weights = MobileNet_V2_Weights.DEFAULT # loading preTrained model
model = models.mobilenet_v2(weights = weights)

for param in model.parameters(): # Freeze the neural network (Pre-trained model)
    param.requires_grad = False

for param in model.features[-5:].parameters(): # Unfreeze the last 5 layers for project scoupe
    param.requires_grad = True
    
num_classes = len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)

# Loss function and optimizer
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0], dtype=torch.float32)
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr = 0.0003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

checkpoint_path = "../models/emotion_model_best.pt"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Loaded existing best model checkpoint.")

print(train_dataset.classes)
# Training the model
epochs = 10
best_acc = 0.0
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if (batch_idx + 1) % 10 == 0:
            print(f"  [Batch {batch_idx+1}/{len(train_loader)}] loading...")

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    if epoch == 0 or acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../models/emotion_model_best.pt")
        print("Saved improved model.")
    
    print(f'[Epoch {epoch+1}/{epochs}] Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%')
    scheduler.step()    

torch.save(model.state_dict(), "../models/emotion_model.pt")
print("Model saved to models/emotion_model.pt")


        
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

TRAIN_DIR="dataset/train"
VAL_DIR="dataset/val"
MODEL_DIR="models"
MODEL_PATH=os.path.join(MODEL_DIR,"liveness_model.pth")

BATCH_SIZE=8
EPOCHS=20
LR=0.001

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using",device)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset=datasets.ImageFolder(TRAIN_DIR,transform=transform)
val_dataset=datasets.ImageFolder(VAL_DIR,transform=transform)

train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3]=nn.Linear(model.classifier[3].in_features,2)
model=model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=LR)

os.makedirs(MODEL_DIR,exist_ok=True)
best_acc=0

for epoch in range(EPOCHS):
    model.train()
    train_correct=train_total=0
    train_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        _,pred=torch.max(outputs,1)
        train_correct+=(pred==labels).sum().item()
        train_total+=labels.size(0)

    model.eval()
    val_correct=val_total=0
    val_loss=0
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            val_loss+=loss.item()
            _,pred=torch.max(outputs,1)
            val_correct+=(pred==labels).sum().item()
            val_total+=labels.size(0)

    train_acc=100*train_correct/train_total
    val_acc=100*val_correct/val_total

    print(f"Epoch {epoch+1}/{EPOCHS} Train:{train_acc:.2f}% Val:{val_acc:.2f}%")

    if val_acc>best_acc:
        best_acc=val_acc
        torch.save(model.state_dict(),MODEL_PATH)
        print("Model saved:",MODEL_PATH)

print("Training completed.")
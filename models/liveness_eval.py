import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TEST_DIR="dataset/test"
MODEL_PATH="models/liveness_model.pth"
OUTPUT_DIR="outputs"
os.makedirs(OUTPUT_DIR,exist_ok=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_dataset=datasets.ImageFolder(TEST_DIR,transform=transform)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)

model=models.mobilenet_v3_small(weights=None)
model.classifier[3]=nn.Linear(model.classifier[3].in_features,2)
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.to(device)
model.eval()

live_scores=[]
spoof_scores=[]

with torch.no_grad():
    for img,label in test_loader:
        img=img.to(device)
        out=model(img)
        prob=torch.softmax(out,dim=1)[0,0].item()  # probability LIVE
        if label.item()==0:
            live_scores.append(prob)
        else:
            spoof_scores.append(prob)

thresholds=np.arange(0,1.01,0.01)
apcers=[]
bpcers=[]
best_eer=1
eer_thr=0

for t in thresholds:
    bpcer=sum(s<t for s in live_scores)/max(len(live_scores),1)
    apcer=sum(s>=t for s in spoof_scores)/max(len(spoof_scores),1)
    apcers.append(apcer)
    bpcers.append(bpcer)
    if abs(apcer-bpcer)<best_eer:
        best_eer=abs(apcer-bpcer)
        eer=(apcer+bpcer)/2
        eer_thr=t

target=0.03
idx=min(range(len(thresholds)),key=lambda i:abs(bpcers[i]-target))
print(f"Threshold @ BPCER≈3%: {thresholds[idx]:.2f}")
print(f"APCER: {apcers[idx]*100:.2f}%")
print(f"BPCER: {bpcers[idx]*100:.2f}%")
print(f"EER: {eer*100:.2f}% at threshold {eer_thr:.2f}")

plt.figure(figsize=(6,4))
plt.hist(live_scores,bins=10,alpha=0.6,label="LIVE")
plt.hist(spoof_scores,bins=10,alpha=0.6,label="SPOOF")
plt.axvline(eer_thr,linestyle="--")
plt.legend()
plt.xlabel("LIVE Probability")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"score_distribution.png"))
plt.close()

plt.figure(figsize=(5,5))
plt.plot(bpcers,apcers)
plt.xlabel("BPCER")
plt.ylabel("APCER")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"apcer_bpcer_curve.png"))
plt.close()

print("Saved:")
print(os.path.join(OUTPUT_DIR,"score_distribution.png"))
print(os.path.join(OUTPUT_DIR,"apcer_bpcer_curve.png"))